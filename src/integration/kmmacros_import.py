"""Create new Keyboard Maestro macros via .kmmacros plist import.

Owner: Keyboard-Maestro-MCP-2 server tools (km_macro_editor.create,
km_create_macro).

Why this exists: Keyboard Maestro 11's AppleScript dictionary has no
``make new macro`` verb. The shipping workaround is to write a minimal
.kmmacros plist to disk and have the Editor open it, which imports the
macro into the named group. This module is the single home for that
workaround so both creation entry points share one tested path.

State diagram (per call):
    resolve group -> build plist -> write tempfile -> KM imports file ->
    poll list_macros for new UID -> unlink tempfile -> return result

Failure modes (all return Either.left(KMError)):
    - GROUP_NOT_FOUND: ``group_id`` does not match any group by name/UID.
    - IMPORT_FAILED: osascript returned non-zero, OR the new UID never
      appeared in ``list_macros`` within the verify window.
    - Underlying ``execute_applescript_async`` failures propagate.
"""

from __future__ import annotations

import asyncio
import os
import plistlib
import tempfile
import uuid
from typing import TYPE_CHECKING, Any

from src.core.either import Either
from src.core.logging import get_logger
from src.integration.km_client import KMError

if TYPE_CHECKING:
    from src.integration.km_client import KMClient

logger = get_logger(__name__)

_VERIFY_TIMEOUT_SECONDS = 5.0
_VERIFY_POLL_INTERVAL = 0.25


def _new_macro_uid() -> str:
    return str(uuid.uuid4()).upper()


def build_kmmacros_plist(
    group_name: str,
    group_uid: str,
    macro_name: str,
    macro_uid: str,
    actions_xml: str = "",
    triggers_xml: str = "",
) -> bytes:
    """Build the minimal XML .kmmacros plist for one macro.

    KM matches the import target by ``UID`` of the group dict; ``Name`` is
    only used when KM creates a brand-new group, so we always set both to
    keep imports idempotent against the live group.

    ``actions_xml`` is a pre-rendered concatenation of action ``<dict>``
    strings (from ``_build_action_xml``). Each is parsed through plistlib
    to a dict and embedded in the macro's ``Actions`` array. Roundtripping
    through plistlib normalizes whitespace and key ordering, but the
    emitter outputs are already KM-canonical (verified by KM-author probe)
    so the parsed dicts re-serialize to equivalent plist.

    ``triggers_xml`` follows the same shape — a concatenation of trigger
    ``<dict>`` strings — and is embedded in the macro's ``Triggers``
    array. Embedding triggers at import time is the only race-free way to
    attach a hotkey to a freshly-created macro; the post-import
    ``make new trigger`` AppleScript path is rejected by KM 11's strict
    dictionary when the macro's group hasn't fully activated yet.
    """
    actions = _parse_action_dicts(actions_xml)
    triggers = _parse_action_dicts(triggers_xml)
    document = [
        {
            "Activate": "Normal",
            "KeyCount": 0,
            "Macros": [
                {
                    "Actions": actions,
                    "CreationDate": 0,
                    "ModificationDate": 0,
                    "Name": macro_name,
                    "Triggers": triggers,
                    "UID": macro_uid,
                },
            ],
            "Name": group_name,
            "UID": group_uid,
        },
    ]
    return plistlib.dumps(document, fmt=plistlib.FMT_XML)


def _parse_action_dicts(actions_xml: str) -> list[dict[str, Any]]:
    """Parse a concatenation of ``<dict>...</dict>`` strings into plist dicts.

    Wraps the concatenation in a plist ``<array>`` envelope so plistlib can
    parse it as a single document. Returns ``[]`` for empty input.
    """
    stripped = actions_xml.strip()
    if not stripped:
        return []
    envelope = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
        f'<plist version="1.0"><array>{stripped}</array></plist>'
    )
    return plistlib.loads(envelope.encode("utf-8"))


async def _resolve_group(
    km_client: KMClient,
    group_id: str,
) -> Either[KMError, tuple[str, str]]:
    """Return (group_name, group_uid) for a caller-supplied name or UID."""
    groups_result = await km_client.list_groups_async()
    if groups_result.is_left():
        return Either.left(groups_result.get_left())

    needle = group_id.strip().lower()
    for entry in groups_result.get_right():
        name = entry.get("groupName", "")
        uid = entry.get("groupID", "")
        if name.lower() == needle or uid.lower() == needle:
            return Either.right((name, uid))

    return Either.left(
        KMError.execution_error(f"Group not found: {group_id!r}"),
    )


async def _ask_km_to_import(
    km_client: KMClient,
    path: str,
) -> Either[KMError, None]:
    escaped = path.replace('"', '\\"')
    script = (
        'tell application "Keyboard Maestro" to '
        f'open POSIX file "{escaped}"'
    )
    result = await km_client.execute_applescript_async(script)
    if result.is_left():
        return Either.left(result.get_left())
    return Either.right(None)


async def _wait_for_macro_uid(
    km_client: KMClient,
    macro_uid: str,
) -> bool:
    """Poll list_macros_async until ``macro_uid`` is visible or we time out.

    Uses the async path with ``enabled_only=False``: KM imports macros in the
    disabled state, and ``KMClient.list_macros`` (sync) routes through an
    AppleScript command that has no ``list_macros`` handler, so the right path
    is the async one that calls ``_list_macros_applescript`` directly.
    """
    target = macro_uid.lower()
    deadline = asyncio.get_event_loop().time() + _VERIFY_TIMEOUT_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        listing = await km_client.list_macros_async(enabled_only=False)
        if listing.is_right():
            for macro in listing.get_right():
                if str(macro.get("id", "")).lower() == target:
                    return True
        await asyncio.sleep(_VERIFY_POLL_INTERVAL)
    return False


async def create_empty_macro(
    km_client: KMClient,
    group_id: str,
    new_name: str,
    actions_xml: str = "",
    triggers_xml: str = "",
) -> Either[KMError, dict[str, Any]]:
    """Import a macro called ``new_name`` into the group ``group_id``.

    ``actions_xml`` (optional): pre-rendered concatenation of action
    ``<dict>`` strings to embed in the macro at creation time. Empty by
    default — kept for backward compatibility with callers that wanted
    just an empty macro to fill via subsequent ``km_action_builder.append``.

    ``triggers_xml`` (optional): pre-rendered concatenation of trigger
    ``<dict>`` strings to embed at the same time. Required for hotkey
    templates because the post-import ``make new trigger`` path races
    against KM's macro-group activation and silently no-ops.

    Returns ``{"macro_id", "name", "group_id", "group_name"}`` on success.
    """
    resolved = await _resolve_group(km_client, group_id)
    if resolved.is_left():
        return Either.left(resolved.get_left())
    group_name, group_uid = resolved.get_right()

    macro_uid = _new_macro_uid()
    plist_bytes = build_kmmacros_plist(
        group_name, group_uid, new_name, macro_uid, actions_xml, triggers_xml,
    )

    # delete=False because KM imports the file by path; we unlink in finally.
    with tempfile.NamedTemporaryFile(
        suffix=".kmmacros", delete=False, prefix="km_mcp_create_",
    ) as tmp:
        tmp.write(plist_bytes)
        tmp_path = tmp.name
    try:
        import_result = await _ask_km_to_import(km_client, tmp_path)
        if import_result.is_left():
            logger.error("kmmacros import failed for %s in %s", new_name, group_name)
            return Either.left(import_result.get_left())

        if not await _wait_for_macro_uid(km_client, macro_uid):
            logger.error("kmmacros import succeeded but UID %s never appeared", macro_uid)
            return Either.left(
                KMError.execution_error(
                    f"Imported {new_name!r} but UID {macro_uid} was not visible "
                    f"in KM within {_VERIFY_TIMEOUT_SECONDS:.1f}s",
                ),
            )

        return Either.right(
            {
                "macro_id": macro_uid,
                "name": new_name,
                "group_id": group_uid,
                "group_name": group_name,
            },
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Failed to unlink temp .kmmacros file %s", tmp_path)
