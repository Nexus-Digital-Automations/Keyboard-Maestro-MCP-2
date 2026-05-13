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

_VERIFY_TIMEOUT_SECONDS = 10.0
_VERIFY_POLL_INTERVAL = 0.25


def _new_macro_uid() -> str:
    return str(uuid.uuid4()).upper()


def build_kmmacros_plist(
    group_name: str,
    group_uid: str,
    macro_name: str,
    macro_uid: str,
) -> bytes:
    """Build the minimal XML .kmmacros plist for one empty macro.

    KM matches the import target by ``UID`` of the group dict; ``Name`` is
    only used when KM creates a brand-new group, so we always set both to
    keep imports idempotent against the live group.
    """
    document = [
        {
            "Activate": "Normal",
            "KeyCount": 0,
            "Macros": [
                {
                    "Actions": [],
                    "CreationDate": 0,
                    "ModificationDate": 0,
                    "Name": macro_name,
                    "Triggers": [],
                    "UID": macro_uid,
                },
            ],
            "Name": group_name,
            "UID": group_uid,
        },
    ]
    return plistlib.dumps(document, fmt=plistlib.FMT_XML)


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
    """Poll list_macros until ``macro_uid`` is visible or we time out."""
    target = macro_uid.lower()
    deadline = asyncio.get_event_loop().time() + _VERIFY_TIMEOUT_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        listing = await asyncio.get_event_loop().run_in_executor(
            None, km_client.list_macros,
        )
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
) -> Either[KMError, dict[str, Any]]:
    """Import an empty macro called ``new_name`` into the group ``group_id``.

    Returns ``{"macro_id", "name", "group_id", "group_name"}`` on success.
    """
    resolved = await _resolve_group(km_client, group_id)
    if resolved.is_left():
        return Either.left(resolved.get_left())
    group_name, group_uid = resolved.get_right()

    macro_uid = _new_macro_uid()
    plist_bytes = build_kmmacros_plist(group_name, group_uid, new_name, macro_uid)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".kmmacros", delete=False, prefix="km_mcp_create_",
    )
    try:
        tmp.write(plist_bytes)
        tmp.close()
        import_result = await _ask_km_to_import(km_client, tmp.name)
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
            os.unlink(tmp.name)
        except OSError:
            logger.warning("Failed to unlink temp .kmmacros file %s", tmp.name)
