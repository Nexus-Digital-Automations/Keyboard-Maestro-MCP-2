"""Macro plist rebuild primitives — export-edit-reimport pipeline.

@stable Owns the only known path for structural mutation of a KM 11
macro's plist (Triggers, Actions, Name). AppleScript exposes
``set xml of trigger`` but rejects ``set xml of macro`` (error -10006,
read-only property — verified by Phase 0 probe on 2026-05-14). The
pipeline therefore: fetch xml → mutate → mint fresh UID → delete
original → import .kmmacros → poll for new UID.

DECISION RECORD (2026-05-14): UID rotation is unavoidable. Callers that
hold cross-macro references by UID must use the returned
``old_uid → new_uid`` mapping to rewrite them.

The helpers below originally lived in ``server.tools.system_trigger_tools``
where they only served ``km_add_system_trigger``. Hoisting them here lets
``km_set_macro_triggers`` and any future plist-mutation tool share one
pipeline rather than duplicating it.
"""
from __future__ import annotations

import asyncio
import os
import plistlib
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from src.core.either import Either
from src.core.logging import get_logger
from src.core.types import MacroId
from src.integration.km_client import KMError

logger = get_logger(__name__)

_VERIFY_TIMEOUT_SECONDS = 5.0
_VERIFY_POLL_INTERVAL = 0.25


@dataclass(frozen=True)
class MacroSnapshot:
    """Parsed macro plist captured at one point in time, plus its group context."""

    plist: dict[str, Any]
    group_name: str
    group_uid: str
    original_uid: str


@dataclass(frozen=True)
class RebuildResult:
    """Outcome of a successful rebuild. ``old_uid != new_uid`` always (Path B)."""

    old_uid: str
    new_uid: str
    group_name: str


async def fetch_macro_snapshot(
    km: Any,
    macro_id: str,
) -> Either[KMError, MacroSnapshot]:
    """Read the macro's full plist + containing-group context.

    Failure modes:
    - NOT_FOUND_ERROR: macro_id doesn't resolve.
    - EXECUTION_ERROR: AppleScript failed, or KM returned malformed XML.
    """
    selector = km._macro_selector(macro_id.strip())  # noqa: SLF001 — KMClient sibling
    script = f"""
    tell application "Keyboard Maestro"
        try
            if not (exists {selector}) then
                return "ERROR: macro not found"
            end if
            set m to {selector}
            set macroXml to xml of m
            set g to macro group of m
            set gName to name of g
            set gUid to id of g
            return macroXml & "###SEP###" & gName & "###SEP###" & gUid
        on error errMsg
            return "ERROR: " & errMsg
        end try
    end tell
    """
    result = await km.execute_applescript_async(script)
    if result.is_left():
        return Either.left(result.get_left())
    output = result.get_right()
    if output.startswith("ERROR:"):
        msg = output[6:].strip()
        if "not found" in msg.lower():
            return Either.left(KMError.not_found_error(msg))
        return Either.left(KMError.execution_error(msg))
    parts = output.split("###SEP###")
    if len(parts) < 3:
        return Either.left(KMError.execution_error("fetch xml returned unexpected format"))
    macro_xml, group_name, group_uid = parts[0].strip(), parts[1].strip(), parts[2].strip()
    try:
        plist = plistlib.loads(macro_xml.encode("utf-8"))
    except (plistlib.InvalidFileException, ValueError) as exc:
        return Either.left(KMError.execution_error(f"plist parse failed: {exc}"))
    original_uid = str(plist.get("UID", macro_id.strip()))
    return Either.right(
        MacroSnapshot(
            plist=plist,
            group_name=group_name,
            group_uid=group_uid,
            original_uid=original_uid,
        ),
    )


async def rebuild_macro_via_reimport(
    km: Any,
    snapshot: MacroSnapshot,
    new_plist: dict[str, Any],
) -> Either[KMError, RebuildResult]:
    """Re-import the macro with ``new_plist``; mint a fresh UID and return both.

    Caller has full control over ``new_plist`` — Triggers, Actions, Name, etc.
    can all be different from ``snapshot.plist``. The UID field is overwritten
    here regardless of what the caller passed.

    Failure modes:
    - EXECUTION_ERROR: delete-before-reimport rejected, import rejected, or
      the new UID never surfaced in KM's macro list within the verify timeout.
    """
    fresh_uid = str(uuid.uuid4()).upper()
    rebuilt = dict(new_plist)
    rebuilt["UID"] = fresh_uid
    document_bytes = _wrap_kmmacros(rebuilt, snapshot.group_name, snapshot.group_uid)

    delete_result = await km.delete_macro_async(MacroId(snapshot.original_uid))
    if delete_result.is_left():
        err = delete_result.get_left()
        logger.warning(
            "rebuild_macro: pre-import delete failed macro=%s err=%s",
            snapshot.original_uid,
            err.message,
        )
        return Either.left(KMError.execution_error(f"pre-import delete failed: {err.message}"))

    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — KM opens it after we close
        suffix=".kmmacros",
        delete=False,
        prefix="km_mcp_rebuild_",
    )
    try:
        tmp.write(document_bytes)
        tmp.close()
        import_err = await _open_kmmacros(km, tmp.name)
        if import_err is not None:
            logger.warning("rebuild_macro: import rejected macro=%s err=%s", snapshot.original_uid, import_err)
            return Either.left(KMError.execution_error(f"KM rejected rebuilt .kmmacros: {import_err}"))
        if not await _await_uid_visible(km, fresh_uid):
            logger.warning("rebuild_macro: new UID not visible after %.1fs uid=%s", _VERIFY_TIMEOUT_SECONDS, fresh_uid)
            return Either.left(
                KMError.execution_error(
                    f"Imported macro but UID {fresh_uid} did not appear within "
                    f"{_VERIFY_TIMEOUT_SECONDS:.1f}s",
                ),
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            logger.warning("rebuild_macro: failed to unlink temp .kmmacros %s", tmp.name)

    return Either.right(
        RebuildResult(
            old_uid=snapshot.original_uid,
            new_uid=fresh_uid,
            group_name=snapshot.group_name,
        ),
    )


def _wrap_kmmacros(macro_plist: dict[str, Any], group_name: str, group_uid: str) -> bytes:
    document = [
        {
            "Activate": "Normal",
            "KeyCount": 0,
            "Macros": [macro_plist],
            "Name": group_name,
            "UID": group_uid,
        },
    ]
    return plistlib.dumps(document, fmt=plistlib.FMT_XML)


async def _open_kmmacros(km: Any, path: str) -> str | None:
    escaped = path.replace('"', '\\"')
    script = (
        'tell application "Keyboard Maestro" to '
        f'open POSIX file "{escaped}"'
    )
    result = await km.execute_applescript_async(script)
    if result.is_left():
        return str(result.get_left().message)
    return None


async def _await_uid_visible(km: Any, uid: str) -> bool:
    target = uid.lower()
    deadline = asyncio.get_event_loop().time() + _VERIFY_TIMEOUT_SECONDS
    while asyncio.get_event_loop().time() < deadline:
        listing = await km.list_macros_async(enabled_only=False)
        if listing.is_right():
            for macro in listing.get_right():
                if str(macro.get("id", "")).lower() == target:
                    return True
        await asyncio.sleep(_VERIFY_POLL_INTERVAL)
    return False
