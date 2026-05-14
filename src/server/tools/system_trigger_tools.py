"""Attach a system-event trigger to a macro via .kmmacros re-import.

@stable km_add_system_trigger uses an export-edit-reimport pipeline
because KM 11 AppleScript rejects ``make new trigger with properties
{xml:...}`` for any plist whose MacroTriggerType is not ``HotKey``.

Pipeline (one call):
    1. Fetch macro xml + group context via AppleScript.
    2. Parse the macro <dict> plist.
    3. Append the requested trigger dict to ``Triggers``.
    4. Mint a fresh macro UID (KM's import would prompt-and-block if we
       reused the old UID).
    5. Delete the original macro.
    6. Write a temp .kmmacros file with the modified macro and ask KM
       to open it.
    7. Poll list_macros until the new UID is visible.

Trade-off (decision record): the macro UUID changes. Other macros that
reference this one by UID via ExecuteMacro must be rewritten by the
caller. The response surfaces ``old_macro_id`` and ``new_macro_id`` so
the caller can do that.

Failure modes:
- VALIDATION_ERROR: bad trigger_kind or missing macro_id.
- NOT_FOUND: the source macro does not exist.
- EXPORT_FAILED: AppleScript couldn't return the macro xml.
- IMPORT_FAILED: kmmacros file written but KM didn't surface the new UID.
"""

from __future__ import annotations

import asyncio
import os
import plistlib
import tempfile
import uuid
from typing import Annotated, Any, Literal

from fastmcp import Context  # noqa: TC002 — pydantic Field reads annotation at runtime
from pydantic import Field

from src.core.logging import get_logger
from src.core.types import MacroId
from src.server.initialization import get_km_client

logger = get_logger(__name__)

_VERIFY_TIMEOUT_SECONDS = 5.0
_VERIFY_POLL_INTERVAL = 0.25

# Trigger plist values: confirmed against KM 11 Engine binary strings.
_TRIGGER_PLIST = {
    "login": {"MacroTriggerType": "Login"},
    "engine_launch": {"MacroTriggerType": "EngineLaunch"},
    "system_wake": {"MacroTriggerType": "WakeTrigger"},
}


async def km_add_system_trigger(
    macro_id: Annotated[
        str,
        Field(description="Target macro UUID or name.", min_length=1, max_length=255),
    ],
    trigger_kind: Annotated[
        Literal["login", "engine_launch", "system_wake"],
        Field(
            description=(
                "'login' fires when the user logs in; 'engine_launch' fires "
                "when the KM Engine starts; 'system_wake' fires when the Mac "
                "wakes from sleep."
            ),
        ),
    ],
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Attach a system-event trigger to ``macro_id``.

    Returns a success envelope with the **new** macro UID (export-edit-
    reimport invalidates the old UID). Use the ``old_macro_id`` →
    ``new_macro_id`` mapping to fix cross-macro references.
    """
    if trigger_kind not in _TRIGGER_PLIST:
        return _fail(
            "VALIDATION_ERROR",
            f"trigger_kind {trigger_kind!r} not supported. "
            f"Pass one of {sorted(_TRIGGER_PLIST)}.",
            "See km_add_system_trigger docstring.",
        )

    km = get_km_client()
    fetched = await _fetch_macro_xml(km, macro_id)
    if fetched is None:
        return _fail(
            "NOT_FOUND",
            f"macro {macro_id!r} not found or KM refused to return its xml.",
            "Verify the UID/name exists via km_list_macros.",
        )
    macro_plist_xml, group_name, group_uid = fetched

    try:
        macro_dict = plistlib.loads(macro_plist_xml.encode("utf-8"))
    except (plistlib.InvalidFileException, ValueError) as exc:
        logger.warning("plist parse failed for macro=%s err=%s", macro_id, exc)
        return _fail(
            "EXPORT_FAILED",
            f"Could not parse macro plist: {exc}",
            "File a bug with the failing macro UID.",
        )

    macro_dict.setdefault("Triggers", []).append(dict(_TRIGGER_PLIST[trigger_kind]))
    new_uid = str(uuid.uuid4()).upper()
    old_uid = macro_dict.get("UID", macro_id)
    macro_dict["UID"] = new_uid

    kmmacros_bytes = _wrap_in_kmmacros_doc(macro_dict, group_name, group_uid)

    delete_result = await km.delete_macro_async(MacroId(macro_id.strip()))
    if delete_result.is_left():
        err = delete_result.get_left()
        logger.warning("delete before reimport failed: macro=%s err=%s", macro_id, err.message)
        return _fail("EXPORT_FAILED", err.message, "Macro might be locked or already gone.")

    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file outlives this fn; KM opens it after we close
        suffix=".kmmacros", delete=False, prefix="km_mcp_systrig_",
    )
    try:
        tmp.write(kmmacros_bytes)
        tmp.close()
        import_err = await _open_in_km(km, tmp.name)
        if import_err is not None:
            return _fail("IMPORT_FAILED", import_err, "KM rejected the modified .kmmacros file.")
        if not await _wait_for_uid(km, new_uid):
            return _fail(
                "IMPORT_FAILED",
                f"Imported macro but UID {new_uid} did not appear in KM within "
                f"{_VERIFY_TIMEOUT_SECONDS:.1f}s.",
                "Open the KM Editor and check the import dialog state.",
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            logger.warning("Failed to unlink temp .kmmacros %s", tmp.name)

    if ctx:
        await ctx.info(
            f"km_add_system_trigger: macro={macro_id} kind={trigger_kind} "
            f"old_uid={old_uid} new_uid={new_uid}",
        )
    return {
        "success": True,
        "data": {
            "old_macro_id": old_uid,
            "new_macro_id": new_uid,
            "group_name": group_name,
            "trigger_kind": trigger_kind,
            "uuid_changed": True,
        },
        "warning": (
            "The macro's UUID changed. Any ExecuteMacro action in other "
            "macros that referenced the old UID must be rewritten to use "
            "the new UID."
        ),
    }


async def _fetch_macro_xml(
    km: Any,
    macro_id: str,
) -> tuple[str, str, str] | None:
    """Return (macro_plist_xml, group_name, group_uid) or None on miss."""
    selector = km._macro_selector(macro_id.strip())  # noqa: SLF001 — sibling helper
    script = f'''
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
    '''
    result = await km.execute_applescript_async(script)
    if result.is_left():
        logger.warning("fetch xml failed: %s", result.get_left().message)
        return None
    output = result.get_right()
    if output.startswith("ERROR:"):
        logger.warning("fetch xml refused: %s", output[6:].strip())
        return None
    parts = output.split("###SEP###")
    if len(parts) < 3:
        logger.warning("fetch xml returned unexpected format")
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _wrap_in_kmmacros_doc(
    macro_dict: dict[str, Any],
    group_name: str,
    group_uid: str,
) -> bytes:
    document = [
        {
            "Activate": "Normal",
            "KeyCount": 0,
            "Macros": [macro_dict],
            "Name": group_name,
            "UID": group_uid,
        },
    ]
    return plistlib.dumps(document, fmt=plistlib.FMT_XML)


async def _open_in_km(km: Any, path: str) -> str | None:
    escaped = path.replace('"', '\\"')
    script = (
        'tell application "Keyboard Maestro" to '
        f'open POSIX file "{escaped}"'
    )
    result = await km.execute_applescript_async(script)
    if result.is_left():
        return result.get_left().message
    return None


async def _wait_for_uid(km: Any, uid: str) -> bool:
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


def _fail(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "recovery_suggestion": suggestion,
        },
    }
