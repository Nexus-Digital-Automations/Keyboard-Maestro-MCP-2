"""Universal trigger CRUD tool — full XML round-trip for every KM trigger type.

Owner: this file.

Companion km_client primitives: ``list_macro_triggers_with_xml_async``,
``get_macro_trigger_xml_async``, ``append_macro_trigger_xml_async``,
``update_macro_trigger_xml_async``, and the existing
``remove_macro_trigger_async`` / ``clear_macro_triggers_async``.

Distinct from ``km_trigger_manager``: that tool is a typed convenience
API for hotkey/application triggers; this one is the universal escape
hatch that accepts arbitrary KM trigger XML (or a structured dict
serialised via plistlib) and works for every ``MacroTriggerType``.
"""

import asyncio
import logging
import plistlib
import re
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
from ..initialization import get_km_client

logger = logging.getLogger(__name__)

_TRIGGER_DICT_RE = re.compile(r"<dict>.*?</dict>", re.DOTALL)


def _failure(
    code: str,
    message: str,
    suggestion: str,
    *,
    field: str | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message, "recovery_suggestion": suggestion}
    if field is not None:
        err["field"] = field
    return {"success": False, "error": err}


async def _kme_alive_or_error() -> dict[str, Any] | None:
    km = get_km_client()
    connection = await asyncio.get_event_loop().run_in_executor(None, km.check_connection)
    if connection.is_left() or not connection.get_right():
        return _failure(
            "KME_UNREACHABLE",
            "Cannot connect to Keyboard Maestro Engine.",
            "Start Keyboard Maestro and ensure the Engine is running.",
        )
    return None


def _trigger_dict_to_xml(payload: dict[str, Any]) -> str:
    """Serialise a Python dict to the KM trigger plist `<dict>...</dict>` form.

    KM expects only the bare ``<dict>`` element, not the full plist header.
    plistlib emits a full document, so we extract the first ``<dict>`` body.
    """
    full = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False).decode("utf-8")
    match = _TRIGGER_DICT_RE.search(full)
    if not match:
        raise ValueError("plistlib emitted no <dict> for trigger payload")
    return match.group(0)


def _parse_trigger_type(trigger_xml: str) -> str:
    """Extract ``MacroTriggerType`` value from a trigger plist XML fragment."""
    m = re.search(
        r"<key>MacroTriggerType</key>\s*<string>([^<]*)</string>",
        trigger_xml,
    )
    return m.group(1) if m else "Unknown"


def _resolve_payload(
    xml: str | None,
    trigger: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | dict[str, Any]:
    """Return ``(xml, summary)`` or a `_failure(...)` dict."""
    if xml is not None and trigger is not None:
        return _failure(
            "BOTH_XML_AND_TRIGGER",
            "Pass exactly one of 'xml' or 'trigger', not both.",
            "Remove whichever you didn't mean.",
            field="xml/trigger",
        )
    if xml is not None:
        cleaned = xml.strip()
        if not cleaned.startswith("<dict>") or not cleaned.endswith("</dict>"):
            return _failure(
                "VALIDATION_ERROR",
                "'xml' must be a single <dict>...</dict> element.",
                "Wrap or trim to a bare <dict> trigger element.",
                field="xml",
            )
        return cleaned, {"type": _parse_trigger_type(cleaned)}
    if trigger is not None:
        try:
            serialised = _trigger_dict_to_xml(trigger)
        except (ValueError, TypeError, OverflowError) as exc:
            return _failure(
                "VALIDATION_ERROR",
                f"Could not serialise 'trigger' to plist XML: {exc}",
                "Use plist-compatible types (str/int/bool/list/dict).",
                field="trigger",
            )
        ttype = str(trigger.get("MacroTriggerType", _parse_trigger_type(serialised)))
        return serialised, {"type": ttype}
    return _failure(
        "MISSING_PAYLOAD",
        "Provide either 'xml' or 'trigger'.",
        "Pass a structured dict via 'trigger', or raw XML via 'xml'.",
        field="xml/trigger",
    )


async def _op_list(macro_id: str) -> dict[str, Any]:
    result = await get_km_client().list_macro_triggers_with_xml_async(MacroId(macro_id))
    if result.is_left():
        return _failure("LIST_FAILED", result.get_left().message, "Verify the macro exists.")
    triggers = result.get_right()
    for t in triggers:
        t["type"] = _parse_trigger_type(t.get("xml", ""))
    return {"success": True, "data": {"macro_id": macro_id, "triggers": triggers}}


async def _op_get(macro_id: str, index: int) -> dict[str, Any]:
    result = await get_km_client().get_macro_trigger_xml_async(MacroId(macro_id), index)
    if result.is_left():
        msg = result.get_left().message
        code = "INDEX_OUT_OF_RANGE" if "out of" in msg.lower() or "invalid" in msg.lower() else "LIST_FAILED"
        return _failure(code, msg, "Call operation='list' to see valid indices.")
    xml_text = result.get_right()
    return {
        "success": True,
        "data": {
            "macro_id": macro_id,
            "index": index,
            "type": _parse_trigger_type(xml_text),
            "xml": xml_text,
        },
    }


async def _op_add(macro_id: str, xml_text: str, summary: dict[str, Any]) -> dict[str, Any]:
    km = get_km_client()
    add_result = await km.append_macro_trigger_xml_async(MacroId(macro_id), xml_text)
    if add_result.is_left():
        msg = add_result.get_left().message
        code = "KM_REJECTED_XML" if "xml" in msg.lower() else "ADD_FAILED"
        return _failure(code, msg, "Verify the trigger XML matches KM's plist schema.")
    list_result = await km.list_macro_triggers_with_xml_async(MacroId(macro_id))
    new_index = len(list_result.get_right()) if list_result.is_right() else None
    return {
        "success": True,
        "data": {"macro_id": macro_id, "index": new_index, "type": summary["type"]},
    }


async def _op_update(
    macro_id: str,
    index: int,
    xml_text: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    result = await get_km_client().update_macro_trigger_xml_async(
        MacroId(macro_id), index, xml_text,
    )
    if result.is_left():
        msg = result.get_left().message
        if "out of" in msg.lower() or "invalid index" in msg.lower():
            code = "INDEX_OUT_OF_RANGE"
        elif "xml" in msg.lower():
            code = "KM_REJECTED_XML"
        else:
            code = "UPDATE_FAILED"
        return _failure(code, msg, "Call operation='list' to confirm indices and current XML.")
    return {
        "success": True,
        "data": {"macro_id": macro_id, "index": index, "type": summary["type"]},
    }


async def _op_remove(macro_id: str, index: int) -> dict[str, Any]:
    result = await get_km_client().remove_macro_trigger_async(MacroId(macro_id), index)
    if result.is_left():
        msg = result.get_left().message
        code = "INDEX_OUT_OF_RANGE" if "out of" in msg.lower() else "REMOVE_FAILED"
        return _failure(code, msg, "Call operation='list' to find a valid 1-based index.")
    return {"success": True, "data": {"macro_id": macro_id, "index": index}}


async def _op_replace_all(
    macro_id: str,
    triggers: list[dict[str, Any]],
) -> dict[str, Any]:
    km = get_km_client()
    cleared = await km.clear_macro_triggers_async(MacroId(macro_id))
    if cleared.is_left():
        return _failure(
            "REPLACE_FAILED",
            f"clear failed: {cleared.get_left().message}",
            "Verify the macro exists and KM Engine is responsive.",
        )
    for position, entry in enumerate(triggers, start=1):
        raw_xml = entry.get("xml") if "xml" in entry else None
        as_dict = None if raw_xml is not None else entry
        resolved = _resolve_payload(raw_xml, as_dict)
        if isinstance(resolved, dict):
            err = resolved.setdefault("error", {})
            err["failed_index"] = position
            return resolved
        xml_text, _summary = resolved
        added = await km.append_macro_trigger_xml_async(MacroId(macro_id), xml_text)
        if added.is_left():
            return _failure(
                "REPLACE_FAILED",
                f"append at index {position} failed: {added.get_left().message}",
                "Earlier triggers were already inserted; inspect with operation='list'.",
                field=f"triggers[{position - 1}]",
            )
    return {"success": True, "data": {"macro_id": macro_id, "count": len(triggers)}}


async def km_trigger_crud(
    operation: Annotated[
        Literal["list", "get", "add", "update", "remove", "replace_all"],
        Field(description="CRUD verb."),
    ],
    macro_id: Annotated[
        str,
        Field(description="Target macro name or UUID.", min_length=1, max_length=255),
    ],
    index: Annotated[
        int | None,
        Field(default=None, description="1-indexed trigger position. Required for get/update/remove.", ge=1),
    ] = None,
    xml: Annotated[
        str | None,
        Field(
            default=None,
            description="Trigger plist XML (<dict>...</dict>). For add/update.",
            max_length=64_000,
        ),
    ] = None,
    trigger: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "Structured trigger dict, serialised to plist XML. Alternative "
                "to 'xml' for add/update. e.g. "
                "{'MacroTriggerType': 'HotKey', 'KeyCode': 49, 'Modifiers': 256}."
            ),
        ),
    ] = None,
    triggers: Annotated[
        list[dict[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Full replacement list for operation='replace_all'. Each element "
                "is either a trigger dict or {'xml': '<dict>...</dict>'}."
            ),
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Full CRUD for triggers on an existing Keyboard Maestro macro.

    Failure modes (stable error codes):
    - VALIDATION_ERROR: missing/invalid argument for the chosen operation
    - MISSING_PAYLOAD: neither 'xml' nor 'trigger' was supplied
    - BOTH_XML_AND_TRIGGER: both were supplied
    - INDEX_OUT_OF_RANGE: ``index`` past the end of the trigger list
    - KM_REJECTED_XML: KM AppleScript refused the supplied XML
    - KME_UNREACHABLE: Keyboard Maestro Engine offline
    - LIST_FAILED / ADD_FAILED / UPDATE_FAILED / REMOVE_FAILED / REPLACE_FAILED:
      AppleScript reported an error
    """
    if ctx:
        await ctx.info(f"km_trigger_crud op={operation} macro={macro_id!r}")

    macro_id = macro_id.strip()
    if not macro_id:
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro name or UUID.")

    connection_error = await _kme_alive_or_error()
    if connection_error is not None:
        return connection_error

    if operation == "list":
        return await _op_list(macro_id)

    if operation == "get":
        if index is None:
            return _failure("VALIDATION_ERROR", "index is required for 'get'.", "Pass index >= 1.", field="index")
        return await _op_get(macro_id, index)

    if operation == "remove":
        if index is None:
            return _failure("VALIDATION_ERROR", "index is required for 'remove'.", "Pass index >= 1.", field="index")
        return await _op_remove(macro_id, index)

    if operation == "add":
        resolved = _resolve_payload(xml, trigger)
        if isinstance(resolved, dict):
            return resolved
        xml_text, summary = resolved
        return await _op_add(macro_id, xml_text, summary)

    if operation == "update":
        if index is None:
            return _failure(
                "VALIDATION_ERROR", "index is required for 'update'.", "Pass index >= 1.", field="index",
            )
        resolved = _resolve_payload(xml, trigger)
        if isinstance(resolved, dict):
            return resolved
        xml_text, summary = resolved
        return await _op_update(macro_id, index, xml_text, summary)

    if triggers is None:
        return _failure(
            "VALIDATION_ERROR",
            "triggers list is required for 'replace_all'.",
            "Pass triggers=[{...}, ...].",
            field="triggers",
        )
    return await _op_replace_all(macro_id, triggers)
