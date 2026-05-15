"""Action builder tool — list, append, delete, clear actions inside a macro.

Adapter over ``KMClient`` action primitives. Owns the action_type-to-XML
mapping. All AppleScript / KM-engine logic lives in
``src/integration/km_client.py``.

Built-in action types (verified against KM's "Copy as XML" output): ``pause``,
``type_text``, ``paste``, ``set_variable``, ``run_applescript``,
``execute_macro``. Plus ``plug_in``, which dispatches to any installed
third-party plug-in by folder name. Adding more built-ins is a new branch
in ``_build_action_xml`` — no architecture change. ``insert``/``reorder``
ops deferred (need index-aware AppleScript); user can sequence by
appending in order.
"""

import asyncio
import logging
import plistlib
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
from ...integration.km_macro_rebuild import (
    fetch_macro_snapshot,
    rebuild_macro_via_reimport,
)
from ..initialization import get_km_client
from ._action_templates import (
    find_macro_action_type,
    render_action_xml,
    validate_pasted_dict_xml,
)
from .plugin_action_tools import _scan_installed_plugins

logger = logging.getLogger(__name__)


def _failure(code: str, message: str, suggestion: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, "recovery_suggestion": suggestion},
    }


async def _check_kme_alive() -> dict[str, Any] | None:
    km = get_km_client()
    connection = await asyncio.get_event_loop().run_in_executor(
        None, km.check_connection,
    )
    if connection.is_left() or not connection.get_right():
        return _failure(
            "KM_CONNECTION_FAILED",
            "Cannot connect to Keyboard Maestro Engine.",
            "Start Keyboard Maestro and ensure the Engine is running.",
        )
    return None


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _build_action_xml(action_type: str, config: dict[str, Any]) -> str | None:
    """Return KM action <dict> XML for a v1 action_type, or None if unsupported.

    Each KM macro action is a plist <dict> with at minimum a MacroActionType
    key. Field names below match KM's persisted XML schema (verified
    against KM's "Copy as XML" output for each action type).
    """
    if action_type == "pause":
        seconds = config.get("seconds", 1)
        return (
            "<dict>"
            "<key>MacroActionType</key><string>Pause</string>"
            f"<key>Time</key><string>{_xml_escape(str(seconds))}</string>"
            "</dict>"
        )
    if action_type == "type_text":
        text = _xml_escape(str(config.get("text", "")))
        return (
            "<dict>"
            "<key>Action</key><string>ByTyping</string>"
            "<key>MacroActionType</key><string>InsertText</string>"
            f"<key>Text</key><string>{text}</string>"
            "</dict>"
        )
    if action_type == "paste":
        text = _xml_escape(str(config.get("text", "")))
        return (
            "<dict>"
            "<key>Action</key><string>ByPasting</string>"
            "<key>MacroActionType</key><string>InsertText</string>"
            f"<key>Text</key><string>{text}</string>"
            "</dict>"
        )
    if action_type == "set_variable":
        variable = config.get("variable")
        if not variable:
            return None
        return (
            "<dict>"
            "<key>MacroActionType</key><string>SetVariableToText</string>"
            f"<key>Variable</key><string>{_xml_escape(str(variable))}</string>"
            f"<key>Text</key><string>{_xml_escape(str(config.get('text', '')))}</string>"
            "</dict>"
        )
    if action_type == "run_applescript":
        source = _xml_escape(str(config.get("source", "")))
        # KM auto-fills two keys when omitted that we override explicitly:
        #   StopOnFailure=false       silently swallows AppleScript errors, so
        #                             a surrounding TryCatch never fires.
        #                             Emit StopOnFailure=true to propagate
        #                             (closes audit defect D1).
        #   IncludedVariables=["9999"]  KM's sentinel for "all instance vars".
        #                             Emit an empty array so the action is
        #                             self-contained / matches a hand-authored
        #                             no-inclusion action.
        # UseText=true keeps KM from rendering "Execute 'Unknown' AppleScript".
        return (
            "<dict>"
            "<key>IncludedVariables</key><array/>"
            "<key>MacroActionType</key><string>ExecuteAppleScript</string>"
            "<key>StopOnFailure</key><true/>"
            f"<key>Text</key><string>{source}</string>"
            "<key>TextSource</key><string>Text</string>"
            "<key>UseText</key><true/>"
            "</dict>"
        )
    if action_type == "execute_macro":
        # KM 11's ExecuteMacro stores the target as a bare top-level
        # MacroUID string — verified by inject-and-read-back probe against
        # KM 11.0.4 (canonical shape captured in km_action_templates.json).
        # Earlier shapes (Macro=<dict>MacroName,MacroUID</dict>, bare
        # Macro=<string>) are silently dropped on import — the resulting
        # macro shows Actions=<array/> with no warning. The caller
        # (_do_append) resolves target_macro → (name, uid); only uid is
        # written into the plist.
        uid = config.get("target_macro_uid")
        if not uid:
            return None
        return (
            "<dict>"
            "<key>Asynchronously</key><false/>"
            "<key>MacroActionType</key><string>ExecuteMacro</string>"
            f"<key>MacroUID</key><string>{_xml_escape(str(uid))}</string>"
            "<key>TimeOutAbortsMacro</key><true/>"
            "<key>UseParameter</key><false/>"
            "</dict>"
        )
    if action_type == "activate_application":
        return _build_activate_application_xml(config)
    if action_type == "manipulate_window":
        return _build_manipulate_window_xml(config)
    if action_type == "execute_shell_script":
        return _build_execute_shell_script_xml(config)
    if action_type == "plug_in":
        return _build_plug_in_xml(config)
    if action_type == "paste_xml":
        raw = (config or {}).get("xml")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("paste_xml requires action_config={'xml': '<dict>...</dict>'}")
        return validate_pasted_dict_xml(raw)
    macro_action_type = find_macro_action_type(action_type)
    if macro_action_type is None:
        return None
    return render_action_xml(macro_action_type, config or {})


# Verified against KM's "Copy as XML" output for an MCP Smoke Plugin action
# inserted via the editor's Insert Action menu:
#   MacroActionType=PlugIn (NOT ExecutePlugIn)
#   PlugInFolderName = bundle directory basename (with .kmactions if present)
#   PlugInParameters = dict keyed by parameter Label (case-sensitive)
#   DisplayKind = one of None / Variable / Window / Briefly / Clipboard / Typing / Pasting
#   Variable key present only when DisplayKind=Variable
_PLUG_IN_DISPLAY_KINDS = frozenset(
    {"None", "Variable", "Window", "Briefly", "Clipboard", "Typing", "Pasting"}
)


def _resolve_plug_in_folder_name(identifier: str) -> str | None:
    """Map a plug-in plist Name or folder basename to its on-disk folder name.

    KM's macro XML stores ``PlugInFolderName`` (the directory basename, with
    ``.kmactions`` if present) — not the plist ``Name`` value. Clients pick a
    plug-in by the friendly identifier returned by ``km_list_action_types``,
    and we resolve to the folder here. Returns ``None`` if no installed
    plug-in matches.
    """
    target = identifier.strip()
    if not target:
        return None
    for spec in _scan_installed_plugins():
        folder = Path(spec["bundle_path"]).name
        if target in (spec["identifier"], folder):
            return folder
    return None


def _build_activate_application_xml(config: dict[str, Any]) -> str | None:
    """Emit an ``ActivateApplication`` action plist.

    Config: ``app_name`` and ``bundle_id`` (at least one required), optional
    ``app_path`` (NewFile in plist), ``all_windows`` (default true),
    ``reopen_windows`` (default false), ``already_activated`` (Normal /
    Hide / SwitchToLast — KM defaults Normal).
    """
    app_name = str(config.get("app_name", ""))
    bundle_id = str(config.get("bundle_id", ""))
    if not app_name and not bundle_id:
        return None
    path = str(config.get("app_path", ""))
    all_windows = config.get("all_windows", True)
    reopen = config.get("reopen_windows", False)
    already = str(config.get("already_activated", "Normal"))
    path_key = (
        f"<key>NewFile</key><string>{_xml_escape(path)}</string>" if path else ""
    )
    return (
        "<dict>"
        f"<key>AllWindows</key><{'true' if all_windows else 'false'}/>"
        f"<key>AlreadyActivatedActionType</key><string>{_xml_escape(already)}</string>"
        "<key>Application</key>"
        "<dict>"
        f"<key>BundleIdentifier</key><string>{_xml_escape(bundle_id)}</string>"
        f"<key>Name</key><string>{_xml_escape(app_name)}</string>"
        f"{path_key}"
        "</dict>"
        "<key>MacroActionType</key><string>ActivateApplication</string>"
        f"<key>ReopenWindows</key><{'true' if reopen else 'false'}/>"
        "<key>TimeOutAbortsMacro</key><true/>"
        "</dict>"
    )


def _build_manipulate_window_xml(config: dict[str, Any]) -> str | None:
    """Emit a ``ManipulateWindow`` action plist.

    Config: ``action`` (Move / Resize / SelectWindow / Close / Minimize / Zoom
    — KM has many; pass through), ``targeting`` (FrontWindow / NamedWindow
    / SpecificWindow), ``x`` / ``y`` / ``width`` / ``height`` (calculation
    expressions, strings), ``window_index`` (calculation expression for
    SpecificWindow), ``window_name`` (for NamedWindow).
    """
    action = str(config.get("action", "MoveAndResize"))
    targeting = str(config.get("targeting", "FrontWindow"))
    targeting_type = "Front" if targeting == "FrontWindow" else "Specific"
    return (
        "<dict>"
        f"<key>Action</key><string>{_xml_escape(action)}</string>"
        f"<key>HeightExpression</key><string>{_xml_escape(str(config.get('height', '300')))}</string>"
        f"<key>HorizontalExpression</key><string>{_xml_escape(str(config.get('x', '125')))}</string>"
        "<key>MacroActionType</key><string>ManipulateWindow</string>"
        "<key>TargetApplication</key><dict/>"
        f"<key>Targeting</key><string>{_xml_escape(targeting)}</string>"
        f"<key>TargetingType</key><string>{_xml_escape(targeting_type)}</string>"
        f"<key>VerticalExpression</key><string>{_xml_escape(str(config.get('y', '125')))}</string>"
        f"<key>WidthExpression</key><string>{_xml_escape(str(config.get('width', '300')))}</string>"
        f"<key>WindowIndexExpression</key><string>{_xml_escape(str(config.get('window_index', '1')))}</string>"
        f"<key>WindowName</key><string>{_xml_escape(str(config.get('window_name', '')))}</string>"
        "</dict>"
    )


def _build_execute_shell_script_xml(config: dict[str, Any]) -> str | None:
    """Emit an ``ExecuteShellScript`` action plist.

    Config: ``source`` (the shell script text — required), ``display`` (Window
    / Briefly / TypeResults / PasteResults / SaveResultsToVariable / Nothing
    — default Nothing), optional ``destination_variable`` for
    SaveResultsToVariable.
    """
    source = str(config.get("source", ""))
    if not source:
        return None
    display = str(config.get("display", "Nothing"))
    dest_var = str(config.get("destination_variable", ""))
    dest_key = (
        f"<key>Variable</key><string>{_xml_escape(dest_var)}</string>" if dest_var else ""
    )
    return (
        "<dict>"
        f"<key>DisplayKind</key><string>{_xml_escape(display)}</string>"
        "<key>HonourFailureSettings</key><true/>"
        "<key>IncludeStdErr</key><false/>"
        "<key>MacroActionType</key><string>ExecuteShellScript</string>"
        "<key>Path</key><string></string>"
        f"<key>Text</key><string>{_xml_escape(source)}</string>"
        "<key>TimeOutAbortsMacro</key><true/>"
        "<key>TrimResults</key><true/>"
        "<key>TrimResultsNew</key><true/>"
        "<key>UseText</key><true/>"
        f"{dest_key}"
        "</dict>"
    )


def _build_plug_in_xml(config: dict[str, Any]) -> str | None:
    """Emit a Keyboard Maestro ``PlugIn`` action plist for an installed plug-in.

    Required config: ``plugin_identifier`` (the friendly name returned by
    ``km_list_action_types``). Optional: ``parameters`` (dict keyed by the
    plug-in's parameter labels), ``display_kind`` (default ``None``),
    ``variable`` (only honored when ``display_kind="Variable"``). Returns
    ``None`` when the plug-in is not installed or ``display_kind`` is unknown.
    """
    identifier = config.get("plugin_identifier")
    if not identifier or not isinstance(identifier, str):
        return None
    folder_name = _resolve_plug_in_folder_name(identifier)
    if folder_name is None:
        return None
    display_kind = config.get("display_kind", "None")
    if display_kind not in _PLUG_IN_DISPLAY_KINDS:
        return None
    params = config.get("parameters") or {}
    params_xml = "".join(
        f"<key>{_xml_escape(str(k))}</key><string>{_xml_escape(str(v))}</string>"
        for k, v in params.items()
    )
    variable_xml = ""
    if display_kind == "Variable":
        variable_xml = (
            f"<key>Variable</key><string>{_xml_escape(str(config.get('variable', '')))}</string>"
        )
    return (
        "<dict>"
        f"<key>DisplayKind</key><string>{display_kind}</string>"
        "<key>IncludeStdErr</key><true/>"
        "<key>MacroActionType</key><string>PlugIn</string>"
        f"<key>PlugInFolderName</key><string>{_xml_escape(folder_name)}</string>"
        f"<key>PlugInParameters</key><dict>{params_xml}</dict>"
        "<key>TimeOutAbortsMacro</key><true/>"
        "<key>TrimResultsNew</key><true/>"
        f"{variable_xml}"
        "</dict>"
    )


async def _resolve_execute_macro_target(
    target_macro: Any,
) -> tuple[str, str] | None:
    """Resolve target_macro (name or UUID) to ``(name, uid)`` for ExecuteMacro XML.

    KM's ExecuteMacro action stores the target as ``Macro = {MacroName, MacroUID}``;
    a bare name doesn't survive plist validation. Returns ``None`` if the
    target is missing, malformed, or not present in KM's macro list.
    """
    if not target_macro or not isinstance(target_macro, str):
        return None
    target = target_macro.strip()
    if not target:
        return None
    list_result = await get_km_client().list_macros_async(enabled_only=False)
    if list_result.is_left():
        logger.warning(
            "execute_macro target resolve: list_macros failed: %s",
            list_result.get_left().message,
        )
        return None
    for record in list_result.get_right():
        record_uid = str(record.get("id") or record.get("macroId") or "")
        record_name = str(record.get("name") or record.get("macroName") or "")
        if not record_uid or not record_name:
            continue
        if target in (record_uid, record_name):
            return record_name, record_uid
    return None


async def _do_list(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().list_macro_actions_async(MacroId(macro_id.strip()))
    if result.is_left():
        return _failure("LIST_FAILED", result.get_left().message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "actions": result.get_right()}}


async def _append_execute_macro_via_rebuild(
    macro_id: str,
    target_macro: Any,
) -> dict[str, Any]:
    """Append an ExecuteMacro action via export-edit-reimport.

    KM 11's ``make new action`` AppleScript verb rejects every ExecuteMacro
    plist shape (verified by R9 probe — KM substitutes a
    ``Log "Invalid XML From AppleScript"`` placeholder). The
    export-edit-reimport pipeline is the only working path; it rotates the
    macro's UID as a side-effect, surfaced in the response.

    Failure modes:
    - ``EXECUTE_MACRO_TARGET_NOT_FOUND``: target_macro doesn't resolve.
    - ``EXPORT_FAILED`` / ``IMPORT_FAILED``: rebuild pipeline rejected.
    - ``APPEND_FAILED``: emitter returned no XML.
    """
    resolved = await _resolve_execute_macro_target(target_macro)
    if resolved is None:
        return _failure(
            "EXECUTE_MACRO_TARGET_NOT_FOUND",
            f"target_macro {target_macro!r} did not resolve to a known macro. "
            "KM's ExecuteMacro action requires a real MacroUID.",
            "Pass an existing macro name or UUID; list with km_list_macros.",
        )
    target_name, target_uid = resolved
    xml = _build_action_xml(
        "execute_macro",
        {"target_macro_name": target_name, "target_macro_uid": target_uid},
    )
    if xml is None:
        return _failure(
            "APPEND_FAILED",
            "execute_macro emitter returned no XML despite resolved target.",
            "File a bug — emitter and resolver disagree.",
        )
    action_dict = plistlib.loads(
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        f'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
        f'<plist version="1.0">{xml}</plist>'.encode(),
    )
    km = get_km_client()
    snap_result = await fetch_macro_snapshot(km, macro_id)
    if snap_result.is_left():
        err = snap_result.get_left()
        code = "EXECUTE_MACRO_MACRO_NOT_FOUND" if err.code == "NOT_FOUND_ERROR" else "EXPORT_FAILED"
        logger.warning("execute_macro append: fetch failed macro=%s err=%s", macro_id, err.message)
        return _failure(code, err.message, "Verify the macro UID/name via km_list_macros.")
    snapshot = snap_result.get_right()
    new_plist = dict(snapshot.plist)
    new_plist["Actions"] = [*new_plist.get("Actions", []), action_dict]
    rebuild_result = await rebuild_macro_via_reimport(km, snapshot, new_plist)
    if rebuild_result.is_left():
        err = rebuild_result.get_left()
        logger.warning("execute_macro append: rebuild failed macro=%s err=%s", macro_id, err.message)
        return _failure(
            "IMPORT_FAILED",
            err.message,
            "Inspect KM Editor for import dialogs or naming conflicts.",
        )
    rebuilt = rebuild_result.get_right()
    return {
        "success": True,
        "data": {
            "old_macro_id": rebuilt.old_uid,
            "new_macro_id": rebuilt.new_uid,
            "group_name": rebuilt.group_name,
            "action_type": "execute_macro",
            "appended": True,
            "uuid_changed": True,
        },
        "warning": (
            "The macro's UUID changed. Any ExecuteMacro action in other "
            "macros that referenced the old UID must be rewritten to use "
            "the new UID."
        ),
    }


async def _do_append(
    macro_id: str | None,
    action_type: str | None,
    action_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or not action_type:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and action_type are required for operation='append'.",
            "Pass both, plus action_config with type-appropriate fields.",
        )
    config = dict(action_config or {})
    if action_type == "execute_macro":
        return await _append_execute_macro_via_rebuild(
            macro_id.strip(), config.get("target_macro"),
        )
    try:
        xml = _build_action_xml(action_type, config)
    except ValueError as exc:
        logger.warning("action_builder render failed: type=%s err=%s", action_type, exc)
        return _failure(
            "VALIDATION_ERROR",
            f"action_config invalid for action_type '{action_type}': {exc}",
            "Check parameter names against km_list_action_types output for this type.",
        )
    if xml is None:
        return _failure(
            "UNSUPPORTED_ACTION_TYPE",
            f"action_type '{action_type}' not supported, missing required config field, "
            "or (for plug_in) the plugin_identifier is not installed.",
            "Supported: pause, type_text, paste, set_variable (needs 'variable'), "
            "run_applescript, execute_macro (needs 'target_macro'), "
            "plug_in (needs 'plugin_identifier' matching a name returned by km_list_action_types; "
            "optional 'parameters' dict, 'display_kind' in None|Variable|Window|Briefly|"
            "Clipboard|Typing|Pasting, 'variable' for display_kind='Variable').",
        )
    result = await get_km_client().append_macro_action_async(MacroId(macro_id.strip()), xml)
    if result.is_left():
        return _failure("APPEND_FAILED", result.get_left().message, "Verify macro exists.")
    return {
        "success": True,
        "data": {"macro_id": macro_id, "action_type": action_type, "appended": True},
    }


async def _do_delete(macro_id: str | None, action_index: int | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip() or action_index is None:
        return _failure(
            "VALIDATION_ERROR",
            "macro_id and action_index are required.",
            "Use operation='list' first to find the 1-indexed position.",
        )
    result = await get_km_client().delete_macro_action_async(
        MacroId(macro_id.strip()), action_index,
    )
    if result.is_left():
        return _failure("DELETE_FAILED", result.get_left().message, "Verify the index is valid.")
    return {"success": True, "data": {"macro_id": macro_id, "action_index": action_index}}


async def _do_clear(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().clear_macro_actions_async(MacroId(macro_id.strip()))
    if result.is_left():
        return _failure("CLEAR_FAILED", result.get_left().message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "cleared": True}}


async def km_action_builder(
    operation: Annotated[
        Literal["list", "append", "delete", "clear"],
        Field(description="Action operation."),
    ],
    macro_id: Annotated[
        str | None,
        Field(default=None, description="Target macro name or UUID.", max_length=255),
    ] = None,
    action_type: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "For 'append': pause, type_text, paste, set_variable, run_applescript, "
                "execute_macro, plug_in, paste_xml, or any built-in identifier from "
                "km_list_action_types (e.g. 'speak_text')."
            ),
            max_length=64,
        ),
    ] = None,
    action_config: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "For 'append'. pause: {seconds: 1}. type_text/paste: {text: '...'}. "
                "set_variable: {variable: 'Name', text: '...'}. "
                "run_applescript: {source: '...'}. execute_macro: {target_macro: 'Name'}. "
                "plug_in: {plugin_identifier: 'MCP Smoke Plugin', parameters: {Label: value}, "
                "display_kind: 'Variable', variable: 'OutVar'} — plugin_identifier matches "
                "an entry from km_list_action_types. "
                "paste_xml: {xml: '<dict>...</dict>'} for any action whose <dict> body you "
                "already have (validated as plist with MacroActionType key). "
                "Built-in catalog identifiers (e.g. speak_text): keys named in the "
                "entry's `parameters` list."
            ),
        ),
    ] = None,
    action_index: Annotated[
        int | None,
        Field(default=None, description="1-indexed action position for 'delete'.", ge=1),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Build action sequences inside Keyboard Maestro macros.

    Failure modes:
    - VALIDATION_ERROR: missing or invalid argument for the operation
    - KM_CONNECTION_FAILED: Keyboard Maestro Engine is not reachable
    - UNSUPPORTED_ACTION_TYPE: action_type not in v1, or required config
      field missing (e.g., set_variable without 'variable')
    - LIST_FAILED / APPEND_FAILED / DELETE_FAILED / CLEAR_FAILED:
      AppleScript reported an error (macro not found, index out of range,
      KM rejected the action XML)
    """
    if ctx:
        await ctx.info(f"km_action_builder op={operation} macro={macro_id!r}")

    connection_error = await _check_kme_alive()
    if connection_error is not None:
        return connection_error

    if operation == "list":
        return await _do_list(macro_id)
    if operation == "append":
        return await _do_append(macro_id, action_type, action_config)
    if operation == "delete":
        return await _do_delete(macro_id, action_index)
    return await _do_clear(macro_id)
