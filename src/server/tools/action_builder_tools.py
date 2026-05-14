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
from pathlib import Path
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field

from ...core.types import MacroId
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
        return (
            "<dict>"
            "<key>MacroActionType</key><string>ExecuteAppleScript</string>"
            f"<key>Text</key><string>{source}</string>"
            "<key>TextSource</key><string>Text</string>"
            "</dict>"
        )
    if action_type == "execute_macro":
        target = config.get("target_macro")
        if not target:
            return None
        return (
            "<dict>"
            "<key>MacroActionType</key><string>ExecuteMacro</string>"
            f"<key>Macro</key><string>{_xml_escape(str(target))}</string>"
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


async def _do_list(macro_id: str | None) -> dict[str, Any]:
    if not macro_id or not macro_id.strip():
        return _failure("VALIDATION_ERROR", "macro_id is required.", "Pass macro_id.")
    result = await get_km_client().list_macro_actions_async(MacroId(macro_id.strip()))
    if result.is_left():
        return _failure("LIST_FAILED", result.get_left().message, "Verify the macro exists.")
    return {"success": True, "data": {"macro_id": macro_id, "actions": result.get_right()}}


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
    try:
        xml = _build_action_xml(action_type, action_config or {})
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
