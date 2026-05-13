# Keyboard Maestro MCP Server

An MCP server that lets a client drive [Keyboard Maestro](https://www.keyboardmaestro.com/) on macOS — run macros, manage variables, configure triggers and hotkeys, edit macros, and exercise the Keyboard Maestro Engine.

This server **only** controls Keyboard Maestro. It does not embed an LLM, vision model, IoT controller, voice recognizer, analytics engine, identity layer, or raw mouse/keyboard simulator — those belong elsewhere (the computer-use MCP covers raw input and screen interaction).

## Tool surface (27 tools)

Auto-discovered from `src/server/tools/` at startup. Each tool routes through `src/integration/km_client.py` to the Keyboard Maestro Engine via AppleScript or the KM web server.

| Area | Tool | Source |
|---|---|---|
| Execution | `km_execute_macro`, `km_list_macros`, `km_variable_manager` | `core_tools.py` |
| Macro creation | `km_create_macro`, `km_list_templates` | `creation_tools.py` |
| Macro editing | `km_macro_editor` | `macro_editor_tools.py` |
| Macro groups | `km_macro_group_manager`, `km_move_macro_to_group` | `macro_group_tools.py`, `macro_move_tools.py` |
| Engine | `km_engine_control` | `engine_tools.py` |
| Tokens | `km_token_processor`, `km_token_stats` | `token_tools.py` |
| Actions | `km_action_builder` (appends/lists/deletes/clears actions; supports the six verified built-in types plus `plug_in` for any installed third-party plug-in), `km_list_action_types` (lists what `km_action_builder` can append, including every installed plug-in), `km_build_plugin_action` (writes a new `.kmactions` bundle to disk) | `action_builder_tools.py`, `action_tools.py`, `plugin_action_tools.py` |
| Conditions | `km_add_condition` | `condition_tools.py` |
| Control flow | `km_control_flow` | `control_flow_tools.py` |
| Triggers | `km_trigger_crud`, `km_trigger_manager`, `km_add_system_trigger` | `trigger_crud_tools.py`, `trigger_tools.py`, `system_trigger_tools.py` |
| Hotkeys | `km_create_hotkey_trigger`, `km_list_hotkey_triggers` | `hotkey_tools.py` |
| Windows | `km_window_manager` | `window_tools.py` |
| Applications | `km_application_control` | `application_tools.py` |
| Notifications | `km_notifications`, `km_notification_status`, `km_dismiss_notifications` | `notification_tools.py` |

Some tools currently have partial XML emitters (`km_add_condition`, `km_control_flow`, `km_create_macro` for non-`custom` templates). `km_action_builder` covers seven action types end-to-end (`pause`, `type_text`, `paste`, `set_variable`, `run_applescript`, `execute_macro`, `plug_in`); to add another built-in type, add a branch in `action_builder_tools._build_action_xml` and a registry entry in `action_tools._BUILTIN_ACTION_TYPES` in the same change. See `docs/km_mcp_audit_report.md` for the known-stub matrix.

## Install

```bash
git clone https://github.com/anthropics/keyboard-maestro-mcp.git
cd keyboard-maestro-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
```

Prerequisites:
- macOS 10.15+
- Python 3.10+
- Keyboard Maestro 10.0+ with **Web Server** enabled (Preferences → Web Server)
- Accessibility permissions granted to the Python interpreter

## Run

```bash
km-mcp-server
# or
python -m src.main_dynamic
```

## Claude Desktop configuration

In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "keyboard-maestro": {
      "command": "/absolute/path/to/keyboard-maestro-mcp/.venv/bin/python",
      "args": ["-m", "src.main_dynamic"],
      "cwd": "/absolute/path/to/keyboard-maestro-mcp",
      "env": {
        "KM_WEB_SERVER_PORT": "4490",
        "KM_CONNECTION_TIMEOUT": "30",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Architecture

```
src/
├── main_dynamic.py          # FastMCP server entry point
├── server/
│   ├── tool_registry.py     # Auto-discovers km_* functions from src/server/tools/
│   ├── dynamic_registration.py
│   ├── tool_config.py       # Security policy overrides per tool
│   └── tools/               # 19 tool modules — each km_* function becomes an MCP tool
├── integration/
│   ├── km_client.py         # Single bridge to Keyboard Maestro (AppleScript + HTTP)
│   ├── km_conditions.py     # KM condition spec generation
│   ├── km_triggers.py       # KM trigger spec generation
│   ├── km_control_flow.py   # KM if/while/for action insertion
│   └── km_macro_editor.py   # KM macro modification operations
├── core/                    # Engine abstractions, types, errors, contracts
└── security/                # Input sanitization + validation
```

All MCP tools live in `src/server/tools/` and depend inward on `src/core/`, `src/integration/`, and `src/security/`. There is exactly one client to Keyboard Maestro (`src/integration/km_client.py`); tools never speak AppleScript directly.

## License

MIT
