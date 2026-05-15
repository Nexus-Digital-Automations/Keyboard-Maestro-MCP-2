# Keyboard Maestro MCP — Client Usage Guide

You are an AI client connected to the **Keyboard-Maestro-MCP-2** server.
This guide tells you exactly which tool to call for each task, the
canonical call shape, and the failure modes you must handle. Read the
decision tree first; jump to a tool section only when you've picked one.

The server has **30 tools** across 10 categories. They drive a running
Keyboard Maestro 11 engine via AppleScript + Web API. They **do not**
duplicate the computer-use surface — no mouse, no keyboard simulation
(beyond `type_text`/`paste` inside macros), no screen capture. For those,
use the `computer-use` MCP.

---

## Decision tree — pick the right tool

### "I want to find / read existing macros"

| Need | Tool |
|---|---|
| List macros (optionally filtered by group / enabled) | `km_list_macros` |
| List macro groups | `km_macro_group_manager(operation="list")` |
| List hotkey triggers (optionally with conflicts) | `km_list_hotkey_triggers` |
| List installed action types + plug-ins | `km_list_action_types` |
| List macro templates | `km_list_templates` |
| Search the action catalog by task | `km_search_actions(query="...")` |
| Read the action list inside a macro | `km_action_builder(operation="list", macro_id=...)` |
| Read trigger list on a macro | `km_trigger_crud(operation="list", macro_id=...)` |

### "I want to run a macro that already exists"

- `km_execute_macro(identifier="<UUID or name>", method="applescript")` —
  default. Falls back automatically if AppleScript path fails.
- For macros not currently in an active group, you'll get
  `EXECUTION_ERROR: do script found no macros with a matching name…`.
  Fix: ensure the macro's parent group is "Always Activated" in the KM
  editor, then retry.

### "I want to build a macro from scratch"

Pick **one** path:

1. **Template start (recommended for common cases).** `km_create_macro`
   with `template ∈ {app_launcher, text_expansion, file_processor,
   window_manager, hotkey_action, custom}`. Atomically creates the
   group-membered macro with the right action sequence baked in.
   See `km_list_templates` for parameter schemas.
2. **Blank macro + append actions.** `km_macro_editor(operation="create",
   group_id=..., new_name=...)` → returns UUID → then drive
   `km_action_builder` / `km_control_flow` / `km_add_condition` against
   that UUID. Lets you build arbitrary macros but more round-trips.

After creation, bind triggers:
- Hotkey → `km_create_hotkey_trigger` (conflict-checked).
- System events (login / engine launch / wake) → `km_add_system_trigger`.
  **Warning:** rotates the macro's UID. Response includes
  `old_macro_id` and `new_macro_id`; rewrite any cross-macro
  `ExecuteMacro` references.
- Replace all triggers at once → `km_set_macro_triggers(triggers=[...])`.
  Also rotates the UID.
- General trigger CRUD → `km_trigger_crud` (list/get/add/update/remove/
  replace_all).

### "I want to edit an existing macro's actions"

| Operation | Tool |
|---|---|
| Append a single action | `km_action_builder(operation="append", action_type=..., action_config={...})` |
| Append a conditional | `km_add_condition` |
| Append a loop / switch / try / if | `km_control_flow(control_type=..., ...)` |
| Delete one action by 1-indexed position | `km_action_builder(operation="delete", action_index=N)` |
| Clear all actions | `km_action_builder(operation="clear")` |
| Rename / enable / disable / duplicate | `km_macro_editor(operation=...)` |
| Move between groups | `km_move_macro_to_group(create_group_if_missing=true)` |

### "I want to display something to the user"

- `km_notifications(notification_type="notification" | "hud" | "alert" |
  "sound", ...)`. **`alert` with `duration=N` auto-dismisses** after N
  seconds (post-f23cd30 fix). `hud` accepts a `position` parameter.

### "I want to control the desktop"

- `km_window_manager` — move / resize / arrange / get_info / get_screens.
  - Arrangements: `left_half / right_half / top_left_quarter` etc.
  - `arrange` response includes `window_info_source` field set to
    `"post_operation"` on success, `"pre_operation"` if the bounds re-
    query failed (use this to detect stale data).
- `km_application_control` — launch / quit / activate / list_running /
  get_state.
  - **Gotcha:** `get_state` with a bundle ID returns `state="unknown"`.
    Use the app **name** form instead (`app_identifier="Finder"`, not
    `"com.apple.finder"`).

### "I want to interact with the KM engine itself"

- `km_engine_control(operation="status")` — version + macro counts.
- `km_engine_control(operation="reload")` — reload after external edits.
- `km_engine_control(operation="calculate", expression="2*3+4")` →
  string result.
- `km_engine_control(operation="process_tokens", expression="...")` —
  expand `%Variable%X%`, `%Calculate%expr%`, `%ICUDateTime%fmt%`.
  **Limitation:** single-value system tokens (`%CurrentUser%`,
  `%FrontWindowName%`, `%FrontAppName%`, `%FinderInsertionLocation%`)
  return literally outside a macro execution context. KM's `process
  tokens` AppleScript verb only resolves these inside a running macro.
- `km_engine_control(operation="search_replace", text=..., search_pattern
  =..., replace_pattern=..., use_regex=true|false)`.

### "I want to read or set variables"

- `km_variable_manager(operation="set"|"get"|"delete"|"list", name=...,
  value=..., scope="global"|"local"|"instance"|"password")`.
- `password` scope is memory-only — never written to disk.
- For `local`/`instance` scope, pass `_instance_id` to target a specific
  execution.

### "I want to author a plug-in action"

- `km_create_plugin_action` — emits a `.kmactions` bundle on disk.
  `output_dir` must be under the MCP server CWD (path-traversal guard).

### "I want to refresh the cached action template catalog"

- `km_refresh_action_templates(macro_id=..., confirm=true, limit=...)`.
  **Heavy operation** — takes over the KM editor for a long walk
  through the Insert Action menu. Don't run while a user is editing.
- Accepts macro names (not just UUIDs); response includes
  `resolved_macro_id`.

---

## Action emitters — what `km_action_builder` understands

`km_action_builder(operation="append")` accepts these `action_type`
values directly:

- **Built-ins:** `pause`, `type_text`, `paste`, `set_variable`,
  `run_applescript`, `execute_macro`, `plug_in`, `paste_xml`.
- **Plug-ins** by installed identifier (see `km_list_action_types
  (category="plug_in")`).
- **Any built-in catalog identifier** from `km_list_action_types` — the
  emitter consumes keys named in that entry's `parameters` list.

For anything else (Notification, SpeakText, ExecuteShellScript,
Comment, ManipulateWindow, OpenURL, …) use `paste_xml` with the
canonical `<dict>...</dict>` body. The captured templates live in
`src/server/data/km_action_templates.json` keyed by `MacroActionType`.

**`execute_macro` append rebuilds the macro** (KM 11 rejects mid-macro
trigger / action UID mutation via `set xml`, so the tool exports →
edits → re-imports). The macro's UID rotates; response includes
`old_macro_id` / `new_macro_id`.

---

## Control-flow emitters — `km_control_flow`

| `control_type` | Required args | Inner-action params |
|---|---|---|
| `if_then_else` | `condition`, `operator`, `operand` | `actions_true`, `actions_false` |
| `for_loop` | `iterator`, `collection_dict={"type": ..., ...}` | `loop_actions` |
| `while_loop` | `condition`, `operator`, `operand` | `loop_actions` |
| `until_loop` | `condition`, `operator`, `operand` | `loop_actions` |
| `switch_case` | `source ∈ {Variable, Clipboard, NamedClipboard, Calculation, Text}`, `condition`, `cases` | `default_actions` |
| `try_catch` | none | `try_actions`, `catch_actions` |

Supported `for_loop` collection types (the `type` key inside
`collection_dict`): `Applications`, `Dictionaries`, `DictionaryKeys`,
`Files` (with `path`), `FinderSelection`, `FoundImages`, `JSON`,
`LinesIn` (with `text`), `PastClipboards`, `Range` (with `start`,
`end`), `SubstringsIn`, `Variables` (with `name`), `Volumes`.

Supported `switch_case` condition types per case: `Is`, `IsNot`,
`Contains`, `DoesNotContain`, `Otherwise` (the default sentinel).
Anything else gets rejected with `VALIDATION_ERROR` (no silent KM
coercion).

**Inner action types in `actions_*` / `loop_actions` / `try_actions`
etc.** are limited to what `action_builder` supports as inner items.
`execute_macro` as an inner type **is not supported** — for that
pattern, append the control-flow action and then append a separate
`execute_macro` action at the right index.

---

## Conditions — `km_add_condition`

Appends a single-condition `IfThenElse`. Supported `condition_type`:
`variable`, `text`, `application`, `calculation`. Legacy `system` /
`logic` are rejected (no direct KM mapping).

Operand format:
- `variable` / `text`: `"VarName=compare-value"` (or just `"VarName"`
  for exists/empty).
- `application`: bundle id or app name.
- `calculation`: the KM expression as a string.

**Round-6 bug fix (live):** operand value is now preserved correctly in
the emitted plist. Pre-fix the comparison value was silently dropped
to a placeholder. Verify with `osascript get xml of macro` —
`<key>VariableValue</key><string>YourValue</string>` should appear,
not `<string>value</string>`.

---

## Hard limitations — don't fight these

1. **Per-trigger enable/disable doesn't exist in KM 11.** Trigger
   plists store no `Disabled` key — KM strips any injected one. Use
   `km_macro_editor(operation="set_enabled", macro_id=..., enabled=...)`
   for the parent macro instead.
2. **`run_applescript` `StopOnFailure` defaults to false** in any MCP
   process started before commit `b689323`. A Claude Code restart is
   required to pick up the fix. Any macro using `run_applescript`
   inside this server today won't abort the macro on script failure.
3. **Macro UID rotation.** Three operations rebuild the macro and
   rotate its UID:
   - `km_add_system_trigger`
   - `km_set_macro_triggers`
   - `km_action_builder append action_type=execute_macro`
   Response always includes `old_macro_id` + `new_macro_id` so you can
   rewrite cross-macro `ExecuteMacro` references in other macros.
4. **Macro groups must be "Always Activated"** to be executable via
   `km_execute_macro` from outside a triggered context. If you create
   a sandbox group and try to execute its macros, you'll get a
   `do script found no macros with a matching name` error unless you
   flip the group to Always Activated in the KM editor.
5. **`km_application_control(operation="get_state")` with a bundle ID
   returns `state="unknown"`.** Use the app **name** form.
6. **System tokens outside a macro context return literally.** Tokens
   like `%CurrentUser%` only resolve when KM's process-tokens verb is
   called inside a running macro. For static reads from MCP, use
   `osascript` via `run_applescript` inside a macro, or read OS env
   vars on the client side.
7. **Window operations against Finder.** `move`/`resize`/`arrange`
   AppleScript succeeds but Finder reuses window indices in surprising
   ways. `get_info` may poll the menubar instead of the moved window.
   Target-app quirk, not a tool bug.

---

## Per-tool recipes

Each recipe shows a complete, working call. UUID placeholders use the
literal string `<UUID>` — substitute the real value returned by the
preceding tool.

### `km_create_macro`

```
km_create_macro(
  name="My App Launcher",
  template="app_launcher",
  group_name="My Macros",
  parameters={"app_name": "Finder", "bundle_id": "com.apple.finder"}
)
```

Template parameter schemas (call `km_list_templates` for the canonical
list):
- `app_launcher`: `app_name`, `bundle_id`
- `text_expansion`: `expansion_text`
- `file_processor`: `script` (shell)
- `window_manager`: `operation` ∈ {move|resize|arrange}, `x`, `y`,
  `width`, `height`
- `hotkey_action`: `action` ∈ {`open_app`|`type_text`|`run_script`},
  plus `text` or `app` or `script`, plus `hotkey`, `modifiers`

### `km_action_builder` — full pause + variable + applescript sequence

```
km_action_builder(operation="append", macro_id="<UUID>",
  action_type="pause", action_config={"seconds": 0.5})

km_action_builder(operation="append", macro_id="<UUID>",
  action_type="set_variable",
  action_config={"variable": "Local__Stamp",
                 "text": "%ICUDateTime%yyyy-MM-dd HH:mm:ss%"})

km_action_builder(operation="append", macro_id="<UUID>",
  action_type="run_applescript",
  action_config={"source": "return short user name of (system info)"})
```

### `km_action_builder` — `paste_xml` for actions outside the built-in catalog

```
km_action_builder(operation="append", macro_id="<UUID>",
  action_type="paste_xml",
  action_config={"xml": "<dict>"
    "<key>MacroActionType</key><string>Notification</string>"
    "<key>Title</key><string>Done</string>"
    "<key>Subtitle</key><string></string>"
    "<key>Text</key><string>Macro finished.</string>"
    "<key>SoundName</key><string></string>"
  "</dict>"})
```

### `km_control_flow` — for-each over a range

```
km_control_flow(macro_identifier="<UUID>", control_type="for_loop",
  iterator="i",
  collection_dict={"type": "Range", "start": "1", "end": "5"},
  loop_actions=[
    {"type": "set_variable", "variable": "Local__LastI", "text": "%Variable%i%"},
    {"type": "pause", "seconds": 0.1}
  ])
```

### `km_control_flow` — switch on variable with `Otherwise` fallback

```
km_control_flow(macro_identifier="<UUID>", control_type="switch_case",
  source="Variable", condition="Local__Mode",
  cases=[
    {"condition_type": "Is", "test_value": "demo",
     "actions": [{"type": "set_variable", "variable": "Local__Branch", "text": "demo"}]},
    {"condition_type": "Is", "test_value": "prod",
     "actions": [{"type": "set_variable", "variable": "Local__Branch", "text": "prod"}]},
  ],
  default_actions=[
    {"type": "set_variable", "variable": "Local__Branch", "text": "fell-through"}
  ])
```

### `km_control_flow` — try/catch wrapping a failing AppleScript

```
km_control_flow(macro_identifier="<UUID>", control_type="try_catch",
  try_actions=[{"type": "run_applescript",
                "source": "error \"intentional\" number 42"}],
  catch_actions=[{"type": "set_variable",
                  "variable": "Local__Caught", "text": "yes"}])
```

### `km_add_condition` — variable-equals branch

```
km_add_condition(macro_identifier="<UUID>",
  condition_type="variable",
  operator="equals",
  operand="Local__Mode=demo",
  action_on_true="MyHelperMacro")
```

The optional `action_on_true` / `action_on_false` emit one inner
`ExecuteMacro` each — for richer branches use `km_control_flow
(control_type="if_then_else")` instead.

### `km_create_hotkey_trigger`

```
km_create_hotkey_trigger(macro_id="<UUID>",
  key="f8", modifiers=["cmd", "ctrl", "opt"],
  activation_mode="pressed", check_conflicts=true)
```

If a conflict is found, response includes `suggest_alternatives` data
with conflict-free combos.

### `km_set_macro_triggers` — replace the whole trigger list

```
km_set_macro_triggers(macro_id="<UUID>", triggers=[
  {"MacroTriggerType": "HotKey", "KeyCode": 101, "Modifiers": 768,
   "FireType": "Pressed"},
  {"MacroTriggerType": "Login"},
])
```

Pass `triggers=[]` to strip all triggers (macro then only fires via
`ExecuteMacro`). UID rotates — read `new_macro_id` from the response.

### `km_notifications` — alert with auto-dismiss

```
km_notifications(notification_type="alert",
  title="Confirm",
  message="Proceed with cleanup?",
  buttons=["OK", "Cancel"],
  duration=5.0)
```

`duration` is honored (post-f23cd30). Without it, alerts block the
calling tool for the full 30-second AppleScript timeout if the user
doesn't click.

### `km_window_manager` — half-screen arrangement

```
km_window_manager(operation="arrange",
  window_identifier="Safari",
  arrangement="left_half",
  screen="main")
```

Response includes `window_info_source` — check for `"post_operation"`;
`"pre_operation"` means the post-arrange bounds re-query failed and
the bounds reflect pre-arrange state.

### `km_engine_control` — token expansion

```
km_engine_control(operation="process_tokens",
  expression="The time is %ICUDateTime%HH:mm:ss% and 2+2=%Calculate%2+2%.")
```

Returns `{processed, tokens_found, token_count}`.

### `km_variable_manager` — set a password variable

```
km_variable_manager(operation="set",
  name="SecretToken", value="<value>",
  scope="password")
```

Password-scope variables are memory-only and never written to disk;
they survive only within the running engine session.

### `km_create_plugin_action`

```
km_create_plugin_action(
  output_dir="/abs/path/under/server/cwd",
  name="My Plugin",
  title="Run %Param%Mode% on %Param%Target%",
  script_source="#!/bin/bash\necho \"$KMPARAM_Mode -> $KMPARAM_Target\"",
  parameters=[
    {"Label": "Mode", "Type": "PopupMenu",
     "Default": "fast", "Menu": "fast|slow"},
    {"Label": "Target", "Type": "String", "Default": ""}
  ],
  results=["Variable"]
)
```

---

## Failure codes you must handle

| Code | Meaning | Recovery |
|---|---|---|
| `KM_CONNECTION_FAILED` / `KME_UNREACHABLE` | KM Engine not running | Ask user to launch Keyboard Maestro. |
| `VALIDATION_ERROR` | Argument shape rejected | Read `message` — fields and recovery suggestion are explicit. |
| `MACRO_NOT_FOUND` / `NOT_FOUND_ERROR` | Identifier doesn't resolve | Try `km_list_macros` to find the right UUID or name. |
| `GROUP_NOT_FOUND` | Target group missing | Pass `create_group_if_missing=true` (move tool) or create explicitly. |
| `EXECUTE_MACRO_TARGET_NOT_FOUND` | `execute_macro` target isn't a known macro | `km_list_macros` and pass the real name/UUID. |
| `EXECUTION_ERROR: do script found no macros with a matching name` | Macro exists but its group isn't currently active | Flip the parent group to "Always Activated" in the KM editor. |
| `KM_REJECTED_XML` | KM AppleScript refused a `paste_xml` body | Check the `<dict>` is valid and the `MacroActionType` is one KM 11 knows. |
| `UNSUPPORTED_TEMPLATE_ACTION` | Template `inner action` not in the recipe list | Use `template="custom"` + chain `km_action_builder`. |
| `UNSUPPORTED_OPERATION` | Tool path intentionally unimplemented | Read recovery suggestion — usually points to an alternative tool. |
| `SECURITY_BLOCKED` | Target app is on the blocklist (Keychain etc.) | Choose a different app or use `force_quit=false`. |
| `TOGGLE_FAILED` (trigger set_enabled) | KM 11 per-trigger enabled bit doesn't exist | Use `km_macro_editor(operation="set_enabled")` on the parent macro. |
| `EXECUTE_MACRO_TARGET_NOT_FOUND` | Passed a group name | Pass a macro name/UUID instead. |

---

## Recommended call patterns

### Authoring a new macro from scratch

1. `km_macro_group_manager(operation="list")` — confirm target group exists.
2. `km_macro_editor(operation="create", group_id=..., new_name=...)` —
   capture `macro_id` from response.
3. Loop: `km_action_builder(operation="append", ...)` /
   `km_control_flow(...)` / `km_add_condition(...)` to build the body.
4. `km_create_hotkey_trigger(macro_id=...)` if user-runnable.
5. `km_action_builder(operation="list", macro_id=...)` — verify final
   action count + order.

### Editing an existing macro safely

1. `km_action_builder(operation="list", macro_id=...)` — read current
   structure, plan changes.
2. If appending `execute_macro` or system triggers, **save the UUID
   beforehand**: response will rotate it. Use the `new_macro_id` from
   the response for all subsequent calls.
3. After mutation, `km_action_builder(operation="list")` again to
   verify.

### Bulk operations

- For multiple independent appends, send them in parallel **only if
  order doesn't matter**. KM serializes plist edits internally but
  AppleScript dispatch can race.
- For ordered sequences, await each append before issuing the next.

### Debugging

- `osascript -e 'tell application "Keyboard Maestro" to get xml of
  macro id "<UUID>"'` gives you the raw plist for any macro. Use this
  to verify what KM actually stored after an emit.
- `km_engine_control(operation="status")` reports engine version,
  reachability, and macro counts — a fast health check.

---

## What this server **does not** do

- No mouse / keyboard simulation outside macros — use the
  `computer-use` MCP.
- No screen capture, OCR, or image recognition.
- No browser automation — use Playwright or the chrome-extension MCP.
- No file I/O outside the project sandbox (path-traversal guard).
- No network requests, IoT, voice, identity, or LLM calls.

Use these tools to drive Keyboard Maestro. For anything else, pick the
right MCP for the job.
