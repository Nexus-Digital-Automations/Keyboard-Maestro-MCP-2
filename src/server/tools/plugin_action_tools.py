"""Keyboard Maestro third-party plug-in: build new bundles and discover installed ones.

Owner: Keyboard-Maestro-MCP-2 server team. Two public surfaces share this
module because they speak the same plist schema:

- ``km_build_plugin_action`` emits a self-contained ``.kmactions`` bundle (the
  format Keyboard Maestro loads from ``~/Library/Application Support/Keyboard
  Maestro/Keyboard Maestro Actions/``). The MCP caller supplies a JSON spec —
  sidebar name, editor title, script source, parameter fields, allowed result
  targets — and the tool writes ``Keyboard Maestro Action.plist``, an
  executable script, and (optionally) an ``Icon.png`` into a workspace
  directory of the caller's choosing. No system writes; the user installs the
  bundle by copying it into KM's Actions folder themselves.
- ``_scan_installed_plugins`` (used by ``km_list_action_types``) parses the
  same plist schema in reverse: read every installed bundle and return a
  normalized list. Read-only.

The plist schema matches what shipping third-party plug-ins use (verified
against eight installed plug-ins): ``Name`` (sidebar label), ``Title`` (canvas
display string with ``%Param%Label%`` placeholders), ``Script``, ``Parameters``,
``Results`` (pipe-joined list of allowed result targets), and optional
``Author``, ``Help``, ``HelpURL``, ``KeyWords``, ``Icon`` (filename). KM ignores
unknown keys, so older bundles emitted with ``Identifier``/``Authentication``/
``Timeout`` still loaded but had broken Title/Results/Icon rendering.

Failure modes: unknown result target, ``parameters[i]`` missing ``Label`` /
``Type``, ``PopupMenu`` parameter without ``Menu``, ``output_dir`` outside the
current working directory (path-traversal guard), bundle already exists with
``on_existing='error'``. Each is returned as the standard MCP envelope
``{success: False, error: {...}}`` — no partial bundles on disk.
"""

import base64
import functools
import logging
import os
import plistlib
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

logger = logging.getLogger(__name__)

# Result-target tokens emitted in the pipe-delimited ``Results`` plist value.
# Verified against installed third-party plug-ins; ``TypedString``/``Token``/
# ``Asynchronously`` are documented KM result modes not represented in the
# sample set but accepted defensively.
RESULT_TARGETS = frozenset(
    {
        "None",
        "Variable",
        "Clipboard",
        "TypedString",
        "Pasting",
        "Typing",
        "Window",
        "Briefly",
        "Token",
        "Asynchronously",
    }
)
PARAM_TYPES = frozenset(
    {
        "String",
        "Text",
        "Password",
        "Calculation",
        "PopupMenu",
        "Checkbox",
        "Hidden",
        "TokenString",
        "TokenText",
    }
)
ON_EXISTING_MODES = frozenset({"error", "replace"})
ICON_FILENAME = "Icon.png"

# Where Keyboard Maestro Editor scans for plug-in bundles at launch. Override
# via env for tests; we never write here from this module.
INSTALLED_KM_ACTIONS_DIR = (
    Path.home() / "Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions"
)

FOLDER_FORBIDDEN_RE = re.compile(r'[/\\:?<>|"\x00-\x1f]')
SCRIPT_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
HTTPS_URL_RE = re.compile(r"^https?://")


def _validate_parameter(param: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalise one parameter dict, raising ValueError on any defect.

    Returns the cleaned dict ready for plist serialisation. ``Menu`` is joined
    with ``|`` — verified against shipping plug-ins like ``Choose File(s)``,
    whose menu plist value is ``'Folder...|Desktop|Home|...'``. ``Default`` is
    coerced to a str so the plist is unambiguous (KM accepts only strings for
    default values).
    """
    if not isinstance(param, dict):
        raise ValueError(f"parameters[{index}] must be a dict; got {type(param).__name__}")
    label = param.get("Label")
    ptype = param.get("Type")
    if not isinstance(label, str) or not label.strip() or len(label) > 200:
        raise ValueError(f"parameters[{index}].Label must be a non-empty string ≤200 chars")
    if ptype not in PARAM_TYPES:
        raise ValueError(
            f"parameters[{index}].Type must be one of {sorted(PARAM_TYPES)}; got {ptype!r}"
        )

    cleaned: dict[str, Any] = {"Label": label, "Type": ptype}
    default = param.get("Default")
    menu = param.get("Menu")

    if ptype == "PopupMenu":
        if not isinstance(menu, list) or not menu or not all(isinstance(m, str) and m for m in menu):
            raise ValueError(
                f"parameters[{index}].Menu must be a non-empty list of non-empty strings "
                f"when Type='PopupMenu'"
            )
        cleaned["Menu"] = "|".join(menu)
    elif menu is not None:
        raise ValueError(
            f"parameters[{index}].Menu is only valid when Type='PopupMenu'; "
            f"remove it for Type={ptype!r}"
        )

    if default is not None:
        if ptype == "Checkbox":
            cleaned["Default"] = "1" if default in (True, "1", 1, "true", "True") else "0"
        else:
            cleaned["Default"] = str(default)
    return cleaned


def _sanitize_folder_basename(display_name: str) -> str:
    """KM folder names allow most characters; strip filesystem-hostile ones only."""
    cleaned = FOLDER_FORBIDDEN_RE.sub("", display_name).strip() or "Untitled"
    return cleaned[:200]


def _resolve_under_cwd(output_dir: str) -> Path:
    """Reject paths that escape the working directory after symlink resolution."""
    if not output_dir or not Path(output_dir).is_absolute():
        raise ValueError(f"output_dir must be an absolute path; got {output_dir!r}")
    resolved = Path(output_dir).resolve()
    cwd = Path.cwd().resolve()
    if resolved != cwd and not resolved.is_relative_to(cwd):
        raise ValueError(
            f"output_dir {resolved} is outside the working directory {cwd}; "
            f"path traversal blocked"
        )
    if not resolved.exists():
        raise ValueError(f"output_dir {resolved} does not exist")
    if not resolved.is_dir():
        raise ValueError(f"output_dir {resolved} is not a directory")
    return resolved


def _build_plist(
    *,
    name: str,
    title: str,
    script_filename: str,
    parameters: list[dict[str, Any]],
    results: list[str],
    author: str | None,
    help_text: str | None,
    help_url: str | None,
    keywords: list[str] | None,
    has_icon: bool,
) -> dict[str, Any]:
    """Return the dict KM serialises into ``Keyboard Maestro Action.plist``.

    Schema matches shipping third-party plug-ins: ``Name`` is the sidebar label,
    ``Title`` is the canvas display string with ``%Param%Label%`` placeholders,
    ``Results`` is a pipe-joined enumeration of allowed result targets, ``Icon``
    is a filename (the PNG sits alongside in the bundle).
    """
    plist: dict[str, Any] = {
        "Name": name,
        "Title": title,
        "Script": script_filename,
        "Results": "|".join(results),
        "Parameters": parameters,
    }
    if author:
        plist["Author"] = author
    if help_text:
        plist["Help"] = help_text
    if help_url:
        if not HTTPS_URL_RE.match(help_url):
            raise ValueError(f"help_url must start with http:// or https://; got {help_url!r}")
        plist["HelpURL"] = help_url
    if keywords:
        if not all(isinstance(k, str) and k for k in keywords):
            raise ValueError("keywords must be a list of non-empty strings")
        plist["KeyWords"] = list(keywords)
    if has_icon:
        plist["Icon"] = ICON_FILENAME
    return plist


def _validate_results(results: list[str]) -> list[str]:
    """Each entry must be a known KM result target; empty list rejected."""
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a non-empty list of result-target strings")
    for i, entry in enumerate(results):
        if entry not in RESULT_TARGETS:
            raise ValueError(
                f"results[{i}] = {entry!r} is not a known KM result target; "
                f"expected one of {sorted(RESULT_TARGETS)}"
            )
    return list(results)


def _decode_icon(icon_base64: str) -> bytes:
    """Decode a base64 PNG, raising ValueError on bad encoding or wrong magic."""
    try:
        png = base64.b64decode(icon_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"icon_base64 is not valid base64: {exc}") from exc
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("icon_base64 must decode to a PNG (magic bytes mismatch)")
    return png


def _error_envelope(code: str, message: str, *, field: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "recovery_suggestion": "Adjust the failing field and retry.",
    }
    if field:
        error["field"] = field
    return {
        "success": False,
        "error": error,
        "metadata": {
            "tool": "km_build_plugin_action",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def _normalize_plugin_parameters(raw_params: list[Any]) -> list[dict[str, Any]]:
    """Map the bundle plist's ``Parameters`` array into client-facing dicts.

    Drops entries without a ``Label`` (KM uses ``Label`` to render the field
    and to derive the parameter key in the macro's ExecutePlugIn dict — a
    plug-in with no Label is unusable).
    """
    out: list[dict[str, Any]] = []
    for raw in raw_params or []:
        if not isinstance(raw, dict):
            continue
        label = raw.get("Label")
        if not isinstance(label, str) or not label:
            continue
        menu = raw.get("Menu")
        out.append({
            "label": label,
            "type": raw.get("Type", "String"),
            "default": raw.get("Default"),
            "menu_options": menu.split("|") if isinstance(menu, str) and menu else [],
        })
    return out


def _scan_installed_plugins() -> list[dict[str, Any]]:
    """List installed Keyboard Maestro third-party plug-ins, one dict per bundle.

    Reads each ``Keyboard Maestro Action.plist`` under
    ``INSTALLED_KM_ACTIONS_DIR`` (or the path in the ``KM_PLUGIN_ACTIONS_DIR``
    env var, used by tests). Bundles whose plist is malformed or missing
    ``Name``/``Title`` are skipped with a warning — KM itself ignores them.

    Returns ``[]`` if the directory does not exist. Read-only.
    """
    root_override = os.environ.get("KM_PLUGIN_ACTIONS_DIR")
    root = Path(root_override) if root_override else INSTALLED_KM_ACTIONS_DIR
    if not root.is_dir():
        return []
    plugins: list[dict[str, Any]] = []
    for bundle in sorted(root.iterdir()):
        if not bundle.is_dir() or bundle.name.startswith("."):
            continue
        plist_path = bundle / "Keyboard Maestro Action.plist"
        if not plist_path.is_file():
            continue
        try:
            with plist_path.open("rb") as fp:
                spec = plistlib.load(fp)
        except (plistlib.InvalidFileException, OSError, ValueError) as exc:
            logger.warning("skipping plug-in %s: %s", bundle.name, exc)
            continue
        name, title = spec.get("Name"), spec.get("Title")
        if not isinstance(name, str) or not isinstance(title, str):
            continue
        results = spec.get("Results") or ""
        # KM plug-in plists are inconsistent: docs say "KeyWords" but some
        # bundles ship "Keywords" — accept both. Help URL and Author are
        # plist-optional and we just forward whatever's there.
        keywords_raw = spec.get("KeyWords") or spec.get("Keywords") or []
        keywords = [str(k) for k in keywords_raw if isinstance(k, str) and k]
        plugins.append({
            "identifier": name,
            "title": title,
            "help": spec.get("Help"),
            "help_url": spec.get("HelpURL"),
            "author": spec.get("Author"),
            "keywords": keywords,
            "parameters": _normalize_plugin_parameters(spec.get("Parameters") or []),
            "result_targets": [s for s in results.split("|") if s],
            "bundle_path": str(bundle),
        })
    return plugins


async def km_create_plugin_action(
    output_dir: Annotated[
        str,
        Field(description="Absolute path under CWD where the .kmactions folder will be created."),
    ],
    name: Annotated[
        str,
        Field(
            description="Sidebar label under Plug In > Third Party. Also the bundle folder name.",
            min_length=1,
            max_length=200,
        ),
    ],
    title: Annotated[
        str,
        Field(
            description=(
                "Canvas display title. Supports '%Param%Label%' to interpolate parameter "
                "values, e.g. \"Wait for button '%Param%Title%' in %Param%App%\"."
            ),
            min_length=1,
            max_length=400,
        ),
    ],
    script_source: Annotated[
        str,
        Field(description="Full script body, including shebang. Written to <script_filename>.", min_length=1),
    ],
    script_filename: Annotated[
        str,
        Field(description="Filename of the script inside the bundle (chmod 0755 on write)."),
    ] = "script",
    parameters: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Parameter field specs. Each: {Label, Type, Default?, Menu?}."),
    ] = None,
    results: Annotated[
        list[str] | None,
        Field(
            description=(
                "Allowed result targets, pipe-joined into the plist's Results key. "
                "Pick from None, Variable, Clipboard, TypedString, Pasting, Typing, "
                "Window, Briefly, Token, Asynchronously. Defaults to ['Variable']."
            ),
        ),
    ] = None,
    author: Annotated[
        str | None,
        Field(description="Optional author name embedded in the plist.", max_length=200),
    ] = None,
    help_text: Annotated[
        str | None,
        Field(description="Optional inline help shown in KM's editor."),
    ] = None,
    help_url: Annotated[
        str | None,
        Field(description="Optional http(s) URL to fuller docs."),
    ] = None,
    keywords: Annotated[
        list[str] | None,
        Field(description="Optional search keywords for KM's action picker."),
    ] = None,
    icon_base64: Annotated[
        str | None,
        Field(description="Optional base64-encoded PNG written to Icon.png inside the bundle."),
    ] = None,
    on_existing: Annotated[
        str,
        Field(description="'error' (default) refuses overwrite; 'replace' rmtrees first."),
    ] = "error",
    ctx: Context = None,
) -> dict[str, Any]:
    """Emit a Keyboard Maestro third-party plug-in action bundle to disk.

    Architecture:
        - Pattern: builder that produces a filesystem artifact (DTO in = bundle out).
        - Security: path-traversal guard (resolve + is_relative_to CWD); plistlib
          handles XML escaping; script written with 0o755 (owner-execute, world-read);
          icon PNG validated by magic bytes before write.
        - Performance: one plist serialisation + two-to-three file writes; O(n) in
          script and icon bytes.

    Failure modes are documented in the module docstring. Returns the standard
    MCP envelope: ``{success, data|error, metadata}``.
    """
    started = time.monotonic()
    if ctx:
        await ctx.info(f"Building plug-in action '{name}'")

    try:
        if on_existing not in ON_EXISTING_MODES:
            raise ValueError(
                f"on_existing must be one of {sorted(ON_EXISTING_MODES)}; got {on_existing!r}"
            )
        if not SCRIPT_FILENAME_RE.match(script_filename):
            raise ValueError(
                f"script_filename must match [A-Za-z0-9._-]{{1,128}}; got {script_filename!r}"
            )
        cleaned_results = _validate_results(results or ["Variable"])
        cleaned_params = [
            _validate_parameter(p, i) for i, p in enumerate(parameters or [])
        ]
        icon_bytes = _decode_icon(icon_base64) if icon_base64 else None
        plist_data = _build_plist(
            name=name,
            title=title,
            script_filename=script_filename,
            parameters=cleaned_params,
            results=cleaned_results,
            author=author,
            help_text=help_text,
            help_url=help_url,
            keywords=keywords,
            has_icon=icon_bytes is not None,
        )
        output_root = _resolve_under_cwd(output_dir)
    except ValueError as exc:
        logger.warning("km_build_plugin_action validation rejected input: %s", exc)
        return _error_envelope("VALIDATION_FAILED", str(exc))

    bundle = output_root / f"{_sanitize_folder_basename(name)}.kmactions"
    if bundle.exists():
        if on_existing == "error":
            return _error_envelope(
                "BUNDLE_EXISTS",
                f"{bundle} already exists; pass on_existing='replace' to overwrite.",
                field="output_dir",
            )
        try:
            shutil.rmtree(bundle)
        except OSError as exc:
            logger.exception("Failed to remove existing bundle %s", bundle)
            return _error_envelope("BUNDLE_REPLACE_FAILED", f"rmtree {bundle}: {exc}")

    try:
        bundle.mkdir(parents=True, exist_ok=False)
        plist_path = bundle / "Keyboard Maestro Action.plist"
        with plist_path.open("wb") as fp:
            plistlib.dump(plist_data, fp, fmt=plistlib.FMT_XML, sort_keys=False)
        script_path = bundle / script_filename
        script_path.write_text(script_source, encoding="utf-8")
        script_path.chmod(0o755)
        icon_path: Path | None = None
        if icon_bytes is not None:
            icon_path = bundle / ICON_FILENAME
            icon_path.write_bytes(icon_bytes)
    except OSError as exc:
        logger.exception("Filesystem write failed while building %s", bundle)
        return _error_envelope("BUNDLE_WRITE_FAILED", f"writing {bundle}: {exc}")

    duration_ms = int((time.monotonic() - started) * 1000)
    if ctx:
        await ctx.info(f"Wrote {bundle} in {duration_ms}ms")
    return {
        "success": True,
        "data": {
            "bundle_path": str(bundle),
            "plist_path": str(plist_path),
            "script_path": str(script_path),
            "icon_path": str(icon_path) if icon_path else None,
            "plist_size_bytes": plist_path.stat().st_size,
            "parameter_count": len(cleaned_params),
            "results": cleaned_results,
        },
        "metadata": {
            "tool": "km_build_plugin_action",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "install_hint": (
                "Copy the bundle into "
                "~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/ "
                "(create it if missing), then quit and reopen the Keyboard Maestro "
                "Editor — plug-ins are scanned at Editor launch; reloading the Engine "
                "alone does not refresh the action picker."
            ),
        },
    }


# @deprecated — old name kept registered for one release so existing clients
# don't break. functools.wraps copies __signature__ so FastMCP's discovery
# generates the same parameter schema as km_create_plugin_action.
@functools.wraps(km_create_plugin_action)
async def km_build_plugin_action(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated alias for km_create_plugin_action. Use the new name."""
    logger.warning(
        "km_build_plugin_action is deprecated; rename calls to km_create_plugin_action.",
    )
    return await km_create_plugin_action(*args, **kwargs)

