"""Keyboard Maestro third-party plug-in action builder.

Owner: Keyboard-Maestro-MCP-2 server team. Provides ``km_build_plugin_action``,
which emits a self-contained ``.kmactions`` bundle (the format Keyboard Maestro
loads from ``~/Library/Application Support/Keyboard Maestro/Keyboard Maestro
Actions/``). The MCP caller supplies a JSON spec — display name, reverse-DNS
identifier, script source, parameter fields, result mode — and the tool writes
``Keyboard Maestro Action.plist`` plus an executable script into a workspace
directory of the caller's choosing. No system writes; the user installs the
bundle by copying it into KM's Actions folder themselves.

Failure modes: bad identifier syntax (``com.acme.foo`` form required), unknown
``results_type`` / ``authentication`` enum value, ``parameters[i]`` missing
``Label``/``Type``, ``PopupMenu`` parameter without ``Menu``, ``output_dir``
outside the current working directory (path-traversal guard), bundle already
exists with ``on_existing='error'``. Each is returned as the standard MCP
envelope ``{success: False, error: {...}}`` — no partial bundles on disk.
"""

from __future__ import annotations

import base64
import logging
import plistlib
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import Context

logger = logging.getLogger(__name__)

RESULTS_TYPES = frozenset(
    {"None", "Variable", "Clipboard", "TypedString", "Window", "Briefly"}
)
AUTH_LEVELS = frozenset({"None", "Admin"})
PARAM_TYPES = frozenset(
    {
        "String",
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

# Reverse-DNS: lowercase letters/digits/hyphens, ≥2 dotted segments, each segment
# starts/ends with [a-z0-9]. Matches Apple's UTI / KM conventions.
IDENTIFIER_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$"
)
FOLDER_FORBIDDEN_RE = re.compile(r'[/\\:?<>|"\x00-\x1f]')
SCRIPT_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
HTTPS_URL_RE = re.compile(r"^https?://")


def _validate_identifier(identifier: str) -> str:
    """Reverse-DNS, lowercased. Required by KM to dedupe plug-ins across users."""
    if not isinstance(identifier, str) or not IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"identifier must be reverse-DNS (e.g. 'com.acme.wait-for-button'); "
            f"got {identifier!r}"
        )
    return identifier


def _validate_parameter(param: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalise one parameter dict, raising ValueError on any defect.

    Returns the cleaned dict ready for plist serialisation. ``Menu`` is joined
    with newlines (KM's wire format), ``Default`` is coerced to a str so the
    plist is unambiguous (KM accepts only strings for default values).
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
        cleaned["Menu"] = "\n".join(menu)
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
    identifier: str,
    author: str,
    script_filename: str,
    parameters: list[dict[str, Any]],
    results_type: str,
    timeout_seconds: int,
    authentication: str,
    help_text: str | None,
    help_url: str | None,
    keywords: list[str] | None,
    icon_base64: str | None,
) -> dict[str, Any]:
    plist: dict[str, Any] = {
        "Name": name,
        "Identifier": identifier,
        "Author": author,
        "Script": script_filename,
        "Authentication": authentication,
        "Timeout": timeout_seconds,
        "ResultsType": results_type,
        "Parameters": parameters,
    }
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
    if icon_base64:
        try:
            plist["Icon"] = base64.b64decode(icon_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"icon_base64 is not valid base64: {exc}") from exc
    return plist


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
            "timestamp": datetime.now(UTC).isoformat(),
        },
    }


async def km_build_plugin_action(
    output_dir: Annotated[
        str,
        Field(description="Absolute path under CWD where the .kmactions folder will be created."),
    ],
    name: Annotated[
        str,
        Field(description="Display name shown in KM's actions sidebar.", min_length=1, max_length=200),
    ],
    identifier: Annotated[
        str,
        Field(description="Reverse-DNS identifier, e.g. 'com.acme.wait-for-button'.", max_length=200),
    ],
    author: Annotated[
        str,
        Field(description="Author name embedded in the plist.", min_length=1, max_length=200),
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
    results_type: Annotated[
        str,
        Field(description="KM ResultsType: None|Variable|Clipboard|TypedString|Window|Briefly."),
    ] = "Variable",
    timeout_seconds: Annotated[
        int,
        Field(description="Script timeout in seconds (1..3600).", ge=1, le=3600),
    ] = 60,
    authentication: Annotated[
        str,
        Field(description="KM Authentication mode: 'None' or 'Admin'."),
    ] = "None",
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
        Field(description="Optional base64-encoded PNG used as the action icon."),
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
          handles XML escaping; script written with 0o755 (owner-execute, world-read).
        - Performance: one plist serialisation + two file writes; O(n) in script bytes.

    Failure modes are documented in the module docstring. Returns the standard
    MCP envelope: ``{success, data|error, metadata}``.
    """
    started = time.monotonic()
    if ctx:
        await ctx.info(f"Building plug-in action '{name}' ({identifier})")

    try:
        if results_type not in RESULTS_TYPES:
            raise ValueError(
                f"results_type must be one of {sorted(RESULTS_TYPES)}; got {results_type!r}"
            )
        if authentication not in AUTH_LEVELS:
            raise ValueError(
                f"authentication must be one of {sorted(AUTH_LEVELS)}; got {authentication!r}"
            )
        if on_existing not in ON_EXISTING_MODES:
            raise ValueError(
                f"on_existing must be one of {sorted(ON_EXISTING_MODES)}; got {on_existing!r}"
            )
        if not SCRIPT_FILENAME_RE.match(script_filename):
            raise ValueError(
                f"script_filename must match [A-Za-z0-9._-]{{1,128}}; got {script_filename!r}"
            )
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
            raise ValueError(
                f"timeout_seconds must be an int in [1, 3600]; got {timeout_seconds!r}"
            )
        identifier = _validate_identifier(identifier)
        cleaned_params = [
            _validate_parameter(p, i) for i, p in enumerate(parameters or [])
        ]
        plist_data = _build_plist(
            name=name,
            identifier=identifier,
            author=author,
            script_filename=script_filename,
            parameters=cleaned_params,
            results_type=results_type,
            timeout_seconds=timeout_seconds,
            authentication=authentication,
            help_text=help_text,
            help_url=help_url,
            keywords=keywords,
            icon_base64=icon_base64,
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
            "plist_size_bytes": plist_path.stat().st_size,
            "parameter_count": len(cleaned_params),
            "identifier": identifier,
        },
        "metadata": {
            "tool": "km_build_plugin_action",
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_ms": duration_ms,
            "install_hint": (
                "Copy the bundle into "
                "~/Library/Application Support/Keyboard Maestro/Keyboard Maestro Actions/ "
                "(create it if missing), then restart Keyboard Maestro Engine."
            ),
        },
    }
