"""KM action XML templates: load captured plist templates, render with overrides.

Owned by: server/tools — read-path support for km_action_builder. The captured
XML in km_action_templates.json comes from KM's own editor (Insert Action by
Name → read xml of action) and is therefore guaranteed to be a valid KM action
plist. The render function applies user-supplied parameter values via the
template's `parameter_paths` map (user-facing label → plist key) so the engine
never needs per-action knowledge of KM's plist schema.

@stable
"""

from __future__ import annotations

import json
import plistlib
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TEMPLATES_PATH = _DATA_DIR / "km_action_templates.json"
_CATALOG_PATH = _DATA_DIR / "km_builtin_actions.json"

# Plist header/footer wrap a stored <dict> body so plistlib.loads can parse it.
# Stored templates omit the wrapper because KM's `make new action` expects a
# bare <dict> element, not a full plist document.
_PLIST_HEAD = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    b'<plist version="1.0">\n'
)
_PLIST_TAIL = b"\n</plist>\n"


@lru_cache(maxsize=1)
def load_templates() -> dict[str, dict[str, Any]]:
    """Return {MacroActionType: template_entry} from km_action_templates.json.

    Strips `_meta` and any other underscore-prefixed top-level keys.
    Raises FileNotFoundError if the data file is missing.
    """
    raw = json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    """Return the list of catalog entries from km_builtin_actions.json.

    Raises FileNotFoundError if the data file is missing.
    """
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    return list(raw.get("actions", []))


def find_macro_action_type(identifier: str) -> str | None:
    """Map a catalog identifier (e.g. 'speak_text') to its KM MacroActionType.

    Returns None if the identifier is not in the catalog.
    """
    for entry in load_catalog():
        if entry.get("identifier") == identifier:
            return entry.get("MacroActionType")
    return None


def render_action_xml(macro_action_type: str, params: dict[str, Any]) -> str | None:
    """Render an action <dict> XML by applying params to the captured template.

    Returns None if no template is registered for `macro_action_type` — callers
    use this to fall through to a different emitter or raise UNSUPPORTED.
    Raises ValueError if a parameter key isn't declared in the template's
    parameter_paths, or if the stored template XML is malformed.
    """
    template = load_templates().get(macro_action_type)
    if template is None:
        return None
    plist = _parse_dict_body(template["xml"])
    _apply_params(plist, template.get("parameter_paths", {}), params)
    return _serialize_dict_body(plist)


def _parse_dict_body(dict_xml: str) -> dict[str, Any]:
    body = _PLIST_HEAD + dict_xml.encode("utf-8") + _PLIST_TAIL
    parsed = plistlib.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("template XML must be a plist <dict>")
    return parsed


def _serialize_dict_body(plist: dict[str, Any]) -> str:
    full = plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=False).decode("utf-8")
    start = full.index("<dict>")
    end = full.rindex("</dict>") + len("</dict>")
    return full[start:end]


def _apply_params(
    plist: dict[str, Any],
    paths: dict[str, str],
    params: dict[str, Any],
) -> None:
    for user_key, value in params.items():
        plist_key = paths.get(user_key)
        if plist_key is None:
            raise ValueError(f"unknown parameter {user_key!r}")
        plist[plist_key] = value
