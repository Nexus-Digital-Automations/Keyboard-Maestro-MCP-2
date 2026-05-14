"""KM 11 ``For Each`` action plist emitter.

Owner: km-mcp control-flow workstream (session 20260514-145029-69306).

Emits the ``<dict>`` plist KM accepts as a ``For`` macro action,
with a populated ``Collections/CollectionList`` array. Companion to
``km_if_then_else_xml`` — same shape, different MacroActionType.

Captured 2026-05-14 by importing skeleton ``.kmmacros`` files and
reading back KM's normalized output. The 13 supported CollectionType
values come from KM 11's ``C{Type}Collection.nib`` editor resources;
any other value is rejected here rather than letting KM silently strip
the entry on import.

@stable
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape as _xml_escape

# Each entry: (key-emitter callable). Callables accept the user-supplied
# config dict and return the inner ``<key>...</key>...`` body (no surrounding
# <dict>) for the CollectionList entry. CollectionType is added by the caller.
_CollectionEmitter = "callable[[dict[str, Any]], str]"


class UnsupportedCollectionType(ValueError):
    """Raised when collection_type is not one of KM 11's 13 supported types."""


def escape(text: str) -> str:
    """XML-escape text for inclusion in a plist <string>."""
    return _xml_escape(str(text), {'"': "&quot;"})


def build_for_loop_xml(
    variable_name: str,
    collection_xml: str,
    actions_xml: str = "",
    *,
    timeout_aborts: bool = True,
) -> str:
    """Wrap a populated CollectionList entry + inner actions into a For action.

    ``collection_xml`` is the pre-rendered CollectionList ``<dict>`` entry from
    ``build_collection_dict``. ``actions_xml`` is the concatenated action
    ``<dict>`` strings (KM expects them as ``<array>`` items).
    """
    return (
        "<dict>"
        "<key>Actions</key>"
        f"<array>{actions_xml}</array>"
        "<key>Collections</key>"
        "<dict>"
        "<key>CollectionList</key>"
        f"<array>{collection_xml}</array>"
        "</dict>"
        "<key>MacroActionType</key>"
        "<string>For</string>"
        "<key>TimeOutAbortsMacro</key>"
        f"<{'true' if timeout_aborts else 'false'}/>"
        "<key>Variable</key>"
        f"<string>{escape(variable_name)}</string>"
        "</dict>"
    )


def build_collection_dict(collection_type: str, **kwargs: Any) -> str:
    """Return one CollectionList ``<dict>`` entry for the given KM collection type.

    Supported collection_type values (KM 11 editor resources):
      Applications, Dictionaries, DictionaryKeys, Files, FinderSelection,
      FoundImages, JSON, LinesIn, PastClipboards, Range, SubstringsIn,
      Variables, Volumes.

    Type-specific kwargs are passed through verbatim. Missing keys are
    omitted; KM fills in defaults during plist normalization.
    """
    emitter = _COLLECTION_EMITTERS.get(collection_type)
    if emitter is None:
        raise UnsupportedCollectionType(
            f"collection_type {collection_type!r} not supported. "
            f"Supported: {sorted(_COLLECTION_EMITTERS)}.",
        )
    inner = emitter(kwargs)
    return (
        "<dict>"
        f"<key>CollectionType</key><string>{escape(collection_type)}</string>"
        f"{inner}"
        "</dict>"
    )


def _no_extra_keys(_: dict[str, Any]) -> str:
    return ""


def _dictionary_keys(cfg: dict[str, Any]) -> str:
    name = cfg.get("dictionary", "")
    return f"<key>Dictionary</key><string>{escape(name)}</string>"


def _files(cfg: dict[str, Any]) -> str:
    parts = [f"<key>Path</key><string>{escape(cfg.get('path', ''))}</string>"]
    if "include_invisibles" in cfg:
        parts.append(
            f"<key>IncludeInvisibles</key><{'true' if cfg['include_invisibles'] else 'false'}/>",
        )
    if "recursive" in cfg:
        parts.append(
            f"<key>Recursive</key><{'true' if cfg['recursive'] else 'false'}/>",
        )
    sort_order = cfg.get("sort_order", "Alphabetical")
    parts.append(f"<key>SortOrder</key><string>{escape(sort_order)}</string>")
    return "".join(parts)


def _found_images(cfg: dict[str, Any]) -> str:
    fuzz = int(cfg.get("fuzz", 15))
    display = cfg.get("display_matches", False)
    area_type = cfg.get("screen_area_type", "ScreenAll")
    return (
        f"<key>DisplayMatches</key><{'true' if display else 'false'}/>"
        f"<key>Fuzz</key><integer>{fuzz}</integer>"
        "<key>ScreenArea</key>"
        f"<dict><key>ScreenAreaType</key><string>{escape(area_type)}</string></dict>"
    )


def _json_collection(cfg: dict[str, Any]) -> str:
    result = cfg.get("result_type", "Key")
    text = cfg.get("text", "")
    return (
        f"<key>ResultType</key><string>{escape(result)}</string>"
        f"<key>Text</key><string>{escape(text)}</string>"
    )


def _lines_in(cfg: dict[str, Any]) -> str:
    blanks = cfg.get("include_blank_lines", False)
    source = cfg.get("source", "Clipboard")
    parts = [
        f"<key>IncludeBlankLines</key><{'true' if blanks else 'false'}/>",
        f"<key>Source</key><string>{escape(source)}</string>",
    ]
    if "variable" in cfg:
        parts.append(f"<key>Variable</key><string>{escape(cfg['variable'])}</string>")
    if "path" in cfg:
        parts.append(f"<key>Path</key><string>{escape(cfg['path'])}</string>")
    return "".join(parts)


def _range(cfg: dict[str, Any]) -> str:
    start = cfg.get("start", "1")
    end = cfg.get("end", "10")
    upwards = cfg.get("upwards", True)
    return (
        f"<key>EndExpression</key><string>{escape(end)}</string>"
        f"<key>StartExpression</key><string>{escape(start)}</string>"
        f"<key>Upwards</key><{'true' if upwards else 'false'}/>"
    )


def _substrings_in(cfg: dict[str, Any]) -> str:
    result = cfg.get("result_type", "String")
    search = cfg.get("search", "")
    kind = cfg.get("search_kind", "Match")
    source = cfg.get("source", "Clipboard")
    parts = [
        f"<key>ResultType</key><string>{escape(result)}</string>",
        f"<key>Search</key><string>{escape(search)}</string>",
        f"<key>SearchKind</key><string>{escape(kind)}</string>",
        f"<key>Source</key><string>{escape(source)}</string>",
    ]
    if "variable" in cfg:
        parts.append(f"<key>Variable</key><string>{escape(cfg['variable'])}</string>")
    return "".join(parts)


def _volumes(cfg: dict[str, Any]) -> str:
    as_path = cfg.get("as_path", False)
    return f"<key>AsPath</key><{'true' if as_path else 'false'}/>"


_COLLECTION_EMITTERS: dict[str, Any] = {
    "Applications": _no_extra_keys,
    "Dictionaries": _no_extra_keys,
    "DictionaryKeys": _dictionary_keys,
    "Files": _files,
    "FinderSelection": _no_extra_keys,
    "FoundImages": _found_images,
    "JSON": _json_collection,
    "LinesIn": _lines_in,
    "PastClipboards": _no_extra_keys,
    "Range": _range,
    "SubstringsIn": _substrings_in,
    "Variables": _no_extra_keys,
    "Volumes": _volumes,
}


SUPPORTED_COLLECTION_TYPES: tuple[str, ...] = tuple(sorted(_COLLECTION_EMITTERS))
