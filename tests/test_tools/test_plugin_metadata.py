"""Critical-path tests for plug-in metadata extraction (data integrity for catalog input).

Missing or mis-parsed KM plug-in plist fields silently degrade the action
catalog: search misses plug-ins keyed only by their `KeyWords`, and
agents lose discoverability for installed plug-ins. These tests pin the
parse contract against synthetic plug-in bundles.
"""

import plistlib
from pathlib import Path

import pytest
from src.server.tools.action_tools import _plugin_entry
from src.server.tools.plugin_action_tools import _scan_installed_plugins


def _write_bundle(root: Path, name: str, spec: dict) -> None:
    bundle = root / name
    bundle.mkdir(parents=True, exist_ok=True)
    with (bundle / "Keyboard Maestro Action.plist").open("wb") as fp:
        plistlib.dump(spec, fp)


@pytest.fixture
def isolated_plugin_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KM_PLUGIN_ACTIONS_DIR", str(tmp_path))
    return tmp_path


class TestKeywordsExtraction:
    def test_keywords_camelcase_extracted(self, isolated_plugin_root: Path) -> None:
        _write_bundle(isolated_plugin_root, "Sample", {
            "Name": "Sample Action",
            "Title": "Do %Param%Thing%",
            "KeyWords": ["click", "button", "ui"],
        })
        [plugin] = _scan_installed_plugins()
        assert plugin["keywords"] == ["click", "button", "ui"]

    def test_keywords_alt_casing_also_accepted(self, isolated_plugin_root: Path) -> None:
        _write_bundle(isolated_plugin_root, "AltCase", {
            "Name": "Alt Case",
            "Title": "Title",
            "Keywords": ["alpha", "beta"],
        })
        [plugin] = _scan_installed_plugins()
        assert plugin["keywords"] == ["alpha", "beta"]

    def test_missing_keywords_yields_empty_list(self, isolated_plugin_root: Path) -> None:
        _write_bundle(isolated_plugin_root, "NoKw", {
            "Name": "No Keywords",
            "Title": "T",
        })
        [plugin] = _scan_installed_plugins()
        assert plugin["keywords"] == []

    def test_non_string_keywords_filtered_out(self, isolated_plugin_root: Path) -> None:
        _write_bundle(isolated_plugin_root, "Mixed", {
            "Name": "Mixed",
            "Title": "T",
            "KeyWords": ["good", 42, "", "also-good"],
        })
        [plugin] = _scan_installed_plugins()
        assert plugin["keywords"] == ["good", "also-good"]


class TestPluginEntryShape:
    def test_entry_surfaces_keywords_author_helpurl(self, isolated_plugin_root: Path) -> None:
        _write_bundle(isolated_plugin_root, "Full", {
            "Name": "Full",
            "Title": "T",
            "KeyWords": ["k1"],
            "Author": "Alice",
            "HelpURL": "https://example.com/help",
        })
        [scanned] = _scan_installed_plugins()
        entry = _plugin_entry(scanned)
        assert entry["keywords"] == ["k1"]
        assert entry["plugin_metadata"]["author"] == "Alice"
        assert entry["plugin_metadata"]["help_url"] == "https://example.com/help"

    def test_entry_keywords_default_to_empty_when_missing(
        self, isolated_plugin_root: Path,
    ) -> None:
        _write_bundle(isolated_plugin_root, "Bare", {"Name": "Bare", "Title": "T"})
        [scanned] = _scan_installed_plugins()
        entry = _plugin_entry(scanned)
        assert entry["keywords"] == []
        assert entry["plugin_metadata"]["author"] is None
        assert entry["plugin_metadata"]["help_url"] is None
