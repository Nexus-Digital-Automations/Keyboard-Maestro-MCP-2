"""Critical-path tests for action catalog search (data integrity for discovery).

If search drops a built-in or reports phantom hits, agents pick the wrong
action and km_action_builder either rejects them or — worse — silently
appends a no-op. These tests pin the contract: full coverage of the
built-in catalog, predictable ranking on canonical queries, working
filters.
"""

from __future__ import annotations

import pytest
from src.server.tools._action_search import build_index, search
from src.server.tools._action_templates import load_catalog, load_templates


@pytest.fixture(autouse=True)
def reset_caches() -> None:
    # Test ordering can clobber cached state from other suites (e.g. the
    # plug-in metadata tests change KM_PLUGIN_ACTIONS_DIR). Clear before
    # every test so each starts from the real catalog + real plug-in scan.
    build_index.cache_clear()
    load_catalog.cache_clear()
    load_templates.cache_clear()


class TestIndex:
    def test_index_contains_every_catalog_identifier(self) -> None:
        idx = build_index()
        catalog_ids = {entry["identifier"] for entry in load_catalog()}
        index_ids = {entry.identifier for entry in idx}
        assert catalog_ids.issubset(index_ids), (
            f"missing from index: {catalog_ids - index_ids}"
        )

    def test_index_marks_catalog_entries_with_their_macro_action_type(self) -> None:
        idx = {e.identifier: e for e in build_index()}
        assert idx["speak_text"].macro_action_type == "SpeakText"
        assert idx["pause"].macro_action_type == "Pause"


class TestEmptyQuery:
    def test_empty_query_returns_filtered_catalog(self) -> None:
        results = search("", limit=100)
        assert len(results) >= len(load_catalog())

    def test_empty_query_with_category_filter(self) -> None:
        results = search("", category="control", limit=100)
        assert all(r["category"] == "control" for r in results)
        assert {r["identifier"] for r in results} >= {"pause", "execute_macro"}

    def test_empty_query_with_builder_supported_true(self) -> None:
        results = search("", builder_supported=True, limit=100)
        assert all(r["builder_supported"] for r in results)


class TestRanking:
    @pytest.mark.parametrize(("query", "expected_top"), [
        ("speak text aloud", "speak_text"),
        ("set a variable", "set_variable"),
        ("pause for 2 seconds", "pause"),
        ("run an applescript", "run_applescript"),
    ])
    def test_canonical_query_picks_expected_action(
        self, query: str, expected_top: str,
    ) -> None:
        results = search(query, limit=3)
        assert results, f"no results for {query!r}"
        assert results[0]["identifier"] == expected_top, (
            f"top for {query!r} was {results[0]['identifier']!r}, "
            f"expected {expected_top!r}"
        )

    def test_results_sorted_by_descending_score(self) -> None:
        results = search("variable", limit=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_matched_fields_populated_for_strong_matches(self) -> None:
        [top] = search("set_variable", limit=1)
        assert "identifier" in top["matched_fields"]


class TestFiltering:
    def test_category_filter_excludes_other_categories(self) -> None:
        results = search("text", category="text", limit=50)
        assert all(r["category"] == "text" for r in results)

    def test_min_score_drops_low_relevance(self) -> None:
        loose = search("xyzzy", limit=50)
        strict = search("xyzzy", min_score=0.9, limit=50)
        assert len(strict) <= len(loose)
        assert all(r["score"] >= 0.9 for r in strict)

    def test_limit_caps_returned_count(self) -> None:
        results = search("", limit=3)
        assert len(results) == 3


class TestResponseShape:
    def test_response_carries_required_keys(self) -> None:
        [first] = search("pause", limit=1)
        required = {
            "identifier", "category", "title", "description", "keywords",
            "builder_supported", "result_targets", "score", "matched_fields",
            "macro_action_type",
        }
        assert required.issubset(first.keys()), required - first.keys()
