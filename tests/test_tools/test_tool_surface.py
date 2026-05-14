"""Assertion tests for the MCP tool surface.

Pins the public tool name list and the canonical / deprecated split so a
future merge or rename can't silently drop or rename a tool the agents
depend on. Each Phase 7 deprecation alias is listed here with the
canonical target it'll fold into; the assertion list is the source of
truth for which names callers can rely on right now.
"""

from __future__ import annotations

from src.server.tool_registry import ToolDiscovery

# Canonical surface — tools that should continue to exist after the Phase 7
# structural merges land. New tools and renames go here.
_CANONICAL_TOOLS: frozenset[str] = frozenset({
    # Action authoring + discovery
    "km_action_builder",
    "km_search_actions",
    "km_create_plugin_action",
    "km_refresh_action_templates",
    "km_list_action_types",
    "km_add_condition",
    "km_control_flow",
    # Macros
    "km_create_macro",
    "km_macro_editor",
    "km_macro_group_manager",
    "km_list_macros",
    "km_list_templates",
    "km_execute_macro",
    # Triggers (km_trigger_crud is the XML-native advanced path)
    "km_trigger_crud",
    # Variables / tokens / engine
    "km_variable_manager",
    "km_token_processor",
    "km_engine_control",
    # Notifications (km_notifications is the canonical display surface)
    "km_notifications",
    # System
    "km_application_control",
    "km_window_manager",
})

# Deprecated aliases — kept registered for one release per the rollout plan;
# each emits a logger.warning on call pointing at the canonical replacement.
_DEPRECATED_ALIASES: dict[str, str] = {
    "km_build_plugin_action": "km_create_plugin_action",
    "km_notification_status": "km_notifications",
    "km_dismiss_notifications": "km_notifications",
    "km_token_stats": "km_token_processor",
    "km_move_macro_to_group": "km_macro_editor",
    "km_create_hotkey_trigger": "km_trigger_lifecycle (future)",
    "km_list_hotkey_triggers": "km_trigger_lifecycle (future)",
    "km_add_system_trigger": "km_trigger_lifecycle (future)",
    "km_trigger_manager": "km_trigger_lifecycle (future)",
}


def _discovered_names() -> set[str]:
    return set(ToolDiscovery().discover_all_tools().keys())


class TestCanonicalSurface:
    def test_every_canonical_tool_is_registered(self) -> None:
        discovered = _discovered_names()
        missing = _CANONICAL_TOOLS - discovered
        assert not missing, f"canonical tools missing from registry: {missing}"

    def test_every_deprecated_alias_is_still_registered(self) -> None:
        discovered = _discovered_names()
        missing = set(_DEPRECATED_ALIASES) - discovered
        assert not missing, (
            f"deprecated aliases dropped before the one-release window expired: "
            f"{missing}"
        )

    def test_no_tools_outside_canonical_or_deprecated_set(self) -> None:
        discovered = _discovered_names()
        expected = _CANONICAL_TOOLS | set(_DEPRECATED_ALIASES)
        unexpected = discovered - expected
        assert not unexpected, (
            f"new tools appeared without being added to _CANONICAL_TOOLS or "
            f"_DEPRECATED_ALIASES: {unexpected}"
        )


class TestSurfaceMetrics:
    def test_canonical_count_within_target_range(self) -> None:
        # Plan target is 20 canonical tools. The exact count drifts as
        # condition/control-flow merge in or split; assert the broad target
        # range rather than exact equality so curating Phase 7 commits
        # doesn't churn this test.
        assert 18 <= len(_CANONICAL_TOOLS) <= 22, len(_CANONICAL_TOOLS)
