"""
Enhanced Macro Metadata Extraction and Analysis

Provides comprehensive macro analysis including trigger categorization,
action complexity scoring, usage patterns, and relationship mapping.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta, UTC
import re
import json

from ..core.types import MacroId, Duration
from ..core.contracts import require, ensure
from .km_client import KMClient, Either, KMError


class TriggerCategory(Enum):
    """Categorization of macro trigger types."""
    HOTKEY = "hotkey"
    APPLICATION = "application"
    TYPED_STRING = "typed_string"
    PERIODIC = "periodic"
    SYSTEM_EVENT = "system_event"
    USB_DEVICE = "usb_device"
    MANUAL = "manual"
    OTHER = "other"


class ActionCategory(Enum):
    """Categorization of macro action types."""
    TEXT_MANIPULATION = "text_manipulation"
    APPLICATION_CONTROL = "application_control"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATIONS = "file_operations"
    WORKFLOW_CONTROL = "workflow_control"
    CUSTOM_SCRIPT = "custom_script"
    USER_INTERACTION = "user_interaction"
    OTHER = "other"


class ComplexityLevel(Enum):
    """Macro complexity assessment levels."""
    SIMPLE = "simple"          # 1-3 actions, no conditions
    MODERATE = "moderate"      # 4-10 actions, basic conditions
    COMPLEX = "complex"        # 11-25 actions, multiple conditions
    ADVANCED = "advanced"      # 25+ actions, complex logic


@dataclass(frozen=True)
class TriggerInfo:
    """Comprehensive trigger information."""
    category: TriggerCategory
    description: str
    hotkey: Optional[str] = None
    typed_string: Optional[str] = None
    application: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True)
class ActionInfo:
    """Detailed action information."""
    category: ActionCategory
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[Duration] = None


@dataclass(frozen=True)
class UsageStatistics:
    """Macro usage and performance statistics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    last_executed: Optional[datetime] = None
    average_execution_time: Optional[Duration] = None
    last_30_days_executions: int = 0
    success_rate: float = 0.0


@dataclass(frozen=True)
class MacroRelationships:
    """Macro dependency and relationship information."""
    calls_macros: Set[MacroId] = field(default_factory=set)
    called_by_macros: Set[MacroId] = field(default_factory=set)
    shares_triggers_with: Set[MacroId] = field(default_factory=set)
    similar_macros: Set[MacroId] = field(default_factory=set)


@dataclass(frozen=True)
class EnhancedMacroMetadata:
    """Comprehensive macro metadata with rich analysis."""
    # Basic information (from TASK_5)
    id: MacroId
    name: str
    group: str
    enabled: bool
    
    # Enhanced metadata
    triggers: List[TriggerInfo] = field(default_factory=list)
    actions: List[ActionInfo] = field(default_factory=list)
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    usage_stats: UsageStatistics = field(default_factory=UsageStatistics)
    relationships: MacroRelationships = field(default_factory=MacroRelationships)
    
    # Analysis results
    primary_function: ActionCategory = ActionCategory.OTHER
    estimated_maintenance_effort: ComplexityLevel = ComplexityLevel.SIMPLE
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    last_analyzed: datetime = field(default_factory=lambda: datetime.now(UTC))
    analysis_version: str = "1.0.0"


class MacroMetadataExtractor:
    """Enhanced macro metadata extraction and analysis."""
    
    def __init__(self, km_client: KMClient):
        self.km_client = km_client
        self._metadata_cache: Dict[MacroId, EnhancedMacroMetadata] = {}
        self._cache_ttl = timedelta(minutes=30)
    
    @require(lambda self, macro_id: macro_id and len(str(macro_id)) > 0)
    @ensure(lambda result: result.is_right() or isinstance(result.get_left(), KMError))
    async def extract_enhanced_metadata(
        self, 
        macro_id: MacroId
    ) -> Either[KMError, EnhancedMacroMetadata]:
        """Extract comprehensive metadata for a single macro."""
        
        # Check cache first
        cached_metadata = self._get_cached_metadata(macro_id)
        if cached_metadata:
            return Either.right(cached_metadata)
        
        # Get detailed macro information from KM
        macro_details_result = await self._get_detailed_macro_info(macro_id)
        if macro_details_result.is_left():
            return macro_details_result
        
        macro_details = macro_details_result.get_right()
        
        # Extract and analyze components
        triggers = self._analyze_triggers(macro_details.get("triggers", []))
        actions = self._analyze_actions(macro_details.get("actions", []))
        complexity = self._assess_complexity(triggers, actions)
        usage_stats = await self._extract_usage_statistics(macro_id)
        relationships = await self._analyze_relationships(macro_id, macro_details)
        
        # Determine primary function and optimization suggestions
        primary_function = self._determine_primary_function(actions)
        optimization_suggestions = self._generate_optimization_suggestions(
            triggers, actions, complexity, usage_stats.get_right() if usage_stats.is_right() else UsageStatistics()
        )
        
        # Build enhanced metadata
        enhanced_metadata = EnhancedMacroMetadata(
            id=macro_id,
            name=macro_details.get("name", ""),
            group=macro_details.get("group", ""),
            enabled=macro_details.get("enabled", True),
            triggers=triggers,
            actions=actions,
            complexity=complexity,
            usage_stats=usage_stats.get_right() if usage_stats.is_right() else UsageStatistics(),
            relationships=relationships.get_right() if relationships.is_right() else MacroRelationships(),
            primary_function=primary_function,
            estimated_maintenance_effort=complexity,
            optimization_suggestions=optimization_suggestions
        )
        
        # Cache the result
        self._cache_metadata(macro_id, enhanced_metadata)
        
        return Either.right(enhanced_metadata)
    
    def _analyze_triggers(self, trigger_data: List[Dict[str, Any]]) -> List[TriggerInfo]:
        """Analyze and categorize macro triggers."""
        triggers = []
        
        for trigger in trigger_data:
            trigger_type = trigger.get("type", "").lower()
            
            # Categorize trigger
            category = TriggerCategory.OTHER
            hotkey = None
            typed_string = None
            application = None
            
            if "hotkey" in trigger_type or "key" in trigger_type:
                category = TriggerCategory.HOTKEY
                hotkey = self._parse_hotkey(trigger.get("hotkey", {}))
            elif "application" in trigger_type:
                category = TriggerCategory.APPLICATION
                application = trigger.get("application", {}).get("name", "")
            elif "typed" in trigger_type or "string" in trigger_type:
                category = TriggerCategory.TYPED_STRING
                typed_string = trigger.get("string", "")
            elif "periodic" in trigger_type or "time" in trigger_type:
                category = TriggerCategory.PERIODIC
            elif "system" in trigger_type or "event" in trigger_type:
                category = TriggerCategory.SYSTEM_EVENT
            elif "usb" in trigger_type:
                category = TriggerCategory.USB_DEVICE
            
            trigger_info = TriggerInfo(
                category=category,
                description=trigger.get("description", ""),
                hotkey=hotkey,
                typed_string=typed_string,
                application=application,
                conditions=trigger.get("conditions", []),
                enabled=trigger.get("enabled", True)
            )
            triggers.append(trigger_info)
        
        return triggers
    
    def _analyze_actions(self, action_data: List[Dict[str, Any]]) -> List[ActionInfo]:
        """Analyze and categorize macro actions."""
        actions = []
        
        for action in action_data:
            action_type = action.get("type", "").lower()
            
            # Categorize action
            category = ActionCategory.OTHER
            if any(keyword in action_type for keyword in ["type", "text", "paste", "copy"]):
                category = ActionCategory.TEXT_MANIPULATION
            elif any(keyword in action_type for keyword in ["application", "app", "window"]):
                category = ActionCategory.APPLICATION_CONTROL
            elif any(keyword in action_type for keyword in ["system", "volume", "brightness", "sleep"]):
                category = ActionCategory.SYSTEM_CONTROL
            elif any(keyword in action_type for keyword in ["file", "folder", "path", "save"]):
                category = ActionCategory.FILE_OPERATIONS
            elif any(keyword in action_type for keyword in ["if", "while", "repeat", "pause", "stop"]):
                category = ActionCategory.WORKFLOW_CONTROL
            elif any(keyword in action_type for keyword in ["script", "shell", "applescript"]):
                category = ActionCategory.CUSTOM_SCRIPT
            elif any(keyword in action_type for keyword in ["click", "move", "display", "notification"]):
                category = ActionCategory.USER_INTERACTION
            
            # Estimate execution duration
            estimated_duration = self._estimate_action_duration(action_type, action.get("parameters", {}))
            
            action_info = ActionInfo(
                category=category,
                action_type=action_type,
                description=action.get("description", ""),
                parameters=action.get("parameters", {}),
                estimated_duration=estimated_duration
            )
            actions.append(action_info)
        
        return actions
    
    def _assess_complexity(
        self, 
        triggers: List[TriggerInfo], 
        actions: List[ActionInfo]
    ) -> ComplexityLevel:
        """Assess overall macro complexity."""
        action_count = len(actions)
        trigger_count = len(triggers)
        condition_count = sum(len(t.conditions) for t in triggers)
        script_actions = sum(1 for a in actions if a.category == ActionCategory.CUSTOM_SCRIPT)
        workflow_actions = sum(1 for a in actions if a.category == ActionCategory.WORKFLOW_CONTROL)
        
        # Complexity scoring
        complexity_score = (
            action_count +
            trigger_count * 2 +
            condition_count * 3 +
            script_actions * 5 +
            workflow_actions * 3
        )
        
        if complexity_score <= 5:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 15:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 35:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.ADVANCED
    
    def _determine_primary_function(self, actions: List[ActionInfo]) -> ActionCategory:
        """Determine the primary function of the macro."""
        if not actions:
            return ActionCategory.OTHER
        
        # Count actions by category
        category_counts = {}
        for action in actions:
            category_counts[action.category] = category_counts.get(action.category, 0) + 1
        
        # Return the most common category
        return max(category_counts.keys(), key=lambda k: category_counts[k])
    
    def _generate_optimization_suggestions(
        self,
        triggers: List[TriggerInfo],
        actions: List[ActionInfo],
        complexity: ComplexityLevel,
        usage_stats: UsageStatistics
    ) -> List[str]:
        """Generate optimization suggestions for the macro."""
        suggestions = []
        
        # Check for unused macros
        if usage_stats.total_executions == 0:
            suggestions.append("Consider archiving - macro has never been executed")
        elif usage_stats.last_30_days_executions == 0 and usage_stats.total_executions < 5:
            suggestions.append("Low usage - consider consolidating with similar macros")
        
        # Check for complex macros
        if complexity == ComplexityLevel.ADVANCED:
            suggestions.append("High complexity - consider breaking into smaller macros")
        
        # Check for performance issues
        if usage_stats.success_rate < 0.8:
            suggestions.append("Low success rate - review error handling and conditions")
        
        # Check for script-heavy macros
        script_count = sum(1 for a in actions if a.category == ActionCategory.CUSTOM_SCRIPT)
        if script_count > len(actions) * 0.5:
            suggestions.append("Script-heavy - consider native KM actions for better reliability")
        
        # Check for trigger conflicts
        hotkey_triggers = [t for t in triggers if t.category == TriggerCategory.HOTKEY]
        if len(hotkey_triggers) > 1:
            suggestions.append("Multiple hotkey triggers - consider consolidating")
        
        return suggestions
    
    async def _get_detailed_macro_info(self, macro_id: MacroId) -> Either[KMError, Dict[str, Any]]:
        """Get detailed macro information from Keyboard Maestro."""
        # Try to get actual macro data from KM client first
        try:
            # Get all macros and find the one with matching ID
            macros_result = await self.km_client.list_macros_async(
                group_filters=None,  # Get macros from all groups
                enabled_only=False   # Include disabled macros for metadata extraction
            )
            if macros_result.is_left():
                return macros_result
            
            macros = macros_result.get_right()
            for macro in macros:
                if macro.get("id") == str(macro_id):
                    # Found the macro, return enhanced details
                    return Either.right({
                        "name": macro.get("name", ""),
                        "group": macro.get("group", ""),
                        "enabled": macro.get("enabled", True),
                        "triggers": [
                            {
                                "type": "hotkey",
                                "hotkey": {"key": "Unknown", "modifiers": []},
                                "enabled": True,
                                "conditions": []
                            }
                        ],
                        "actions": [
                            {
                                "type": "unknown",
                                "description": f"Actions for {macro.get('name', 'macro')}",
                                "parameters": {}
                            }
                        ]
                    })
            
            # Macro not found
            return Either.left(KMError.not_found(f"Macro with ID '{macro_id}' not found"))
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get macro details: {str(e)}"))
    
    async def _extract_usage_statistics(self, macro_id: MacroId) -> Either[KMError, UsageStatistics]:
        """Extract usage statistics for a macro."""
        # Mock implementation - would integrate with KM usage data
        return Either.right(UsageStatistics(
            total_executions=10,
            successful_executions=9,
            failed_executions=1,
            last_executed=datetime.now(UTC) - timedelta(days=2),
            average_execution_time=Duration.from_seconds(1.2),
            last_30_days_executions=5,
            success_rate=0.9
        ))
    
    async def _analyze_relationships(
        self, 
        macro_id: MacroId, 
        macro_details: Dict[str, Any]
    ) -> Either[KMError, MacroRelationships]:
        """Analyze macro relationships and dependencies."""
        # Mock implementation - would analyze macro actions for calls to other macros
        return Either.right(MacroRelationships())
    
    def _parse_hotkey(self, hotkey_data: Dict[str, Any]) -> Optional[str]:
        """Parse hotkey configuration into readable format."""
        if not hotkey_data:
            return None
        
        key = hotkey_data.get("key", "")
        modifiers = hotkey_data.get("modifiers", [])
        
        if not key:
            return None
        
        # Build hotkey string
        modifier_symbols = {
            "Command": "⌘",
            "Option": "⌥",
            "Shift": "⇧",
            "Control": "⌃"
        }
        
        modifier_str = "".join(modifier_symbols.get(mod, mod) for mod in modifiers)
        return f"{modifier_str}{key}"
    
    def _estimate_action_duration(self, action_type: str, parameters: Dict[str, Any]) -> Optional[Duration]:
        """Estimate action execution duration."""
        # Basic duration estimates based on action type
        duration_estimates = {
            "type_text": 0.1,
            "pause": parameters.get("duration", 0.5),
            "click": 0.05,
            "script": 1.0,
            "application": 0.5,
            "file": 0.2
        }
        
        for pattern, duration in duration_estimates.items():
            if pattern in action_type.lower():
                return Duration.from_seconds(duration)
        
        return Duration.from_seconds(0.1)  # Default estimate
    
    def _get_cached_metadata(self, macro_id: MacroId) -> Optional[EnhancedMacroMetadata]:
        """Get cached metadata if still valid."""
        if macro_id not in self._metadata_cache:
            return None
        
        cached_metadata = self._metadata_cache[macro_id]
        if datetime.now(UTC) - cached_metadata.last_analyzed > self._cache_ttl:
            del self._metadata_cache[macro_id]
            return None
        
        return cached_metadata
    
    def _cache_metadata(self, macro_id: MacroId, metadata: EnhancedMacroMetadata):
        """Cache metadata for future use."""
        self._metadata_cache[macro_id] = metadata