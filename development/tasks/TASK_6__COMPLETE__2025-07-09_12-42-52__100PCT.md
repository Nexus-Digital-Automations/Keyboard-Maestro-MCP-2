# TASK_6: Enhanced Macro Discovery and Metadata

**Created By**: Agent_2 | **Priority**: MEDIUM | **Duration**: 2 hours
**Technique Focus**: Smart Filtering + Hierarchical Organization + Performance Optimization
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_1
**Dependencies**: TASK_5 (Real KM API integration)
**Blocking**: TASK_7 (Real-time synchronization)

## 📖 Required Reading (Complete before starting)
- [x] **TASK_5.md**: Real KM API integration implementation and data structures
- [x] **src/integration/km_client.py**: Enhanced macro listing methods from TASK_5
- [x] **src/server.py**: Updated km_list_macros tool with real data integration
- [x] **Keyboard Maestro Documentation**: Macro metadata, triggers, and usage statistics
- [x] **tests/TESTING.md**: Current integration test status for macro discovery

## 🎯 Implementation Overview
Extend basic macro listing with comprehensive metadata extraction, intelligent filtering, usage analytics, and hierarchical organization to provide AI clients with deep insights into user's automation patterns and macro complexity.

<thinking>
Building on TASK_5's real macro integration, this task adds:
1. Rich Metadata: Extract comprehensive macro information beyond basic listing
2. Smart Filtering: Advanced search, categorization, and pattern recognition  
3. Usage Analytics: Track and analyze macro usage patterns
4. Hierarchical Organization: Understand macro group relationships and dependencies
5. Performance Intelligence: Macro complexity analysis and optimization suggestions
6. Caching Strategy: Intelligent caching of expensive metadata operations
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Rich Metadata Extraction ✅ COMPLETED
- [x] **Trigger analysis**: Extract and categorize trigger types, conditions, hotkeys
- [x] **Action complexity**: Analyze action types, sequences, and dependencies
- [x] **Usage statistics**: Track execution frequency, last used, success rates
- [x] **Macro relationships**: Identify macro dependencies and call chains

### Phase 2: Smart Filtering and Search ✅ COMPLETED
- [x] **Advanced search**: Text search across names, comments, action content
- [x] **Category classification**: Auto-categorize by function (text, system, app control)
- [x] **Complexity scoring**: Rate macros by complexity and maintenance needs
- [x] **Pattern recognition**: Identify similar macros and suggest consolidation

### Phase 3: Performance and Organization ✅ COMPLETED
- [x] **Hierarchical grouping**: Multi-level group organization and inheritance
- [x] **Metadata caching**: Intelligent caching strategy for expensive operations
- [x] **Batch processing**: Efficient bulk metadata extraction
- [x] **Performance monitoring**: Track and optimize metadata extraction performance

## 🔧 Implementation Files & Specifications

### Core Enhancement Files to Create:

#### src/integration/macro_metadata.py - Rich Metadata Extraction
```python
"""
Enhanced Macro Metadata Extraction and Analysis

Provides comprehensive macro analysis including trigger categorization,
action complexity scoring, usage patterns, and relationship mapping.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
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
    last_analyzed: datetime = field(default_factory=datetime.utcnow)
    analysis_version: str = "1.0.0"


class MacroMetadataExtractor:
    """Enhanced macro metadata extraction and analysis."""
    
    def __init__(self, km_client: KMClient):
        self.km_client = km_client
        self._metadata_cache: Dict[MacroId, EnhancedMacroMetadata] = {}
        self._cache_ttl = timedelta(minutes=30)
    
    @require(lambda macro_id: macro_id is not None)
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
            else:
                category = TriggerCategory.OTHER
            
            trigger_info = TriggerInfo(
                category=category,
                description=trigger.get("description", ""),
                hotkey=locals().get("hotkey"),
                typed_string=locals().get("typed_string"),
                application=locals().get("application"),
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
            else:
                category = ActionCategory.OTHER
            
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
    
    def _get_cached_metadata(self, macro_id: MacroId) -> Optional[EnhancedMacroMetadata]:
        """Get cached metadata if still valid."""
        if macro_id not in self._metadata_cache:
            return None
        
        cached_metadata = self._metadata_cache[macro_id]
        if datetime.utcnow() - cached_metadata.last_analyzed > self._cache_ttl:
            del self._metadata_cache[macro_id]
            return None
        
        return cached_metadata
    
    def _cache_metadata(self, macro_id: MacroId, metadata: EnhancedMacroMetadata):
        """Cache metadata for future use."""
        self._metadata_cache[macro_id] = metadata
```

#### src/integration/smart_filtering.py - Advanced Search and Filtering
```python
"""
Smart Filtering and Search for Macro Discovery

Provides advanced search, categorization, pattern recognition,
and intelligent filtering capabilities for macro libraries.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Callable, Any
from enum import Enum
import re
from collections import defaultdict

from .macro_metadata import EnhancedMacroMetadata, ActionCategory, TriggerCategory, ComplexityLevel


class SearchScope(Enum):
    """Scope of macro search operations."""
    NAME_ONLY = "name_only"
    NAME_AND_GROUP = "name_and_group"
    FULL_CONTENT = "full_content"
    METADATA_ONLY = "metadata_only"


class SortCriteria(Enum):
    """Available sorting criteria for macro results."""
    NAME = "name"
    LAST_USED = "last_used"
    USAGE_FREQUENCY = "usage_frequency"
    COMPLEXITY = "complexity"
    GROUP = "group"
    SUCCESS_RATE = "success_rate"
    PRIMARY_FUNCTION = "primary_function"


@dataclass(frozen=True)
class SearchQuery:
    """Comprehensive search query specification."""
    text: Optional[str] = None
    scope: SearchScope = SearchScope.NAME_AND_GROUP
    action_categories: Set[ActionCategory] = None
    trigger_categories: Set[TriggerCategory] = None
    complexity_levels: Set[ComplexityLevel] = None
    groups: Set[str] = None
    enabled_only: bool = True
    min_usage_count: int = 0
    max_days_since_used: Optional[int] = None
    min_success_rate: float = 0.0


@dataclass(frozen=True)
class FilterResult:
    """Result of filtering operation with metadata."""
    macros: List[EnhancedMacroMetadata]
    total_matches: int
    applied_filters: Dict[str, Any]
    search_time_ms: float
    suggestions: List[str]


class SmartMacroFilter:
    """Advanced filtering and search for macro discovery."""
    
    def __init__(self):
        self._search_patterns = self._build_search_patterns()
        self._similarity_threshold = 0.7
    
    def search_macros(
        self,
        macros: List[EnhancedMacroMetadata],
        query: SearchQuery,
        sort_by: SortCriteria = SortCriteria.NAME,
        limit: Optional[int] = None
    ) -> FilterResult:
        """Perform advanced search and filtering on macro collection."""
        import time
        start_time = time.perf_counter()
        
        # Apply filters sequentially
        filtered_macros = macros.copy()
        applied_filters = {}
        
        # Text search
        if query.text:
            filtered_macros = self._apply_text_search(filtered_macros, query.text, query.scope)
            applied_filters["text_search"] = query.text
        
        # Category filters
        if query.action_categories:
            filtered_macros = [m for m in filtered_macros if m.primary_function in query.action_categories]
            applied_filters["action_categories"] = list(query.action_categories)
        
        if query.trigger_categories:
            filtered_macros = [
                m for m in filtered_macros 
                if any(t.category in query.trigger_categories for t in m.triggers)
            ]
            applied_filters["trigger_categories"] = list(query.trigger_categories)
        
        # Complexity filter
        if query.complexity_levels:
            filtered_macros = [m for m in filtered_macros if m.complexity in query.complexity_levels]
            applied_filters["complexity_levels"] = list(query.complexity_levels)
        
        # Group filter
        if query.groups:
            filtered_macros = [m for m in filtered_macros if m.group in query.groups]
            applied_filters["groups"] = list(query.groups)
        
        # Enabled filter
        if query.enabled_only:
            filtered_macros = [m for m in filtered_macros if m.enabled]
            applied_filters["enabled_only"] = True
        
        # Usage filters
        if query.min_usage_count > 0:
            filtered_macros = [
                m for m in filtered_macros 
                if m.usage_stats.total_executions >= query.min_usage_count
            ]
            applied_filters["min_usage_count"] = query.min_usage_count
        
        if query.min_success_rate > 0.0:
            filtered_macros = [
                m for m in filtered_macros 
                if m.usage_stats.success_rate >= query.min_success_rate
            ]
            applied_filters["min_success_rate"] = query.min_success_rate
        
        # Sort results
        sorted_macros = self._sort_macros(filtered_macros, sort_by)
        
        # Apply limit
        if limit:
            sorted_macros = sorted_macros[:limit]
        
        # Generate suggestions
        suggestions = self._generate_search_suggestions(macros, query, len(filtered_macros))
        
        search_time = (time.perf_counter() - start_time) * 1000
        
        return FilterResult(
            macros=sorted_macros,
            total_matches=len(filtered_macros),
            applied_filters=applied_filters,
            search_time_ms=search_time,
            suggestions=suggestions
        )
    
    def find_similar_macros(
        self,
        target_macro: EnhancedMacroMetadata,
        macro_library: List[EnhancedMacroMetadata],
        similarity_threshold: float = None
    ) -> List[EnhancedMacroMetadata]:
        """Find macros similar to the target macro."""
        threshold = similarity_threshold or self._similarity_threshold
        similar_macros = []
        
        for macro in macro_library:
            if macro.id == target_macro.id:
                continue
            
            similarity_score = self._calculate_similarity(target_macro, macro)
            if similarity_score >= threshold:
                similar_macros.append(macro)
        
        # Sort by similarity (highest first)
        similar_macros.sort(
            key=lambda m: self._calculate_similarity(target_macro, m),
            reverse=True
        )
        
        return similar_macros
    
    def categorize_macro_library(
        self,
        macros: List[EnhancedMacroMetadata]
    ) -> Dict[str, List[EnhancedMacroMetadata]]:
        """Automatically categorize macro library by function and pattern."""
        categories = defaultdict(list)
        
        for macro in macros:
            # Primary categorization by function
            primary_category = macro.primary_function.value
            categories[f"function_{primary_category}"].append(macro)
            
            # Secondary categorization by complexity
            complexity_category = f"complexity_{macro.complexity.value}"
            categories[complexity_category].append(macro)
            
            # Tertiary categorization by usage pattern
            if macro.usage_stats.total_executions == 0:
                categories["usage_never_used"].append(macro)
            elif macro.usage_stats.last_30_days_executions > 10:
                categories["usage_frequently_used"].append(macro)
            elif macro.usage_stats.success_rate < 0.5:
                categories["usage_problematic"].append(macro)
            
            # Quaternary categorization by optimization needs
            if macro.optimization_suggestions:
                categories["needs_optimization"].append(macro)
        
        return dict(categories)
    
    def _apply_text_search(
        self,
        macros: List[EnhancedMacroMetadata],
        search_text: str,
        scope: SearchScope
    ) -> List[EnhancedMacroMetadata]:
        """Apply text search based on scope."""
        search_lower = search_text.lower()
        results = []
        
        for macro in macros:
            if scope == SearchScope.NAME_ONLY:
                if search_lower in macro.name.lower():
                    results.append(macro)
            elif scope == SearchScope.NAME_AND_GROUP:
                if search_lower in macro.name.lower() or search_lower in macro.group.lower():
                    results.append(macro)
            elif scope == SearchScope.FULL_CONTENT:
                # Search in name, group, action descriptions, trigger descriptions
                searchable_content = [
                    macro.name,
                    macro.group,
                    *[action.description for action in macro.actions],
                    *[trigger.description for trigger in macro.triggers]
                ]
                if any(search_lower in content.lower() for content in searchable_content if content):
                    results.append(macro)
            elif scope == SearchScope.METADATA_ONLY:
                # Search in optimization suggestions and other metadata
                searchable_metadata = [
                    *macro.optimization_suggestions,
                    macro.primary_function.value,
                    macro.complexity.value
                ]
                if any(search_lower in content.lower() for content in searchable_metadata):
                    results.append(macro)
        
        return results
    
    def _sort_macros(
        self,
        macros: List[EnhancedMacroMetadata],
        sort_by: SortCriteria
    ) -> List[EnhancedMacroMetadata]:
        """Sort macros by specified criteria."""
        if sort_by == SortCriteria.NAME:
            return sorted(macros, key=lambda m: m.name.lower())
        elif sort_by == SortCriteria.LAST_USED:
            return sorted(
                macros,
                key=lambda m: m.usage_stats.last_executed or datetime.min,
                reverse=True
            )
        elif sort_by == SortCriteria.USAGE_FREQUENCY:
            return sorted(macros, key=lambda m: m.usage_stats.total_executions, reverse=True)
        elif sort_by == SortCriteria.COMPLEXITY:
            complexity_order = {
                ComplexityLevel.SIMPLE: 1,
                ComplexityLevel.MODERATE: 2,
                ComplexityLevel.COMPLEX: 3,
                ComplexityLevel.ADVANCED: 4
            }
            return sorted(macros, key=lambda m: complexity_order[m.complexity])
        elif sort_by == SortCriteria.GROUP:
            return sorted(macros, key=lambda m: (m.group.lower(), m.name.lower()))
        elif sort_by == SortCriteria.SUCCESS_RATE:
            return sorted(macros, key=lambda m: m.usage_stats.success_rate, reverse=True)
        elif sort_by == SortCriteria.PRIMARY_FUNCTION:
            return sorted(macros, key=lambda m: (m.primary_function.value, m.name.lower()))
        else:
            return macros
    
    def _calculate_similarity(
        self,
        macro1: EnhancedMacroMetadata,
        macro2: EnhancedMacroMetadata
    ) -> float:
        """Calculate similarity score between two macros."""
        similarity_factors = []
        
        # Name similarity (fuzzy matching)
        name_similarity = self._calculate_string_similarity(macro1.name, macro2.name)
        similarity_factors.append(name_similarity * 0.3)
        
        # Function similarity
        if macro1.primary_function == macro2.primary_function:
            similarity_factors.append(0.4)
        
        # Trigger similarity
        trigger_overlap = len(set(t.category for t in macro1.triggers) & 
                              set(t.category for t in macro2.triggers))
        if macro1.triggers and macro2.triggers:
            trigger_similarity = trigger_overlap / max(len(macro1.triggers), len(macro2.triggers))
            similarity_factors.append(trigger_similarity * 0.2)
        
        # Complexity similarity
        complexity_distance = abs(
            list(ComplexityLevel).index(macro1.complexity) - 
            list(ComplexityLevel).index(macro2.complexity)
        )
        complexity_similarity = 1.0 - (complexity_distance / len(ComplexityLevel))
        similarity_factors.append(complexity_similarity * 0.1)
        
        return sum(similarity_factors)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        if not str1 or not str2:
            return 0.0
        
        # Simple implementation - could be enhanced with proper Levenshtein
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str1 if len(str1) <= len(str2) else str2
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters (simple approximation)
        matches = sum(1 for c in shorter if c in longer)
        return matches / len(longer)
    
    def _generate_search_suggestions(
        self,
        all_macros: List[EnhancedMacroMetadata],
        query: SearchQuery,
        result_count: int
    ) -> List[str]:
        """Generate helpful search suggestions."""
        suggestions = []
        
        if result_count == 0:
            suggestions.append("No results found. Try broadening your search criteria.")
            
            if query.text:
                suggestions.append(f"Try searching for partial matches of '{query.text}'")
            
            if query.enabled_only:
                suggestions.append("Include disabled macros by setting enabled_only=False")
        
        elif result_count > 50:
            suggestions.append("Many results found. Consider adding more specific filters.")
            
            if not query.action_categories:
                popular_categories = self._get_popular_categories(all_macros)
                suggestions.append(f"Filter by category: {', '.join(popular_categories[:3])}")
        
        return suggestions
    
    def _get_popular_categories(self, macros: List[EnhancedMacroMetadata]) -> List[str]:
        """Get most popular action categories in the macro library."""
        category_counts = defaultdict(int)
        for macro in macros:
            category_counts[macro.primary_function.value] += 1
        
        return sorted(category_counts.keys(), key=lambda k: category_counts[k], reverse=True)
```

## 🏗️ Modularity Strategy
- **macro_metadata.py**: Enhanced metadata extraction and analysis (target: 250 lines)
- **smart_filtering.py**: Advanced search and filtering capabilities (target: 225 lines)
- **usage_analytics.py**: Usage pattern analysis and reporting (target: 150 lines)
- **hierarchy_manager.py**: Macro group organization and relationships (target: 125 lines)
- **performance_cache.py**: Intelligent caching for metadata operations (target: 100 lines)

## ✅ Success Criteria
- AI client gains deep insights into user's macro library organization and patterns
- Advanced search and filtering capabilities enable precise macro discovery
- Rich metadata provides actionable insights for macro optimization
- Usage analytics identify automation patterns and improvement opportunities
- Performance optimization ensures sub-second response times for complex queries
- Hierarchical organization reveals macro relationships and dependencies
- Smart caching minimizes expensive metadata extraction operations
- Pattern recognition suggests macro consolidation and optimization opportunities
- Category classification enables intelligent macro organization
- Similarity detection helps identify duplicate or overlapping functionality