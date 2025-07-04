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
from datetime import datetime, UTC

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
        
        # Time-based filter
        if query.max_days_since_used is not None:
            from datetime import timedelta, UTC
            cutoff_date = datetime.now(UTC) - timedelta(days=query.max_days_since_used)
            filtered_macros = [
                m for m in filtered_macros 
                if m.usage_stats.last_executed and m.usage_stats.last_executed >= cutoff_date
            ]
            applied_filters["max_days_since_used"] = query.max_days_since_used
        
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
            
            # Group-based categorization
            if macro.group:
                categories[f"group_{macro.group.lower().replace(' ', '_')}"].append(macro)
        
        return dict(categories)
    
    def generate_insights(
        self,
        macros: List[EnhancedMacroMetadata]
    ) -> Dict[str, Any]:
        """Generate insights about the macro library."""
        if not macros:
            return {"total_macros": 0, "insights": []}
        
        insights = []
        
        # Usage patterns
        total_executions = sum(m.usage_stats.total_executions for m in macros)
        unused_count = sum(1 for m in macros if m.usage_stats.total_executions == 0)
        
        if unused_count > len(macros) * 0.3:
            insights.append(f"{unused_count} macros ({unused_count/len(macros)*100:.1f}%) have never been used")
        
        # Complexity distribution
        complexity_counts = defaultdict(int)
        for macro in macros:
            complexity_counts[macro.complexity] += 1
        
        if complexity_counts[ComplexityLevel.ADVANCED] > len(macros) * 0.2:
            insights.append(f"High complexity detected: {complexity_counts[ComplexityLevel.ADVANCED]} advanced macros may need refactoring")
        
        # Function distribution
        function_counts = defaultdict(int)
        for macro in macros:
            function_counts[macro.primary_function] += 1
        
        top_function = max(function_counts.keys(), key=lambda k: function_counts[k])
        insights.append(f"Most common automation: {top_function.value} ({function_counts[top_function]} macros)")
        
        # Success rate analysis
        low_success_macros = [m for m in macros if m.usage_stats.success_rate < 0.8 and m.usage_stats.total_executions > 5]
        if low_success_macros:
            insights.append(f"{len(low_success_macros)} macros have low success rates and may need debugging")
        
        return {
            "total_macros": len(macros),
            "total_executions": total_executions,
            "unused_macros": unused_count,
            "complexity_distribution": dict(complexity_counts),
            "function_distribution": dict(function_counts),
            "insights": insights
        }
    
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
            match_found = False
            
            if scope == SearchScope.NAME_ONLY:
                if search_lower in macro.name.lower():
                    match_found = True
            elif scope == SearchScope.NAME_AND_GROUP:
                if search_lower in macro.name.lower() or search_lower in macro.group.lower():
                    match_found = True
            elif scope == SearchScope.FULL_CONTENT:
                # Search in name, group, action descriptions, trigger descriptions
                searchable_content = [
                    macro.name,
                    macro.group,
                    *[action.description for action in macro.actions],
                    *[trigger.description for trigger in macro.triggers]
                ]
                if any(search_lower in content.lower() for content in searchable_content if content):
                    match_found = True
            elif scope == SearchScope.METADATA_ONLY:
                # Search in optimization suggestions and other metadata
                searchable_metadata = [
                    *macro.optimization_suggestions,
                    macro.primary_function.value,
                    macro.complexity.value
                ]
                if any(search_lower in content.lower() for content in searchable_metadata):
                    match_found = True
            
            if match_found:
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
        """Calculate string similarity using a simple approach."""
        if not str1 or not str2:
            return 0.0
        
        # Convert to lowercase for comparison
        str1, str2 = str1.lower(), str2.lower()
        
        # Simple character-based similarity
        longer = str1 if len(str1) > len(str2) else str2
        shorter = str1 if len(str1) <= len(str2) else str2
        
        if len(longer) == 0:
            return 1.0
        
        # Count matching characters at same positions
        matches = sum(1 for i, c in enumerate(shorter) if i < len(longer) and longer[i] == c)
        
        # Add points for substrings
        if shorter in longer:
            matches += len(shorter) * 0.5
        
        return min(matches / len(longer), 1.0)
    
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
            
            if query.complexity_levels:
                suggestions.append("Try including more complexity levels")
        
        elif result_count > 50:
            suggestions.append("Many results found. Consider adding more specific filters.")
            
            if not query.action_categories:
                popular_categories = self._get_popular_categories(all_macros)
                suggestions.append(f"Filter by category: {', '.join(popular_categories[:3])}")
            
            if not query.groups:
                popular_groups = self._get_popular_groups(all_macros)
                suggestions.append(f"Filter by group: {', '.join(popular_groups[:3])}")
        
        # Usage-based suggestions
        if not query.min_usage_count and all_macros:
            avg_usage = sum(m.usage_stats.total_executions for m in all_macros) / len(all_macros)
            if avg_usage > 5:
                suggestions.append(f"Filter frequently used macros with min_usage_count > {int(avg_usage)}")
        
        return suggestions
    
    def _get_popular_categories(self, macros: List[EnhancedMacroMetadata]) -> List[str]:
        """Get most popular action categories in the macro library."""
        category_counts = defaultdict(int)
        for macro in macros:
            category_counts[macro.primary_function.value] += 1
        
        return sorted(category_counts.keys(), key=lambda k: category_counts[k], reverse=True)
    
    def _get_popular_groups(self, macros: List[EnhancedMacroMetadata]) -> List[str]:
        """Get most popular groups in the macro library."""
        group_counts = defaultdict(int)
        for macro in macros:
            if macro.group:
                group_counts[macro.group] += 1
        
        return sorted(group_counts.keys(), key=lambda k: group_counts[k], reverse=True)
    
    def _build_search_patterns(self) -> Dict[str, re.Pattern]:
        """Build compiled regex patterns for search optimization."""
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "url": re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?'),
            "file_path": re.compile(r'[/\\](?:[^/\\]+[/\\])*[^/\\]+'),
            "hotkey": re.compile(r'(⌘|⌥|⇧|⌃|Command|Option|Shift|Control)\s*\+?\s*\w'),
        }