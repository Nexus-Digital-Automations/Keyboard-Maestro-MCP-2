"""
Metadata and Analysis MCP Tools

Advanced macro discovery, metadata analysis, and search functionality.
"""

import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated

from ..core import MacroId
from ..integration.macro_metadata import (
    EnhancedMacroMetadata,
    ActionCategory,
    TriggerCategory,
    ComplexityLevel,
)
from ..integration.smart_filtering import (
    SearchQuery,
    SearchScope,
    SortCriteria,
)

logger = logging.getLogger(__name__)


def register_metadata_tools(mcp):
    """Register metadata and analysis tools with the MCP server."""
    
    @mcp.tool()
    async def km_search_macros_advanced(
        query: Annotated[Optional[str], Field(
            default=None,
            description="Search text for macro names, groups, or content"
        )] = None,
        scope: Annotated[str, Field(
            default="name_and_group",
            description="Search scope: name_only, name_and_group, full_content, metadata_only"
        )] = "name_and_group",
        action_categories: Annotated[Optional[List[str]], Field(
            default=None,
            description="List of action categories to filter by. Examples: ['text_manipulation', 'application_control', 'system_control']. Pass as an array, not a string."
        )] = None,
        complexity_levels: Annotated[Optional[List[str]], Field(
            default=None,
            description="List of complexity levels to filter by. Examples: ['simple', 'moderate', 'complex', 'advanced']. Pass as an array, not a string."
        )] = None,
        min_usage_count: Annotated[int, Field(
            default=0,
            ge=0,
            description="Minimum execution count filter"
        )] = 0,
        sort_by: Annotated[str, Field(
            default="name",
            description="Sort by: name, last_used, usage_frequency, complexity, success_rate"
        )] = "name",
        limit: Annotated[int, Field(
            default=20,
            ge=1,
            le=100,
            description="Maximum number of results"
        )] = 20,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Advanced macro search with comprehensive filtering and metadata analysis.
        
        TASK_6 IMPLEMENTATION: Enhanced macro discovery with rich metadata,
        smart filtering, usage analytics, and hierarchical organization.
        """
        if ctx:
            await ctx.info(f"Advanced search: '{query or 'all macros'}' with {scope} scope")
        
        try:
            # Import here to avoid circular dependencies
            from ..server_utils import get_km_client, get_metadata_extractor, smart_filter
            
            # Get KM client and metadata extractor
            km_client = get_km_client()
            extractor = get_metadata_extractor()
            
            if ctx:
                await ctx.report_progress(10, 100, "Connecting to Keyboard Maestro")
            
            # Get basic macro list first
            macros_result = await km_client.list_macros_async(
                group_filters=None,  # Get macros from all groups
                enabled_only=True    # Focus on enabled macros for search
            )
            
            if macros_result.is_left():
                error = macros_result.get_left()
                if ctx:
                    await ctx.error(f"Cannot connect to Keyboard Maestro: {error}")
                return {
                    "success": False,
                    "error": {
                        "code": "KM_CONNECTION_FAILED",
                        "message": "Cannot connect to Keyboard Maestro for advanced search",
                        "details": str(error),
                        "recovery_suggestion": "Ensure Keyboard Maestro is running and accessible"
                    }
                }
            
            basic_macros = macros_result.get_right()
            
            if ctx:
                await ctx.report_progress(30, 100, f"Extracting metadata for {len(basic_macros)} macros")
            
            # Extract enhanced metadata for each macro
            enhanced_macros: List[EnhancedMacroMetadata] = []
            for i, macro in enumerate(basic_macros):
                macro_id_str = macro.get("id", "")
                if macro_id_str:
                    # Convert string to MacroId branded type
                    macro_id = MacroId(macro_id_str)
                    metadata_result = await extractor.extract_enhanced_metadata(macro_id)
                    if metadata_result.is_right():
                        enhanced_macros.append(metadata_result.get_right())
                
                # Report progress
                if ctx and (i + 1) % 5 == 0:
                    progress = 30 + (i + 1) / len(basic_macros) * 40
                    await ctx.report_progress(int(progress), 100, f"Processed {i + 1}/{len(basic_macros)} macros")
            
            if ctx:
                await ctx.report_progress(70, 100, "Applying smart filters")
            
            # Convert filter parameters
            search_scope = getattr(SearchScope, scope.upper(), SearchScope.NAME_AND_GROUP)
            sort_criteria = getattr(SortCriteria, sort_by.upper(), SortCriteria.NAME)
            
            # Build search query
            action_cats = None
            if action_categories:
                action_cats = set()
                for cat in action_categories:
                    try:
                        action_cats.add(getattr(ActionCategory, cat.upper()))
                    except AttributeError:
                        pass
            
            complexity_levs = None
            if complexity_levels:
                complexity_levs = set()
                for level in complexity_levels:
                    try:
                        complexity_levs.add(getattr(ComplexityLevel, level.upper()))
                    except AttributeError:
                        pass
            
            search_query = SearchQuery(
                text=query,
                scope=search_scope,
                action_categories=action_cats,
                complexity_levels=complexity_levs,
                min_usage_count=min_usage_count,
                enabled_only=True
            )
            
            # Apply smart filtering
            filter_result = smart_filter.search_macros(
                enhanced_macros,
                search_query,
                sort_criteria,
                limit
            )
            
            if ctx:
                await ctx.report_progress(90, 100, "Generating insights")
            
            # Generate library insights
            insights = smart_filter.generate_insights(enhanced_macros)
            
            # Convert results to serializable format
            result_macros = []
            for macro in filter_result.macros:
                result_macros.append({
                    "id": str(macro.id),
                    "name": macro.name,
                    "group": macro.group,
                    "enabled": macro.enabled,
                    "complexity": macro.complexity.value,
                    "primary_function": macro.primary_function.value,
                    "trigger_count": len(macro.triggers),
                    "action_count": len(macro.actions),
                    "usage_stats": {
                        "total_executions": macro.usage_stats.total_executions,
                        "success_rate": macro.usage_stats.success_rate,
                        "last_executed": macro.usage_stats.last_executed.isoformat() if macro.usage_stats.last_executed else None,
                        "last_30_days": macro.usage_stats.last_30_days_executions
                    },
                    "optimization_suggestions": macro.optimization_suggestions,
                    "last_analyzed": macro.last_analyzed.isoformat()
                })
            
            if ctx:
                await ctx.report_progress(100, 100, "Search complete")
                await ctx.info(f"Found {len(result_macros)} matching macros with enhanced metadata")
            
            return {
                "success": True,
                "data": {
                    "macros": result_macros,
                    "total_matches": filter_result.total_matches,
                    "applied_filters": filter_result.applied_filters,
                    "search_time_ms": filter_result.search_time_ms,
                    "suggestions": filter_result.suggestions,
                    "library_insights": insights
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "server_version": "1.0.0",
                    "feature": "enhanced_macro_discovery_task_6",
                    "total_library_size": len(enhanced_macros),
                    "metadata_extraction_time": filter_result.search_time_ms
                }
            }
            
        except Exception as e:
            logger.exception("Error in km_search_macros_advanced")
            if ctx:
                await ctx.error(f"Advanced search error: {e}")
            return {
                "success": False,
                "error": {
                    "code": "SYSTEM_ERROR",
                    "message": "Advanced macro search failed",
                    "details": str(e),
                    "recovery_suggestion": "Check Keyboard Maestro connection and try again"
                }
            }

    @mcp.tool()
    async def km_analyze_macro_metadata(
        macro_id: Annotated[str, Field(
            description="Macro ID or name to analyze"
        )],
        include_relationships: Annotated[bool, Field(
            default=True,
            description="Include relationship analysis with other macros"
        )] = True,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Analyze detailed metadata for a specific macro including complexity,
        usage patterns, optimization suggestions, and relationships.
        
        TASK_6 IMPLEMENTATION: Deep macro analysis for AI-driven insights.
        """
        if ctx:
            await ctx.info(f"Analyzing metadata for macro: {macro_id}")
        
        try:
            # Import here to avoid circular dependencies
            from ..server_utils import get_metadata_extractor, get_km_client, smart_filter
            
            extractor = get_metadata_extractor()
            
            if ctx:
                await ctx.report_progress(25, 100, "Extracting detailed metadata")
            
            # Convert string to MacroId branded type and extract enhanced metadata
            macro_id_branded = MacroId(macro_id)
            metadata_result = await extractor.extract_enhanced_metadata(macro_id_branded)
            
            if metadata_result.is_left():
                error = metadata_result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": "MACRO_NOT_FOUND",
                        "message": f"Cannot analyze macro '{macro_id}'",
                        "details": str(error),
                        "recovery_suggestion": "Verify macro ID and ensure it exists in Keyboard Maestro"
                    }
                }
            
            macro_metadata = metadata_result.get_right()
            
            if ctx:
                await ctx.report_progress(75, 100, "Analyzing patterns and relationships")
            
            # Find similar macros if requested
            similar_macros = []
            if include_relationships:
                km_client = get_km_client()
                all_macros_result = await km_client.list_macros_async(
                    group_filters=None,  # Get macros from all groups for relationship analysis
                    enabled_only=False   # Include disabled macros for comprehensive analysis
                )
                
                if all_macros_result.is_right():
                    # Get enhanced metadata for comparison
                    all_enhanced = []
                    basic_macros = all_macros_result.get_right()
                    for basic_macro in basic_macros[:10]:  # Limit for performance
                        macro_id_str = basic_macro.get("id", "")
                        if macro_id_str:
                            # Convert string to MacroId branded type
                            macro_id = MacroId(macro_id_str)
                            enhanced_result = await extractor.extract_enhanced_metadata(macro_id)
                            if enhanced_result.is_right():
                                all_enhanced.append(enhanced_result.get_right())
                    
                    similar_macros_list = smart_filter.find_similar_macros(
                        macro_metadata, 
                        all_enhanced,
                        similarity_threshold=0.6
                    )
                    
                    for similar in similar_macros_list[:5]:  # Top 5 similar
                        similar_macros.append({
                            "id": str(similar.id),
                            "name": similar.name,
                            "group": similar.group,
                            "similarity_reason": "Similar function and complexity"
                        })
            
            # Build detailed analysis
            analysis_result = {
                "basic_info": {
                    "id": str(macro_metadata.id),
                    "name": macro_metadata.name,
                    "group": macro_metadata.group,
                    "enabled": macro_metadata.enabled
                },
                "complexity_analysis": {
                    "level": macro_metadata.complexity.value,
                    "trigger_count": len(macro_metadata.triggers),
                    "action_count": len(macro_metadata.actions),
                    "estimated_maintenance": macro_metadata.estimated_maintenance_effort.value
                },
                "function_analysis": {
                    "primary_function": macro_metadata.primary_function.value,
                    "trigger_categories": [t.category.value for t in macro_metadata.triggers],
                    "action_categories": list(set(a.category.value for a in macro_metadata.actions))
                },
                "usage_analytics": {
                    "total_executions": macro_metadata.usage_stats.total_executions,
                    "successful_executions": macro_metadata.usage_stats.successful_executions,
                    "failed_executions": macro_metadata.usage_stats.failed_executions,
                    "success_rate": macro_metadata.usage_stats.success_rate,
                    "last_executed": macro_metadata.usage_stats.last_executed.isoformat() if macro_metadata.usage_stats.last_executed else None,
                    "last_30_days_executions": macro_metadata.usage_stats.last_30_days_executions,
                    "average_execution_time": macro_metadata.usage_stats.average_execution_time.total_seconds() if macro_metadata.usage_stats.average_execution_time else None
                },
                "optimization": {
                    "suggestions": macro_metadata.optimization_suggestions,
                    "performance_score": max(0, min(10, int(macro_metadata.usage_stats.success_rate * 10))),
                    "complexity_score": ["simple", "moderate", "complex", "advanced"].index(macro_metadata.complexity.value) + 1
                },
                "triggers": [
                    {
                        "category": t.category.value,
                        "description": t.description,
                        "hotkey": t.hotkey,
                        "enabled": t.enabled
                    } for t in macro_metadata.triggers
                ],
                "actions": [
                    {
                        "category": a.category.value,
                        "type": a.action_type,
                        "description": a.description
                    } for a in macro_metadata.actions
                ]
            }
            
            if include_relationships:
                analysis_result["relationships"] = {
                    "similar_macros": similar_macros,
                    "calls_other_macros": list(macro_metadata.relationships.calls_macros),
                    "called_by_macros": list(macro_metadata.relationships.called_by_macros)
                }
            
            if ctx:
                await ctx.report_progress(100, 100, "Analysis complete")
                await ctx.info(f"Generated comprehensive analysis for {macro_metadata.name}")
            
            return {
                "success": True,
                "data": analysis_result,
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "analysis_version": macro_metadata.analysis_version,
                    "last_analyzed": macro_metadata.last_analyzed.isoformat()
                }
            }
            
        except Exception as e:
            logger.exception("Error in km_analyze_macro_metadata")
            if ctx:
                await ctx.error(f"Metadata analysis error: {e}")
            return {
                "success": False,
                "error": {
                    "code": "SYSTEM_ERROR",
                    "message": "Macro metadata analysis failed",
                    "details": str(e),
                    "recovery_suggestion": "Check macro ID and Keyboard Maestro connection"
                }
            }