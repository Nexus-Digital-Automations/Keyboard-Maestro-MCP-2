"""
Advanced window management tools for sophisticated multi-monitor workflows.

This module implements the km_window_manager_advanced MCP tool, enabling AI to
create complex window arrangements, cross-monitor workflows, and intelligent
workspace management with grid layouts and coordinate mathematics.

Security: Display bounds validation and window identifier verification
Performance: Efficient positioning algorithms with intelligent caching
Integration: Full compatibility with existing window manager (TASK_16)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import asyncio
import logging
import time

from fastmcp import Context
from fastmcp.exceptions import ToolError

from src.core.displays import DisplayManager, WindowGridPattern, DisplayArrangement
from src.window.grid_manager import AdvancedGridManager, GRID_PATTERN_METADATA
from src.window.advanced_positioning import (
    AdvancedPositioning, WorkspaceManager, SmartPositionRequest, WORKSPACE_TEMPLATES
)
from src.core.either import Either
from src.core.errors import ValidationError, WindowError, MCPError
from src.core.contracts import require, ensure
from src.security.input_sanitizer import InputSanitizer
from src.core.types import MacroId

# Setup module logger
logger = logging.getLogger(__name__)


class AdvancedWindowProcessor:
    """Process and validate advanced window management requests with comprehensive security."""
    
    def __init__(self):
        self.display_manager = DisplayManager()
        self.grid_manager = AdvancedGridManager()
        self.positioning = AdvancedPositioning(self.display_manager)
        self.workspace_manager = WorkspaceManager(self.positioning)
    
    @require(lambda self, operation: isinstance(operation, str) and len(operation) > 0)
    @require(lambda self, window_targets: isinstance(window_targets, list) and len(window_targets) > 0)
    async def process_advanced_window_request(
        self,
        operation: str,
        window_targets: List[str],
        layout_pattern: Optional[str] = None,
        target_displays: Optional[List[int]] = None,
        workspace_name: Optional[str] = None,
        positioning_rules: Optional[Dict[str, Any]] = None,
        preserve_ratios: bool = True,
        animate_transitions: bool = False,
        save_layout: bool = False,
        ctx: Optional[Context] = None
    ) -> Either[MCPError, Dict[str, Any]]:
        """
        Process advanced window management with comprehensive validation and security.
        
        Architecture: Multi-monitor with intelligent positioning and workspace management
        Security: Window identifier validation, display bounds checking, operation validation
        Performance: Efficient grid calculations with caching and batch operations
        """
        try:
            start_time = time.time()
            logger.info(f"Processing advanced window operation: {operation}")
            
            # Input sanitization and validation
            sanitizer = InputSanitizer()
            
            # Validate operation type
            operation_result = _validate_operation_type(operation)
            if operation_result.is_left():
                return {
                    "success": False,
                    "error": "INVALID_OPERATION",
                    "message": operation_result.get_left().message
                }
            
            operation_type = operation_result.get_right()
            
            # Validate and sanitize window targets
            sanitized_targets = []
            for target in window_targets:
                target_result = sanitizer.sanitize_window_identifier(target)
                if target_result.is_left():
                    return {
                        "success": False,
                        "error": "INVALID_WINDOW_TARGET",
                        "message": f"Invalid window target '{target}': {target_result.get_left().message}"
                    }
                sanitized_targets.append(target_result.get_right())
            
            # Validate display targets if provided
            if target_displays:
                display_result = await _validate_display_targets(target_displays, self.display_manager)
                if display_result.is_left():
                    return {
                        "success": False,
                        "error": "INVALID_DISPLAY_TARGETS",
                        "message": display_result.get_left().message
                    }
            
            # Validate layout pattern if provided
            grid_pattern = None
            if layout_pattern:
                pattern_result = _validate_layout_pattern(layout_pattern)
                if pattern_result.is_left():
                    return {
                        "success": False,
                        "error": "INVALID_LAYOUT_PATTERN",
                        "message": pattern_result.get_left().message
                    }
                grid_pattern = pattern_result.get_right()
            
            # Process operation based on type
            if operation_type == "grid_layout":
                result = await self._process_grid_layout(
                    sanitized_targets, grid_pattern, target_displays, preserve_ratios
                )
            elif operation_type == "cross_monitor_move":
                result = await self._process_cross_monitor_move(
                    sanitized_targets, target_displays, preserve_ratios
                )
            elif operation_type == "smart_arrange":
                result = await self._process_smart_arrangement(
                    sanitized_targets, positioning_rules, target_displays
                )
            elif operation_type == "workspace_setup":
                result = await self._process_workspace_operation(
                    workspace_name, sanitized_targets, target_displays, save_layout
                )
            else:
                return {
                    "success": False,
                    "error": "UNSUPPORTED_OPERATION",
                    "message": f"Operation '{operation}' not implemented"
                }
            
            if result.is_left():
                return {
                    "success": False,
                    "error": "OPERATION_FAILED",
                    "message": result.get_left().message
                }
            
            operation_result = result.get_right()
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Advanced window operation completed in {processing_time:.2f}ms")
            
            return {
                "success": True,
                "operation": operation,
                "window_targets": sanitized_targets,
                "results": operation_result,
                "layout_pattern": layout_pattern,
                "target_displays": target_displays,
                "workspace_name": workspace_name,
                "preserve_ratios": preserve_ratios,
                "animate_transitions": animate_transitions,
                "save_layout": save_layout,
                "processing_time_ms": processing_time,
                "display_topology": await self._get_display_summary(),
                "available_patterns": [p.value for p in self.grid_manager.get_supported_patterns()],
                "workspace_templates": list(WORKSPACE_TEMPLATES.keys()),
                "next_actions": _suggest_next_actions(operation_type, operation_result)
            }
            
        except Exception as e:
            logger.error(f"Error processing advanced window request: {str(e)}")
            return {
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": f"Failed to process window operation: {str(e)}"
            }
    
    async def _process_grid_layout(
        self,
        window_targets: List[str],
        pattern: Optional[WindowGridPattern],
        target_displays: Optional[List[int]],
        preserve_ratios: bool
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Process grid layout arrangement."""
        try:
            # Get display topology
            topology_result = await self.display_manager.detect_displays()
            if topology_result.is_left():
                return Either.left(topology_result.get_left())
            
            topology = topology_result.get_right()
            
            # Determine target display
            if target_displays and len(target_displays) > 0:
                target_display = topology.get_display_by_id(target_displays[0])
                if target_display is None:
                    return Either.left(WindowError(f"Display {target_displays[0]} not found"))
            else:
                target_display = topology.get_main_display()
            
            # Determine grid pattern
            if pattern is None:
                pattern_result = await self.grid_manager.calculate_optimal_pattern(
                    len(window_targets), target_display
                )
                if pattern_result.is_left():
                    return Either.left(pattern_result.get_left())
                pattern = pattern_result.get_right()
            
            # Arrange windows in grid
            results = await self.grid_manager.arrange_windows_in_grid(
                window_targets, target_display, pattern, padding=15, cache_layout=True
            )
            
            return results
            
        except Exception as e:
            return Either.left(WindowError(f"Grid layout failed: {str(e)}"))
    
    async def _process_cross_monitor_move(
        self,
        window_targets: List[str],
        target_displays: Optional[List[int]],
        preserve_ratios: bool
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Process cross-monitor window migration."""
        try:
            if not target_displays or len(target_displays) == 0:
                return Either.left(WindowError("Target display required for cross-monitor move"))
            
            results = []
            for i, window_id in enumerate(window_targets):
                display_id = target_displays[min(i, len(target_displays) - 1)]
                
                migration_result = await self.positioning.migrate_window_to_display(
                    window_id, display_id, preserve_relative_position=preserve_ratios
                )
                
                if migration_result.is_left():
                    results.append({
                        "window_identifier": window_id,
                        "success": False,
                        "error": migration_result.get_left().message
                    })
                else:
                    migration = migration_result.get_right()
                    results.append({
                        "window_identifier": migration.window_identifier,
                        "source_display": migration.source_display,
                        "target_display": migration.target_display,
                        "old_position": migration.old_position,
                        "new_position": migration.new_position,
                        "migration_time_ms": migration.migration_time_ms,
                        "success": migration.was_successful()
                    })
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(WindowError(f"Cross-monitor move failed: {str(e)}"))
    
    async def _process_smart_arrangement(
        self,
        window_targets: List[str],
        positioning_rules: Optional[Dict[str, Any]],
        target_displays: Optional[List[int]]
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Process intelligent window arrangement."""
        try:
            # Create smart position requests
            requests = []
            for window_id in window_targets:
                content_type = None
                preferred_size = None
                
                if positioning_rules:
                    window_rules = positioning_rules.get(window_id, {})
                    content_type = window_rules.get("content_type")
                    if "preferred_size" in window_rules:
                        size_data = window_rules["preferred_size"]
                        if isinstance(size_data, dict) and "width" in size_data and "height" in size_data:
                            preferred_size = (size_data["width"], size_data["height"])
                
                request = SmartPositionRequest(
                    window_identifier=window_id,
                    content_type=content_type,
                    preferred_size=preferred_size,
                    avoid_overlap=True
                )
                requests.append(request)
            
            # Determine target display
            target_display_id = None
            if target_displays and len(target_displays) > 0:
                target_display_id = target_displays[0]
            
            # Perform intelligent arrangement
            results = await self.positioning.arrange_windows_intelligently(
                requests, target_display_id
            )
            
            return results
            
        except Exception as e:
            return Either.left(WindowError(f"Smart arrangement failed: {str(e)}"))
    
    async def _process_workspace_operation(
        self,
        workspace_name: Optional[str],
        window_targets: List[str],
        target_displays: Optional[List[int]],
        save_layout: bool
    ) -> Either[WindowError, List[Dict[str, Any]]]:
        """Process workspace setup or restoration."""
        try:
            if not workspace_name:
                return Either.left(WindowError("Workspace name required for workspace operations"))
            
            if save_layout:
                # Save current layout
                requests = [
                    SmartPositionRequest(window_id) for window_id in window_targets
                ]
                save_result = await self.workspace_manager.save_workspace_layout(
                    workspace_name, requests
                )
                if save_result.is_left():
                    return Either.left(save_result.get_left())
                
                return Either.right([{
                    "operation": "save_workspace",
                    "workspace_name": workspace_name,
                    "windows_saved": len(window_targets),
                    "success": True
                }])
            else:
                # Restore workspace layout
                target_display_id = target_displays[0] if target_displays else None
                restore_result = await self.workspace_manager.restore_workspace_layout(
                    workspace_name, target_display_id
                )
                return restore_result
            
        except Exception as e:
            return Either.left(WindowError(f"Workspace operation failed: {str(e)}"))
    
    async def _get_display_summary(self) -> Dict[str, Any]:
        """Get summary of current display topology."""
        try:
            topology_result = await self.display_manager.detect_displays()
            if topology_result.is_left():
                return {"error": "Failed to detect displays"}
            
            topology = topology_result.get_right()
            
            return {
                "display_count": len(topology.displays),
                "main_display_id": topology.main_display_id,
                "arrangement": topology.arrangement.value,
                "total_bounds": topology.total_bounds,
                "displays": [
                    {
                        "id": d.display_id,
                        "name": d.name,
                        "resolution": d.resolution,
                        "position": d.position,
                        "is_main": d.is_main
                    }
                    for d in topology.displays
                ]
            }
            
        except Exception:
            return {"error": "Display topology unavailable"}


def _validate_operation_type(operation: str) -> Either[ValidationError, str]:
    """Validate window management operation type."""
    valid_operations = {
        "grid_layout", "cross_monitor_move", "smart_arrange", "workspace_setup"
    }
    
    if operation.lower() not in valid_operations:
        return Either.left(ValidationError(
            field_name="operation",
            value=operation,
            constraint=f"Valid operations: {', '.join(valid_operations)}"
        ))
    
    return Either.right(operation.lower())


async def _validate_display_targets(
    target_displays: List[int],
    display_manager: DisplayManager
) -> Either[ValidationError, None]:
    """Validate target display indices."""
    try:
        topology_result = await display_manager.detect_displays()
        if topology_result.is_left():
            return Either.left(ValidationError(
                field_name="target_displays",
                value=str(target_displays),
                constraint="Failed to detect available displays"
            ))
        
        topology = topology_result.get_right()
        available_ids = [d.display_id for d in topology.displays]
        
        for display_id in target_displays:
            if display_id not in available_ids:
                return Either.left(ValidationError(
                    field_name="target_displays",
                    value=str(display_id),
                    constraint=f"Available displays: {available_ids}"
                ))
        
        return Either.right(None)
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="target_displays",
            value=str(target_displays),
            constraint=f"Display validation failed: {str(e)}"
        ))


def _validate_layout_pattern(pattern: str) -> Either[ValidationError, WindowGridPattern]:
    """Validate window layout pattern."""
    try:
        return Either.right(WindowGridPattern(pattern))
    except ValueError:
        valid_patterns = [p.value for p in WindowGridPattern]
        return Either.left(ValidationError(
            field_name="layout_pattern",
            value=pattern,
            constraint=f"Valid patterns: {', '.join(valid_patterns)}"
        ))


def _suggest_next_actions(operation_type: str, results: List[Dict[str, Any]]) -> List[str]:
    """Suggest next actions based on operation results."""
    suggestions = []
    
    if operation_type == "grid_layout":
        suggestions.extend([
            "Test window arrangement by interacting with applications",
            "Save current layout as workspace for future use",
            "Adjust grid pattern if window sizes don't fit content well"
        ])
    elif operation_type == "cross_monitor_move":
        suggestions.extend([
            "Verify windows are positioned correctly on target displays",
            "Use smart_arrange for better content-aware positioning",
            "Consider saving multi-monitor layout as workspace"
        ])
    elif operation_type == "smart_arrange":
        suggestions.extend([
            "Customize positioning rules for specific window types",
            "Create workspace template for this arrangement",
            "Test overlap avoidance with different window sizes"
        ])
    elif operation_type == "workspace_setup":
        suggestions.extend([
            "Test workspace restoration across different display configurations",
            "Create additional workspace templates for different workflows",
            "Consider automation triggers for automatic workspace switching"
        ])
    
    return suggestions


# Helper functions for common window management patterns
def create_development_workspace(target_display_id: Optional[int] = None) -> Dict[str, Any]:
    """Helper to create development workspace configuration."""
    return {
        "operation": "workspace_setup",
        "workspace_name": "development",
        "window_targets": ["VS Code", "Terminal", "Safari"],
        "target_displays": [target_display_id] if target_display_id else None,
        "positioning_rules": {
            "VS Code": {"content_type": "editor", "preferred_size": {"width": 1200, "height": 800}},
            "Terminal": {"content_type": "terminal", "preferred_size": {"width": 800, "height": 600}},
            "Safari": {"content_type": "browser", "preferred_size": {"width": 1000, "height": 700}}
        }
    }


def create_grid_layout_config(
    window_targets: List[str],
    pattern: str = "2x2",
    target_display_id: Optional[int] = None
) -> Dict[str, Any]:
    """Helper to create grid layout configuration."""
    return {
        "operation": "grid_layout",
        "window_targets": window_targets,
        "layout_pattern": pattern,
        "target_displays": [target_display_id] if target_display_id else None,
        "preserve_ratios": True
    }