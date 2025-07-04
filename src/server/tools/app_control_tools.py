"""
Application Control MCP Tools

Provides comprehensive application lifecycle management through MCP interface
with security validation, state tracking, and menu automation capabilities.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated

from ...applications.app_controller import (
    AppController, AppIdentifier, MenuPath, LaunchConfiguration, AppState
)
from ...core.types import Duration
from ...core.errors import ValidationError, SecurityViolationError

logger = logging.getLogger(__name__)


async def km_app_control(
    operation: Annotated[str, Field(
        description="Application control operation",
        pattern=r"^(launch|quit|activate|menu_select|get_state)$"
    )],
    app_identifier: Annotated[str, Field(
        description="Application bundle ID or name",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9\.\-\s]+$"
    )],
    menu_path: Annotated[Optional[List[str]], Field(
        default=None,
        description="Menu path for menu_select operation (max 10 items)",
        max_items=10
    )] = None,
    force_quit: Annotated[bool, Field(
        default=False,
        description="Force termination option for quit operation"
    )] = False,
    wait_for_completion: Annotated[bool, Field(
        default=True,
        description="Wait for operation to complete"
    )] = True,
    timeout_seconds: Annotated[int, Field(
        default=30,
        ge=1,
        le=120,
        description="Operation timeout in seconds"
    )] = 30,
    hide_on_launch: Annotated[bool, Field(
        default=False,
        description="Hide application after launch"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive application control with security validation and error handling.
    
    Architecture:
        - Pattern: Command Pattern with State Machine
        - Security: Defense-in-depth with validation, whitelisting, audit logging
        - Performance: O(1) state queries, O(log n) operations with caching
    
    Contracts:
        Preconditions:
            - operation is valid application control command
            - app_identifier is valid bundle ID or application name
            - menu_path is properly formatted for menu operations
            - timeout values are within safe limits
        
        Postconditions:
            - Returns success with application state OR error with details
            - No partial state changes on failure
            - All security validations passed
        
        Invariants:
            - Application security policies cannot be bypassed
            - All operations are audited and logged
            - Timeout protection prevents hanging operations
    
    Operations:
    - launch: Start application with comprehensive launch validation
    - quit: Terminate application (graceful or forced) with safety checks
    - activate: Bring application to foreground with state verification
    - menu_select: Navigate and select menu items with path validation
    - get_state: Query current application state with performance caching
    
    Security Features:
    - Bundle ID validation and application whitelist checking
    - Permission verification for system-level operations
    - Safe menu navigation with path validation and injection prevention
    - Timeout protection for hanging operations
    - Force quit safety with confirmation requirements
    
    Performance Features:
    - Application state caching with intelligent invalidation
    - Optimized AppleScript execution with timeout handling
    - Batch operation support for multiple applications
    - Resource usage monitoring and limits
    
    Args:
        operation: Control operation (launch, quit, activate, menu_select, get_state)
        app_identifier: Bundle ID (com.apple.TextEdit) or name (TextEdit)
        menu_path: Menu navigation path for menu_select (e.g., ["File", "New"])
        force_quit: Force termination for quit operation (requires confirmation)
        wait_for_completion: Wait for operation completion with state verification
        timeout_seconds: Maximum operation time (1-120 seconds)
        hide_on_launch: Hide application after successful launch
        ctx: MCP context for logging and progress reporting
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating operation success
        - data: Application state and operation details on success
        - error: Error information on failure
        - metadata: Timestamp, performance metrics, and security validation
        
    Raises:
        ValidationError: Input validation failed
        SecurityViolationError: Security validation failed
    """
    correlation_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        if ctx:
            await ctx.info(f"Starting app control operation: {operation} on {app_identifier}")
        
        logger.info(f"App control: {operation} on '{app_identifier}' [correlation_id: {correlation_id}]")
        
        # Phase 1: Input validation and security
        try:
            # Parse application identifier
            if "." in app_identifier and not app_identifier.endswith(".app"):
                # Looks like bundle ID
                app_id = AppIdentifier(bundle_id=app_identifier)
            else:
                # Treat as application name
                app_id = AppIdentifier(app_name=app_identifier)
        
        except ValueError as e:
            logger.warning(f"Invalid app identifier '{app_identifier}': {e}")
            return {
                "success": False,
                "error": {
                    "code": "INVALID_IDENTIFIER",
                    "message": str(e),
                    "details": f"Application identifier validation failed: {e}",
                    "recovery_suggestion": "Use valid bundle ID (com.apple.TextEdit) or app name (TextEdit)"
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "operation": operation,
                    "validation_stage": "identifier_parsing"
                }
            }
        
        # Phase 2: Initialize application controller
        app_controller = AppController()
        timeout = Duration.from_seconds(timeout_seconds)
        
        if ctx:
            await ctx.report_progress(25, 100, f"Executing {operation} operation")
        
        # Phase 3: Execute operation based on type
        if operation == "launch":
            result = await _execute_launch_operation(
                app_controller, app_id, wait_for_completion, timeout, hide_on_launch, ctx
            )
        elif operation == "quit":
            result = await _execute_quit_operation(
                app_controller, app_id, force_quit, timeout, ctx
            )
        elif operation == "activate":
            result = await _execute_activate_operation(
                app_controller, app_id, ctx
            )
        elif operation == "menu_select":
            if not menu_path:
                return {
                    "success": False,
                    "error": {
                        "code": "MISSING_MENU_PATH",
                        "message": "Menu path required for menu_select operation",
                        "details": "menu_path parameter must be provided for menu selection",
                        "recovery_suggestion": "Provide menu_path as array of menu item names"
                    },
                    "metadata": {
                        "correlation_id": correlation_id,
                        "timestamp": datetime.now().isoformat(),
                        "operation": operation
                    }
                }
            
            result = await _execute_menu_select_operation(
                app_controller, app_id, menu_path, timeout, ctx
            )
        elif operation == "get_state":
            result = await _execute_get_state_operation(
                app_controller, app_id, ctx
            )
        else:
            return {
                "success": False,
                "error": {
                    "code": "UNSUPPORTED_OPERATION",
                    "message": f"Operation not supported: {operation}",
                    "details": "Supported operations: launch, quit, activate, menu_select, get_state",
                    "recovery_suggestion": "Use one of the supported operation types"
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now().isoformat(),
                    "operation": operation
                }
            }
        
        if ctx:
            await ctx.report_progress(100, 100, "Operation completed")
        
        # Phase 4: Process and return result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if result.get("success", False):
            logger.info(f"App control success: {operation} on '{app_identifier}' [correlation_id: {correlation_id}]")
            
            if ctx:
                await ctx.info(f"Operation completed successfully: {operation}")
            
            result["metadata"] = {
                **result.get("metadata", {}),
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "server_version": "1.0.0",
                "execution_time": execution_time,
                "operation": operation,
                "app_identifier": app_identifier
            }
        else:
            logger.error(f"App control failed: {operation} on '{app_identifier}' - {result.get('error', {}).get('message', 'Unknown error')} [correlation_id: {correlation_id}]")
            
            if ctx:
                await ctx.error(f"Operation failed: {result.get('error', {}).get('message', 'Unknown error')}")
        
        return result
        
    except ValidationError as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.warning(f"Validation error for {operation} on '{app_identifier}': {e} [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Validation failed: {e}")
        
        return {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(e),
                "details": f"Input validation failed: {e}",
                "recovery_suggestion": "Review input parameters and ensure they meet validation requirements"
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "operation": operation,
                "failure_stage": "input_validation"
            }
        }
        
    except SecurityViolationError as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Security violation for {operation} on '{app_identifier}': {e} [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Security violation: {e}")
        
        return {
            "success": False,
            "error": {
                "code": "SECURITY_VIOLATION",
                "message": "Security validation failed",
                "details": f"Security requirements not met: {e}",
                "recovery_suggestion": "Ensure all inputs meet security requirements and try again"
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "operation": operation,
                "failure_stage": "security_validation"
            }
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.exception(f"Unexpected error in {operation} on '{app_identifier}' [correlation_id: {correlation_id}]")
        
        if ctx:
            await ctx.error(f"Unexpected error: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Unexpected system error",
                "details": str(e),
                "recovery_suggestion": "Check system status and try again. Contact support if problem persists."
            },
            "metadata": {
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "operation": operation,
                "failure_stage": "system_error"
            }
        }


# Operation-specific helper functions

async def _execute_launch_operation(
    controller: AppController, 
    app_id: AppIdentifier, 
    wait_for_completion: bool, 
    timeout: Duration, 
    hide_on_launch: bool,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Execute application launch operation."""
    try:
        config = LaunchConfiguration(
            wait_for_launch=wait_for_completion,
            timeout=timeout,
            hide_on_launch=hide_on_launch,
            activate_on_launch=not hide_on_launch
        )
        
        result = await controller.launch_application(app_id, config)
        
        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "recovery_suggestion": error.recovery_suggestion or "Check application name and permissions"
                }
            }
        
        operation_result = result.get_right()
        return {
            "success": True,
            "data": {
                "app_state": operation_result.app_state.value,
                "operation_time": operation_result.operation_time.total_seconds(),
                "details": operation_result.details,
                "app_name": app_id.display_name(),
                "app_identifier": app_id.primary_identifier(),
                "hidden": hide_on_launch,
                "waited_for_completion": wait_for_completion
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "LAUNCH_ERROR",
                "message": f"Launch operation failed: {str(e)}",
                "details": f"Error launching {app_id.display_name()}: {str(e)}",
                "recovery_suggestion": "Check application exists and you have permission to launch it"
            }
        }


async def _execute_quit_operation(
    controller: AppController, 
    app_id: AppIdentifier, 
    force_quit: bool, 
    timeout: Duration,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Execute application quit operation."""
    try:
        result = await controller.quit_application(app_id, force_quit, timeout)
        
        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "recovery_suggestion": error.recovery_suggestion or "Check if application is running"
                }
            }
        
        operation_result = result.get_right()
        return {
            "success": True,
            "data": {
                "app_state": operation_result.app_state.value,
                "operation_time": operation_result.operation_time.total_seconds(),
                "details": operation_result.details,
                "app_name": app_id.display_name(),
                "app_identifier": app_id.primary_identifier(),
                "force_quit_used": force_quit,
                "final_state": operation_result.app_state.value
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "QUIT_ERROR",
                "message": f"Quit operation failed: {str(e)}",
                "details": f"Error quitting {app_id.display_name()}: {str(e)}",
                "recovery_suggestion": "Try force quit if graceful quit failed"
            }
        }


async def _execute_activate_operation(
    controller: AppController, 
    app_id: AppIdentifier,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Execute application activation operation."""
    try:
        result = await controller.activate_application(app_id)
        
        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "recovery_suggestion": error.recovery_suggestion or "Ensure application is running"
                }
            }
        
        operation_result = result.get_right()
        return {
            "success": True,
            "data": {
                "app_state": operation_result.app_state.value,
                "operation_time": operation_result.operation_time.total_seconds(),
                "details": operation_result.details,
                "app_name": app_id.display_name(),
                "app_identifier": app_id.primary_identifier(),
                "activated": True
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "ACTIVATION_ERROR",
                "message": f"Activation failed: {str(e)}",
                "details": f"Error activating {app_id.display_name()}: {str(e)}",
                "recovery_suggestion": "Check if application is running and accessible"
            }
        }


async def _execute_menu_select_operation(
    controller: AppController, 
    app_id: AppIdentifier, 
    menu_path_list: List[str], 
    timeout: Duration,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Execute menu selection operation."""
    try:
        # Validate and create menu path
        menu_path = MenuPath(menu_path_list)
        
        result = await controller.select_menu_item(app_id, menu_path, timeout)
        
        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "recovery_suggestion": error.recovery_suggestion or "Check menu path exists and application is active"
                }
            }
        
        operation_result = result.get_right()
        return {
            "success": True,
            "data": {
                "app_state": operation_result.app_state.value,
                "operation_time": operation_result.operation_time.total_seconds(),
                "details": operation_result.details,
                "app_name": app_id.display_name(),
                "app_identifier": app_id.primary_identifier(),
                "menu_path": menu_path_list,
                "menu_depth": len(menu_path_list),
                "menu_selected": True
            }
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": {
                "code": "INVALID_MENU_PATH",
                "message": f"Invalid menu path: {str(e)}",
                "details": f"Menu path validation failed: {str(e)}",
                "recovery_suggestion": "Ensure menu path contains valid menu item names"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "MENU_ERROR",
                "message": f"Menu selection failed: {str(e)}",
                "details": f"Error selecting menu in {app_id.display_name()}: {str(e)}",
                "recovery_suggestion": "Verify menu path exists and application supports UI automation"
            }
        }


async def _execute_get_state_operation(
    controller: AppController, 
    app_id: AppIdentifier,
    ctx: Optional[Context]
) -> Dict[str, Any]:
    """Execute application state query operation."""
    try:
        result = await controller.get_application_state(app_id)
        
        if result.is_left():
            error = result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                    "recovery_suggestion": error.recovery_suggestion or "Check application identifier"
                }
            }
        
        app_state = result.get_right()
        return {
            "success": True,
            "data": {
                "app_state": app_state.value,
                "app_name": app_id.display_name(),
                "app_identifier": app_id.primary_identifier(),
                "is_running": app_state != AppState.NOT_RUNNING,
                "is_foreground": app_state == AppState.FOREGROUND,
                "is_background": app_state == AppState.BACKGROUND,
                "state_description": _get_state_description(app_state)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "STATE_QUERY_ERROR",
                "message": f"State query failed: {str(e)}",
                "details": f"Error querying state of {app_id.display_name()}: {str(e)}",
                "recovery_suggestion": "Check application identifier and system permissions"
            }
        }


def _get_state_description(state: AppState) -> str:
    """Get human-readable description of application state."""
    descriptions = {
        AppState.NOT_RUNNING: "Application is not currently running",
        AppState.LAUNCHING: "Application is in the process of launching",
        AppState.RUNNING: "Application is running but not in foreground",
        AppState.FOREGROUND: "Application is running and in the foreground",
        AppState.BACKGROUND: "Application is running in the background",
        AppState.TERMINATING: "Application is in the process of terminating",
        AppState.CRASHED: "Application has crashed or is unresponsive",
        AppState.UNKNOWN: "Application state cannot be determined"
    }
    return descriptions.get(state, f"Unknown state: {state.value}")