"""
Advanced trigger tools for sophisticated event-driven automation.

This module implements the km_create_trigger_advanced MCP tool, enabling AI to create
intelligent, responsive automation workflows that react to environmental changes.
Supports time-based, file system, application lifecycle, and system event triggers.

Security: Comprehensive input validation and resource protection.
Performance: Efficient event monitoring with debouncing and throttling.
Integration: Full compatibility with condition system (TASK_21) and control flow (TASK_22).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import json

from fastmcp import Context
from fastmcp.exceptions import ToolError

from src.core.triggers import (
    TriggerType, TriggerSpec, TriggerBuilder, TriggerValidator,
    TimeTriggerConfig, FileTriggerConfig, SystemTriggerConfig
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError, MCPError
from src.core.contracts import require, ensure
from src.core.types import MacroId
from src.security.input_sanitizer import InputSanitizer
from src.integration.km_triggers import KMTriggerGenerator, KMTriggerIntegrator
from src.integration.km_client import KMClient

# Setup module logger
logger = logging.getLogger(__name__)


class AdvancedTriggerProcessor:
    """Process and validate advanced trigger requests with comprehensive security."""
    
    def __init__(self, km_client: KMClient):
        self.km_client = km_client
        self.trigger_generator = KMTriggerGenerator()
    
    @require(lambda self, macro_id: isinstance(macro_id, str) and len(macro_id.strip()) > 0)
    @require(lambda self, trigger_type: isinstance(trigger_type, str))
    @require(lambda self, config: isinstance(config, dict))
    async def create_advanced_trigger(
        self,
        macro_identifier: str,
        trigger_type: str,
        trigger_config: Dict[str, Any],
        conditions: Optional[List[Dict[str, Any]]] = None,
        enabled: bool = True,
        priority: int = 0,
        timeout_seconds: int = 30,
        max_executions: Optional[int] = None,
        ctx: Optional[Context] = None
    ) -> Either[MCPError, Dict[str, Any]]:
        """
        Create advanced trigger with comprehensive validation and security.
        
        Architecture: Event-driven with type safety and resource protection
        Security: Input validation, resource limits, path protection
        Performance: Efficient trigger creation with validation caching
        """
        try:
            logger.info(f"Creating advanced trigger for macro: {macro_identifier}")
            
            # Input sanitization and validation
            sanitizer = InputSanitizer()
            
            # Sanitize macro identifier
            macro_id_result = sanitizer.sanitize_macro_identifier(macro_identifier)
            if macro_id_result.is_left():
                return {
                    "success": False,
                    "error": "INVALID_MACRO_ID",
                    "message": macro_id_result.get_left().message
                }
            
            macro_id = MacroId(macro_id_result.get_right())
            
            # Validate trigger type
            trigger_type_result = _validate_trigger_type(trigger_type)
            if trigger_type_result.is_left():
                return {
                    "success": False,
                    "error": "INVALID_TRIGGER_TYPE",
                    "message": trigger_type_result.get_left().message
                }
            
            trigger_type_enum = trigger_type_result.get_right()
            
            # Validate and sanitize trigger configuration
            config_result = _validate_trigger_config(trigger_type_enum, trigger_config)
            if config_result.is_left():
                return {
                    "success": False,
                    "error": "INVALID_TRIGGER_CONFIG",
                    "message": config_result.get_left().message
                }
            
            sanitized_config = config_result.get_right()
            
            # Validate conditions if provided
            if conditions:
                conditions_result = _validate_conditions(conditions)
                if conditions_result.is_left():
                    return {
                        "success": False,
                        "error": "INVALID_CONDITIONS",
                        "message": conditions_result.get_left().message
                    }
                conditions = conditions_result.get_right()
            else:
                conditions = []
            
            # Build trigger specification using fluent API
            builder_result = _build_trigger_spec(
                trigger_type_enum, sanitized_config, conditions,
                enabled, priority, timeout_seconds, max_executions
            )
            
            if builder_result.is_left():
                return {
                    "success": False,
                    "error": "TRIGGER_BUILD_FAILED",
                    "message": builder_result.get_left().message
                }
            
            trigger_spec = builder_result.get_right()
            
            # Additional security validation
            security_result = _perform_security_validation(trigger_spec, sanitized_config)
            if security_result.is_left():
                return {
                    "success": False,
                    "error": "SECURITY_VIOLATION",
                    "message": security_result.get_left().message
                }
            
            # Integrate with Keyboard Maestro
            integrator = KMTriggerIntegrator()
            
            # Validate compatibility
            compatibility_result = integrator.validate_trigger_compatibility(trigger_spec)
            if compatibility_result.is_left():
                return {
                    "success": False,
                    "error": "COMPATIBILITY_ERROR",
                    "message": compatibility_result.get_left().message
                }
            
            # Add trigger to macro
            integration_result = await integrator.add_trigger_to_macro(
                macro_id=macro_id,
                trigger_spec=trigger_spec,
                replace_existing=enabled  # Fixed parameter name
            )
            
            if integration_result.is_left():
                return {
                    "success": False,
                    "error": "INTEGRATION_FAILED",
                    "message": integration_result.get_left().message
                }
            
            integration_details = integration_result.get_right()
            
            logger.info(f"Successfully created trigger {trigger_spec.trigger_id} for macro {macro_id}")
            
            return {
                "success": True,
                "trigger_id": trigger_spec.trigger_id,
                "macro_id": str(macro_id),
                "trigger_type": trigger_type,
                "trigger_config": sanitized_config,
                "conditions": conditions,
                "enabled": enabled,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
                "max_executions": max_executions,
                "replace_existing": enabled,  # Fixed parameter reference
                "km_integration": integration_details,
                "security_validated": True,
                "created_at": trigger_spec.metadata.get("created_at", ""),
                "performance_metrics": {
                    "validation_time_ms": integration_details.get("integration_time_ms", 0),
                    "total_setup_time_ms": integration_details.get("integration_time_ms", 0)
                },
                "trigger_examples": _generate_trigger_examples(trigger_type_enum),
                "next_actions": _suggest_next_actions(trigger_spec)
            }
        
        except Exception as e:
            logger.error(f"Error creating advanced trigger for macro {macro_identifier}: {str(e)}")
            return {
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": f"Failed to create trigger: {str(e)}"
            }


def _validate_trigger_type(trigger_type: str) -> Either[ValidationError, TriggerType]:
    """Validate and convert trigger type string."""
    try:
        # Map user-friendly names to enum values
        type_mapping = {
            "time": [TriggerType.TIME_SCHEDULED, TriggerType.TIME_RECURRING],
            "file": [TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED, TriggerType.FILE_DELETED],
            "system": [TriggerType.APP_LAUNCHED, TriggerType.APP_QUIT, TriggerType.SYSTEM_STARTUP, TriggerType.NETWORK_CONNECTED],
            "user": [TriggerType.USER_IDLE, TriggerType.USER_ACTIVE, TriggerType.USER_LOGIN, TriggerType.USER_LOGOUT]
        }
        
        # First try direct enum lookup
        try:
            return Either.right(TriggerType(trigger_type.lower()))
        except ValueError:
            pass
        
        # Then try category mapping
        if trigger_type.lower() in type_mapping:
            # For category types, use the first as default (user should specify subtype in config)
            return Either.right(type_mapping[trigger_type.lower()][0])
        
        valid_types = list(TriggerType) + list(type_mapping.keys())
        return Either.left(ValidationError(
            field_name="trigger_type",
            value=trigger_type,
            constraint=f"Invalid trigger type. Valid types: {[t.value for t in TriggerType]}"
        ))
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="trigger_type",
            value=trigger_type,
            constraint=f"Error validating trigger type: {str(e)}"
        ))


def _validate_trigger_config(trigger_type: TriggerType, config: Dict[str, Any]) -> Either[ValidationError, Dict[str, Any]]:
    """Validate and sanitize trigger configuration."""
    try:
        sanitized_config = {}
        
        if trigger_type in [TriggerType.TIME_SCHEDULED, TriggerType.TIME_RECURRING]:
            sanitized_config = _validate_time_config(config)
        elif trigger_type in [TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED, TriggerType.FILE_DELETED]:
            result = _validate_file_config(config)
            if result.is_left():
                return result
            sanitized_config = result.get_right()
        elif trigger_type in [TriggerType.APP_LAUNCHED, TriggerType.APP_QUIT]:
            result = _validate_app_config(config)
            if result.is_left():
                return result
            sanitized_config = result.get_right()
        elif trigger_type == TriggerType.USER_IDLE:
            sanitized_config = _validate_idle_config(config)
        elif trigger_type == TriggerType.NETWORK_CONNECTED:
            sanitized_config = _validate_network_config(config)
        else:
            # Generic validation for other types
            sanitized_config = dict(config)
        
        return Either.right(sanitized_config)
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="trigger_config",
            value=str(config),
            constraint=f"Error validating config: {str(e)}"
        ))


def _validate_time_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate time trigger configuration."""
    sanitized = {}
    
    if "schedule_time" in config:
        schedule_time = config["schedule_time"]
        if isinstance(schedule_time, str):
            # Parse ISO format datetime
            sanitized["schedule_time"] = datetime.fromisoformat(schedule_time)
        elif isinstance(schedule_time, datetime):
            sanitized["schedule_time"] = schedule_time
        else:
            raise ValueError("schedule_time must be datetime or ISO format string")
    
    if "recurring_interval_seconds" in config:
        interval_seconds = int(config["recurring_interval_seconds"])
        if interval_seconds < 1:
            raise ValueError("Recurring interval must be at least 1 second")
        sanitized["recurring_interval"] = timedelta(seconds=interval_seconds)
    
    if "cron_pattern" in config:
        pattern = str(config["cron_pattern"]).strip()
        # Basic cron validation is done in TriggerBuilder
        sanitized["recurring_pattern"] = pattern
    
    sanitized["timezone"] = config.get("timezone", "local")
    
    return sanitized


def _validate_file_config(config: Dict[str, Any]) -> Either[ValidationError, Dict[str, Any]]:
    """Validate file trigger configuration."""
    try:
        if "watch_path" not in config:
            return Either.left(ValidationError(
                field_name="watch_path",
                value="missing",
                constraint="File triggers require watch_path"
            ))
        
        # Validate path security
        path_result = TriggerValidator.validate_file_path(config["watch_path"])
        if path_result.is_left():
            return Either.left(ValidationError(
                field_name="watch_path",
                value=config["watch_path"],
                constraint=path_result.get_left().message
            ))
        
        sanitized = {
            "watch_path": path_result.get_right(),
            "recursive": bool(config.get("recursive", False)),
            "ignore_hidden": bool(config.get("ignore_hidden", True)),
            "debounce_seconds": float(config.get("debounce_seconds", 1.0))
        }
        
        if "file_pattern" in config:
            sanitized["file_pattern"] = str(config["file_pattern"])
        
        return Either.right(sanitized)
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="file_config",
            value=str(config),
            constraint=f"Error validating file config: {str(e)}"
        ))


def _validate_app_config(config: Dict[str, Any]) -> Either[ValidationError, Dict[str, Any]]:
    """Validate application trigger configuration."""
    try:
        sanitized = {}
        
        if "app_bundle_id" in config:
            app_result = TriggerValidator.validate_app_identifier(config["app_bundle_id"])
            if app_result.is_left():
                return Either.left(ValidationError(
                    field_name="app_bundle_id",
                    value=config["app_bundle_id"],
                    constraint=app_result.get_left().message
                ))
            sanitized["app_bundle_id"] = app_result.get_right()
        
        if "app_name" in config:
            sanitized["app_name"] = str(config["app_name"]).strip()
        
        if not sanitized:
            return Either.left(ValidationError(
                field_name="app_config",
                value=str(config),
                constraint="App triggers require app_bundle_id or app_name"
            ))
        
        return Either.right(sanitized)
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="app_config",
            value=str(config),
            constraint=f"Error validating app config: {str(e)}"
        ))


def _validate_idle_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user idle configuration."""
    threshold = int(config.get("idle_threshold_seconds", 300))
    if threshold <= 0:
        raise ValueError("Idle threshold must be positive")
    
    return {"idle_threshold_seconds": threshold}


def _validate_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate network trigger configuration."""
    sanitized = {}
    
    if "network_interface" in config:
        interface = str(config["network_interface"]).strip()
        if interface:
            sanitized["network_interface"] = interface
    
    return sanitized


def _validate_conditions(conditions: List[Dict]) -> Either[ValidationError, List[Dict]]:
    """Validate conditional logic for triggers."""
    try:
        validated_conditions = []
        
        for i, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                return Either.left(ValidationError(
                    field_name=f"condition_{i}",
                    value=str(condition),
                    constraint="Condition must be a dictionary"
                ))
            
            # Basic condition validation
            required_fields = ["type", "operator", "operand"]
            for field in required_fields:
                if field not in condition:
                    return Either.left(ValidationError(
                        field_name=f"condition_{i}.{field}",
                        value="missing",
                        constraint=f"Condition must have {field}"
                    ))
            
            validated_conditions.append(condition)
        
        return Either.right(validated_conditions)
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="conditions",
            value=str(conditions),
            constraint=f"Error validating conditions: {str(e)}"
        ))


def _build_trigger_spec(
    trigger_type: TriggerType,
    config: Dict[str, Any],
    conditions: List[Dict],
    enabled: bool,
    priority: int,
    timeout_seconds: int,
    max_executions: Optional[int]
) -> Either[ValidationError, Any]:
    """Build trigger specification using fluent API."""
    try:
        builder = TriggerBuilder()
        
        # Set trigger type and configuration
        if trigger_type == TriggerType.TIME_SCHEDULED:
            if "schedule_time" in config:
                builder = builder.scheduled_at(config["schedule_time"], config.get("timezone", "local"))
        elif trigger_type == TriggerType.TIME_RECURRING:
            if "recurring_interval" in config:
                builder = builder.recurring_every(config["recurring_interval"])
            elif "recurring_pattern" in config:
                builder = builder.cron_pattern(config["recurring_pattern"])
        elif trigger_type == TriggerType.FILE_CREATED:
            builder = builder.when_file_created(config["watch_path"], config.get("recursive", False))
        elif trigger_type == TriggerType.FILE_MODIFIED:
            builder = builder.when_file_modified(config["watch_path"], config.get("file_pattern"))
        elif trigger_type == TriggerType.APP_LAUNCHED:
            if "app_bundle_id" in config:
                builder = builder.when_app_launches(config["app_bundle_id"])
        elif trigger_type == TriggerType.APP_QUIT:
            if "app_bundle_id" in config:
                builder = builder.when_app_quits(config["app_bundle_id"])
        elif trigger_type == TriggerType.USER_IDLE:
            builder = builder.when_user_idle(config["idle_threshold_seconds"])
        elif trigger_type == TriggerType.NETWORK_CONNECTED:
            builder = builder.when_network_connected(config.get("network_interface"))
        
        # Add conditions
        for condition in conditions:
            builder = builder.with_condition(condition)
        
        # Set options
        builder = (builder
                  .enabled(enabled)
                  .with_priority(priority)
                  .with_timeout(timeout_seconds))
        
        if max_executions:
            builder = builder.limit_executions(max_executions)
        
        return builder.build()
        
    except Exception as e:
        return Either.left(ValidationError(
            field_name="trigger_build",
            value="build_error",
            constraint=f"Failed to build trigger: {str(e)}"
        ))


def _perform_security_validation(trigger_spec, config: Dict[str, Any]) -> Either[SecurityError, None]:
    """Perform additional security validation on the trigger."""
    # Resource limits validation
    resource_result = TriggerValidator.validate_resource_limits(trigger_spec)
    if resource_result.is_left():
        return resource_result
    
    # Check for suspicious patterns in configuration
    config_str = json.dumps(config, default=str).lower()
    dangerous_patterns = [
        "rm -rf", "sudo", "password", "key", "secret", 
        "exec", "eval", "system", "shell"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in config_str:
            return Either.left(SecurityError(
                "SUSPICIOUS_CONFIG",
                f"Configuration contains potentially dangerous pattern: {pattern}"
            ))
    
    return Either.right(None)


def _generate_trigger_examples(trigger_type: TriggerType) -> List[Dict[str, Any]]:
    """Generate example configurations for trigger type."""
    examples = []
    
    if trigger_type in [TriggerType.TIME_SCHEDULED, TriggerType.TIME_RECURRING]:
        examples.extend([
            {
                "name": "Daily backup at 2 AM",
                "config": {"cron_pattern": "0 2 * * *"}
            },
            {
                "name": "Every 30 minutes",
                "config": {"recurring_interval_seconds": 1800}
            }
        ])
    
    elif trigger_type in [TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED]:
        examples.extend([
            {
                "name": "Monitor Downloads folder",
                "config": {"watch_path": "~/Downloads", "file_pattern": "*.pdf"}
            },
            {
                "name": "Watch project directory",
                "config": {"watch_path": "~/Projects", "recursive": True}
            }
        ])
    
    elif trigger_type in [TriggerType.APP_LAUNCHED, TriggerType.APP_QUIT]:
        examples.extend([
            {
                "name": "When Photoshop launches",
                "config": {"app_bundle_id": "com.adobe.Photoshop"}
            },
            {
                "name": "When browser quits",
                "config": {"app_bundle_id": "com.google.Chrome"}
            }
        ])
    
    return examples


def _suggest_next_actions(trigger_spec) -> List[str]:
    """Suggest next actions based on trigger type."""
    suggestions = [
        "Test the trigger by temporarily enabling it",
        "Add conditional logic to make trigger smarter",
        "Monitor trigger execution in KM log"
    ]
    
    if trigger_spec.trigger_type in [TriggerType.FILE_CREATED, TriggerType.FILE_MODIFIED]:
        suggestions.append("Ensure watched directory exists and is accessible")
    
    if trigger_spec.trigger_type in [TriggerType.APP_LAUNCHED, TriggerType.APP_QUIT]:
        suggestions.append("Verify application bundle ID is correct")
    
    return suggestions


# Register helper functions for convenience
def create_scheduled_trigger_helper(macro_id: str, schedule_time: str, timezone: str = "local") -> Dict[str, Any]:
    """Helper to create scheduled trigger with minimal configuration."""
    return {
        "trigger_type": "time_scheduled",
        "trigger_config": {
            "schedule_time": schedule_time,
            "timezone": timezone
        }
    }


def create_file_monitor_helper(macro_id: str, watch_path: str, file_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Helper to create file monitoring trigger with minimal configuration."""
    config = {"watch_path": watch_path}
    if file_pattern:
        config["file_pattern"] = file_pattern
    
    return {
        "trigger_type": "file_modified",
        "trigger_config": config
    }