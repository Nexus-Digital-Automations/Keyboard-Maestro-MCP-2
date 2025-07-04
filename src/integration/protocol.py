"""
MCP Protocol Implementation for Keyboard Maestro Integration

Handles Model Context Protocol communication with functional error handling
and immutable message processing patterns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from datetime import datetime
import json
import uuid

from ..core.types import MacroId, ExecutionToken
from ..core.contracts import require, ensure
from .events import KMEvent, EventProcessingResult
from .km_client import KMError, Either


class MCPMessageType(Enum):
    """MCP message types for protocol handling."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MCPMethod(Enum):
    """Supported MCP methods for KM integration."""
    EXECUTE_MACRO = "km_execute_macro"
    LIST_MACROS = "km_list_macros"
    REGISTER_TRIGGER = "km_register_trigger"
    GET_MACRO_STATUS = "km_get_macro_status"
    PROCESS_EVENT = "km_process_event"


@dataclass(frozen=True)
class MCPMessage:
    """Immutable MCP message with validation and transformation support."""
    id: str
    method: Optional[str]
    params: Dict[str, Any]
    message_type: MCPMessageType
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def create_request(
        cls, 
        method: str, 
        params: Dict[str, Any], 
        message_id: Optional[str] = None
    ) -> MCPMessage:
        """Create MCP request message."""
        return cls(
            id=message_id or str(uuid.uuid4()),
            method=method,
            params=params,
            message_type=MCPMessageType.REQUEST
        )
    
    @classmethod
    def create_response(
        cls, 
        request_id: str, 
        result: Dict[str, Any]
    ) -> MCPMessage:
        """Create MCP response message."""
        return cls(
            id=request_id,
            method=None,
            params=result,
            message_type=MCPMessageType.RESPONSE
        )
    
    @classmethod
    def create_error(
        cls, 
        request_id: str, 
        error_code: str, 
        error_message: str, 
        error_data: Optional[Dict[str, Any]] = None
    ) -> MCPMessage:
        """Create MCP error message."""
        error_params = {
            "code": error_code,
            "message": error_message,
            "data": error_data or {}
        }
        return cls(
            id=request_id,
            method=None,
            params=error_params,
            message_type=MCPMessageType.ERROR
        )
    
    def is_valid_request(self) -> bool:
        """Validate request message format."""
        return (
            self.message_type == MCPMessageType.REQUEST and
            self.method is not None and
            isinstance(self.params, dict)
        )
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.params.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = {
            "id": self.id,
            "params": self.params,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.method:
            result["method"] = self.method
        
        if self.message_type == MCPMessageType.ERROR:
            result["error"] = self.params
            del result["params"]
        elif self.message_type == MCPMessageType.RESPONSE:
            result["result"] = self.params
            del result["params"]
        
        return result


@dataclass(frozen=True)
class MCPValidationResult:
    """Result of MCP message validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    sanitized_params: Optional[Dict[str, Any]] = None
    
    @classmethod
    def valid(cls, sanitized_params: Dict[str, Any]) -> MCPValidationResult:
        """Create valid result."""
        return cls(is_valid=True, sanitized_params=sanitized_params)
    
    @classmethod
    def invalid(cls, errors: List[str]) -> MCPValidationResult:
        """Create invalid result."""
        return cls(is_valid=False, errors=errors)


class MCPProtocolHandler:
    """Handles MCP protocol communication with functional patterns."""
    
    def __init__(self):
        self._method_handlers: Dict[str, Callable[[Dict[str, Any]], Either[KMError, Dict[str, Any]]]] = {
            MCPMethod.EXECUTE_MACRO.value: self._handle_execute_macro,
            MCPMethod.LIST_MACROS.value: self._handle_list_macros,
            MCPMethod.REGISTER_TRIGGER.value: self._handle_register_trigger,
            MCPMethod.GET_MACRO_STATUS.value: self._handle_get_macro_status,
            MCPMethod.PROCESS_EVENT.value: self._handle_process_event,
        }
    
    @require(lambda self, message: message.is_valid_request())
    def process_request(self, message: MCPMessage) -> MCPMessage:
        """Process MCP request with functional error handling."""
        # Validate message parameters
        validation = self._validate_request(message)
        if not validation.is_valid:
            return MCPMessage.create_error(
                message.id,
                "INVALID_PARAMS",
                f"Parameter validation failed: {', '.join(validation.errors)}"
            )
        
        # Find and execute handler
        handler = self._method_handlers.get(message.method)
        if not handler:
            return MCPMessage.create_error(
                message.id,
                "METHOD_NOT_FOUND",
                f"Unknown method: {message.method}"
            )
        
        # Execute handler with validated parameters
        result = handler(validation.sanitized_params)
        
        if result.is_right():
            return MCPMessage.create_response(message.id, result.get_right())
        else:
            error = result.get_left()
            return MCPMessage.create_error(
                message.id,
                error.code,
                error.message,
                error.details
            )
    
    def _validate_request(self, message: MCPMessage) -> MCPValidationResult:
        """Validate and sanitize request parameters."""
        method = message.method
        params = message.params
        
        if method == MCPMethod.EXECUTE_MACRO.value:
            return self._validate_execute_macro_params(params)
        elif method == MCPMethod.LIST_MACROS.value:
            return self._validate_list_macros_params(params)
        elif method == MCPMethod.REGISTER_TRIGGER.value:
            return self._validate_register_trigger_params(params)
        elif method == MCPMethod.GET_MACRO_STATUS.value:
            return self._validate_get_macro_status_params(params)
        elif method == MCPMethod.PROCESS_EVENT.value:
            return self._validate_process_event_params(params)
        
        return MCPValidationResult.invalid([f"Unknown method: {method}"])
    
    def _validate_execute_macro_params(self, params: Dict[str, Any]) -> MCPValidationResult:
        """Validate execute_macro parameters."""
        errors = []
        sanitized = {}
        
        # Required: macro_id
        macro_id = params.get("macro_id")
        if not macro_id or not isinstance(macro_id, str):
            errors.append("macro_id is required and must be a string")
        else:
            sanitized["macro_id"] = MacroId(macro_id.strip())
        
        # Optional: trigger_value
        trigger_value = params.get("trigger_value")
        if trigger_value is not None:
            if isinstance(trigger_value, str):
                sanitized["trigger_value"] = trigger_value[:1000]  # Limit length
            else:
                errors.append("trigger_value must be a string")
        
        # Optional: timeout
        timeout = params.get("timeout", 30)
        if isinstance(timeout, (int, float)) and 1 <= timeout <= 300:
            sanitized["timeout"] = timeout
        else:
            errors.append("timeout must be a number between 1 and 300 seconds")
        
        return MCPValidationResult.valid(sanitized) if not errors else MCPValidationResult.invalid(errors)
    
    def _validate_list_macros_params(self, params: Dict[str, Any]) -> MCPValidationResult:
        """Validate list_macros parameters."""
        sanitized = {}
        errors = []
        
        # Optional: group_filter
        group_filter = params.get("group_filter")
        if group_filter is not None:
            if isinstance(group_filter, str) and len(group_filter.strip()) > 0:
                sanitized["group_filter"] = group_filter.strip()
            else:
                errors.append("group_filter must be a non-empty string")
        
        # Optional: enabled_only
        enabled_only = params.get("enabled_only", True)
        if isinstance(enabled_only, bool):
            sanitized["enabled_only"] = enabled_only
        else:
            errors.append("enabled_only must be a boolean")
        
        return MCPValidationResult.valid(sanitized) if not errors else MCPValidationResult.invalid(errors)
    
    def _validate_register_trigger_params(self, params: Dict[str, Any]) -> MCPValidationResult:
        """Validate register_trigger parameters."""
        errors = []
        sanitized = {}
        
        # Required: trigger_id, macro_id, trigger_type
        required_fields = ["trigger_id", "macro_id", "trigger_type"]
        for field in required_fields:
            value = params.get(field)
            if not value or not isinstance(value, str):
                errors.append(f"{field} is required and must be a string")
            else:
                sanitized[field] = value.strip()
        
        # Optional: configuration
        config = params.get("configuration", {})
        if isinstance(config, dict):
            sanitized["configuration"] = config
        else:
            errors.append("configuration must be a dictionary")
        
        return MCPValidationResult.valid(sanitized) if not errors else MCPValidationResult.invalid(errors)
    
    def _validate_get_macro_status_params(self, params: Dict[str, Any]) -> MCPValidationResult:
        """Validate get_macro_status parameters."""
        errors = []
        sanitized = {}
        
        macro_id = params.get("macro_id")
        if not macro_id or not isinstance(macro_id, str):
            errors.append("macro_id is required and must be a string")
        else:
            sanitized["macro_id"] = MacroId(macro_id.strip())
        
        return MCPValidationResult.valid(sanitized) if not errors else MCPValidationResult.invalid(errors)
    
    def _validate_process_event_params(self, params: Dict[str, Any]) -> MCPValidationResult:
        """Validate process_event parameters."""
        errors = []
        sanitized = {}
        
        # Validate event data structure
        event_data = params.get("event")
        if not event_data or not isinstance(event_data, dict):
            errors.append("event is required and must be a dictionary")
        else:
            sanitized["event"] = event_data
        
        return MCPValidationResult.valid(sanitized) if not errors else MCPValidationResult.invalid(errors)
    
    # Handler methods (these would integrate with actual KM client and services)
    
    def _handle_execute_macro(self, params: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Handle macro execution request."""
        # This would integrate with the KMClient
        return Either.right({
            "success": True,
            "execution_token": str(uuid.uuid4()),
            "status": "started"
        })
    
    def _handle_list_macros(self, params: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Handle macro listing request."""
        # This would integrate with the KMClient
        return Either.right({
            "macros": [],
            "total_count": 0,
            "filtered": params.get("group_filter") is not None
        })
    
    def _handle_register_trigger(self, params: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Handle trigger registration request."""
        # This would integrate with trigger management
        return Either.right({
            "trigger_id": params["trigger_id"],
            "status": "registered",
            "active": True
        })
    
    def _handle_get_macro_status(self, params: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Handle macro status request."""
        # This would integrate with the KMClient
        return Either.right({
            "macro_id": params["macro_id"],
            "status": "available",
            "enabled": True,
            "last_executed": None
        })
    
    def _handle_process_event(self, params: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Handle event processing request."""
        # This would integrate with event system
        return Either.right({
            "event_processed": True,
            "processing_time_ms": 25.5,
            "actions_triggered": 1
        })


# Utility functions for MCP protocol handling

def create_mcp_request(method: MCPMethod, **params) -> MCPMessage:
    """Create MCP request message with method enum."""
    return MCPMessage.create_request(method.value, params)


def serialize_mcp_message(message: MCPMessage) -> str:
    """Serialize MCP message to JSON string."""
    return json.dumps(message.to_dict())


def deserialize_mcp_message(json_str: str) -> Either[str, MCPMessage]:
    """Deserialize JSON string to MCP message."""
    try:
        data = json.loads(json_str)
        
        # Determine message type
        if "error" in data:
            msg_type = MCPMessageType.ERROR
            params = data["error"]
        elif "result" in data:
            msg_type = MCPMessageType.RESPONSE
            params = data["result"]
        elif "method" in data:
            msg_type = MCPMessageType.REQUEST
            params = data.get("params", {})
        else:
            return Either.left("Invalid message format")
        
        message = MCPMessage(
            id=data["id"],
            method=data.get("method"),
            params=params,
            message_type=msg_type,
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )
        
        return Either.right(message)
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return Either.left(f"Deserialization error: {str(e)}")


def batch_process_messages(
    handler: MCPProtocolHandler, 
    messages: List[MCPMessage]
) -> List[MCPMessage]:
    """Process multiple MCP messages in batch."""
    return [handler.process_request(msg) for msg in messages if msg.is_valid_request()]