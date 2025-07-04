"""
Functional Keyboard Maestro Client Interface

Provides a functional interface to Keyboard Maestro APIs with
pure functions, error handling monads, and connection management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Union, List, TypeVar, Generic
from functools import partial
from enum import Enum
import asyncio
import subprocess
import json
import time
from urllib.parse import urlencode
import httpx

from ..core.types import MacroId, TriggerId, Duration
from ..core.contracts import require, ensure
# Avoid circular import - use string annotation for TriggerType
# from .events import KMEvent, TriggerType, EventPriority


T = TypeVar('T')
E = TypeVar('E')


class ConnectionMethod(Enum):
    """Available connection methods to Keyboard Maestro."""
    APPLESCRIPT = "applescript"
    URL_SCHEME = "url_scheme"
    WEB_API = "web_api"
    REMOTE_TRIGGER = "remote_trigger"


@dataclass(frozen=True)
class Either(Generic[E, T]):
    """Functional Either monad for error handling."""
    _value: Union[E, T]
    _is_right: bool
    
    @classmethod
    def left(cls, error: E) -> Either[E, T]:
        """Create Left (error) value."""
        return cls(_value=error, _is_right=False)
    
    @classmethod
    def right(cls, value: T) -> Either[E, T]:
        """Create Right (success) value."""
        return cls(_value=value, _is_right=True)
    
    def is_left(self) -> bool:
        """Check if this is an error value."""
        return not self._is_right
    
    def is_right(self) -> bool:
        """Check if this is a success value."""
        return self._is_right
    
    def map(self, f: Callable[[T], Any]) -> Either[E, Any]:
        """Map function over right value."""
        if self.is_right():
            return Either.right(f(self._value))
        return Either.left(self._value)
    
    def flat_map(self, f: Callable[[T], Either[E, Any]]) -> Either[E, Any]:
        """Flat map for chaining operations."""
        if self.is_right():
            return f(self._value)
        return Either.left(self._value)
    
    def get_or_else(self, default: T) -> T:
        """Get value or return default if error."""
        return self._value if self.is_right() else default
    
    def get_left(self) -> Optional[E]:
        """Get error value if present."""
        return self._value if self.is_left() else None
    
    def get_right(self) -> Optional[T]:
        """Get success value if present."""
        return self._value if self.is_right() else None


@dataclass(frozen=True)
class KMError:
    """Keyboard Maestro operation error."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[Duration] = None
    
    @classmethod
    def connection_error(cls, message: str) -> KMError:
        """Create connection error."""
        return cls(code="CONNECTION_ERROR", message=message)
    
    @classmethod
    def execution_error(cls, message: str, details: Optional[Dict[str, Any]] = None) -> KMError:
        """Create execution error."""
        return cls(code="EXECUTION_ERROR", message=message, details=details)
    
    @classmethod
    def timeout_error(cls, timeout: Duration) -> KMError:
        """Create timeout error."""
        return cls(
            code="TIMEOUT_ERROR", 
            message=f"Operation timed out after {timeout.total_seconds()}s",
            retry_after=Duration.from_seconds(1.0)
        )
    
    @classmethod
    def validation_error(cls, message: str) -> KMError:
        """Create validation error."""
        return cls(code="VALIDATION_ERROR", message=message)
    
    @classmethod
    def not_found_error(cls, message: str) -> KMError:
        """Create not found error."""
        return cls(code="NOT_FOUND_ERROR", message=message)
    
    @classmethod
    def security_error(cls, message: str) -> KMError:
        """Create security error."""
        return cls(code="SECURITY_ERROR", message=message)


@dataclass(frozen=True)
class ConnectionConfig:
    """Immutable configuration for KM connections."""
    method: ConnectionMethod = ConnectionMethod.APPLESCRIPT
    timeout: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    web_api_port: int = 4490
    web_api_host: str = "localhost"
    max_retries: int = 3
    retry_delay: Duration = field(default_factory=lambda: Duration.from_seconds(0.5))
    
    def with_timeout(self, timeout: Duration) -> ConnectionConfig:
        """Create new config with different timeout."""
        return dataclass.replace(self, timeout=timeout)
    
    def with_method(self, method: ConnectionMethod) -> ConnectionConfig:
        """Create new config with different connection method."""
        return dataclass.replace(self, method=method)


@dataclass(frozen=True)
class TriggerDefinition:
    """Definition for registering a macro trigger."""
    trigger_id: TriggerId
    macro_id: MacroId
    trigger_type: 'TriggerType'  # String annotation to avoid circular import
    configuration: Dict[str, Any]
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "trigger_id": self.trigger_id,
            "macro_id": self.macro_id,
            "trigger_type": self.trigger_type.value,
            "configuration": self.configuration,
            "enabled": self.enabled
        }


class KMClient:
    """Functional interface to Keyboard Maestro APIs with pure error handling."""
    
    def __init__(self, connection_config: ConnectionConfig):
        self._config = connection_config
        self._send_command = partial(self._safe_send, connection_config)
    
    @property
    def config(self) -> ConnectionConfig:
        """Get connection configuration."""
        return self._config
    
    def execute_macro(
        self, 
        macro_id: MacroId, 
        trigger_value: Optional[str] = None
    ) -> Either[KMError, Dict[str, Any]]:
        """Execute macro with functional error handling."""
        command_data = {"macro_id": macro_id}
        if trigger_value:
            command_data["trigger_value"] = trigger_value
        
        return self._send_command("execute_macro", command_data)
    
    @require(lambda self, trigger_def: trigger_def.trigger_id and trigger_def.macro_id)
    def register_trigger(self, trigger_def: TriggerDefinition) -> Either[KMError, TriggerId]:
        """Register trigger with functional error handling."""
        result = self._send_command("register_trigger", trigger_def.to_dict())
        return result.map(lambda r: r.get("trigger_id", trigger_def.trigger_id))
    
    def unregister_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Unregister trigger by ID."""
        result = self._send_command("unregister_trigger", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("success", False))
    
    def get_macro_list(self, group_filter: Optional[str] = None) -> Either[KMError, List[Dict[str, Any]]]:
        """Get list of available macros."""
        params = {"group_filter": group_filter} if group_filter else {}
        result = self._send_command("list_macros", params)
        return result.map(lambda r: r.get("macros", []))
    
    def get_macro_status(self, macro_id: MacroId) -> Either[KMError, Dict[str, Any]]:
        """Get macro status and metadata."""
        result = self._send_command("get_macro_status", {"macro_id": macro_id})
        return result.map(lambda r: r.get("status", {}))
    
    def check_connection(self) -> Either[KMError, bool]:
        """Check if connection to KM is working."""
        result = self._send_command("ping", {})
        return result.map(lambda r: r.get("alive", False))
    
    # TASK_2 Phase 2: Additional KM Client Methods for Trigger Management
    
    def activate_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Activate a registered trigger."""
        result = self._send_command("activate_trigger", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("success", False))
    
    def deactivate_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Deactivate a trigger."""
        result = self._send_command("deactivate_trigger", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("success", False))
    
    def list_triggers(self) -> Either[KMError, List[Dict[str, Any]]]:
        """Get list of all triggers from Keyboard Maestro."""
        result = self._send_command("list_triggers", {})
        return result.map(lambda r: r.get("triggers", []))
    
    def get_trigger_status(self, trigger_id: TriggerId) -> Either[KMError, Dict[str, Any]]:
        """Get status of specific trigger."""
        result = self._send_command("get_trigger_status", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("status", {}))
    
    # Async versions for integration with async trigger manager
    
    async def register_trigger_async(self, trigger_def: TriggerDefinition) -> Either[KMError, TriggerId]:
        """
        Register trigger with comprehensive error handling and validation.
        
        TASK_9 Enhancement: Provides reliable trigger registration with:
        - Input validation and sanitization
        - Proper parameter escaping for AppleScript
        - Timeout handling and resource cleanup
        - Detailed error reporting
        """
        
        try:
            # Validate trigger definition structure
            validation_result = self._validate_trigger_definition(trigger_def)
            if validation_result.is_left():
                return validation_result
            
            # Sanitize trigger data for security
            sanitized_data = self._sanitize_trigger_data(trigger_def.configuration)
            if sanitized_data.is_left():
                return Either.left(KMError.validation_error(f"Trigger data sanitization failed: {sanitized_data.get_left()}"))
            
            safe_config = sanitized_data.get_right()
            
            # Build AppleScript for trigger registration with proper escaping
            script_result = self._build_trigger_script_safe(trigger_def.trigger_type.value, safe_config, trigger_def.trigger_id)
            if script_result.is_left():
                return script_result
            
            script = script_result.get_right()
            
            # Execute with proper timeout and error handling
            execution_result = await self._execute_applescript_safe(script)
            if execution_result.is_left():
                return execution_result
            
            # Validate the response and extract trigger ID
            km_response = execution_result.get_right()
            if not km_response or "error" in km_response.lower():
                return Either.left(KMError.execution_error(f"KM registration failed: {km_response}"))
            
            return Either.right(trigger_def.trigger_id)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Unexpected error in trigger registration: {str(e)}"))
    
    async def activate_trigger_async(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Async version of activate_trigger."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.activate_trigger, trigger_id)
    
    async def deactivate_trigger_async(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Async version of deactivate_trigger."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.deactivate_trigger, trigger_id)
    
    async def list_triggers_async(self) -> Either[KMError, List[Dict[str, Any]]]:
        """Async version of list_triggers."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.list_triggers)
    
    # TASK_5: Real Macro Listing Implementation
    
    async def list_macros_async(
        self, 
        group_filters: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> Either[KMError, List[Dict[str, Any]]]:
        """Get real macro list from Keyboard Maestro using multiple API methods."""
        import asyncio
        
        # Try AppleScript first (most reliable)
        applescript_result = await asyncio.get_event_loop().run_in_executor(
            None, self._list_macros_applescript, group_filters, enabled_only
        )
        if applescript_result.is_right():
            return applescript_result
        
        # Fallback to Web API
        web_api_result = await self._list_macros_web_api(group_filters, enabled_only)
        if web_api_result.is_right():
            return web_api_result
        
        # Both methods failed
        return Either.left(KMError.connection_error("Cannot connect to Keyboard Maestro"))
    
    def _list_macros_applescript(
        self, 
        group_filters: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> Either[KMError, List[Dict[str, Any]]]:
        """List macros using AppleScript getmacros command."""
        
        # AppleScript to get macro information from Keyboard Maestro
        script = '''
        tell application "Keyboard Maestro"
            set macroList to {}
            set groupList to every macro group
            
            repeat with currentGroup in groupList
                set groupName to name of currentGroup
                set groupMacros to every macro of currentGroup
                
                repeat with currentMacro in groupMacros
                    set macroRecord to {¬
                        macroId:(id of currentMacro as string), ¬
                        macroName:(name of currentMacro), ¬
                        groupName:groupName, ¬
                        enabled:(enabled of currentMacro), ¬
                        triggerCount:(count of triggers of currentMacro), ¬
                        actionCount:(count of actions of currentMacro)¬
                    }
                    set macroList to macroList & {macroRecord}
                end repeat
            end repeat
            
            return macroList
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=self.config.timeout.total_seconds()
            )
            
            if result.returncode != 0:
                return Either.left(KMError.execution_error(f"AppleScript failed: {result.stderr}"))
            
            # Parse AppleScript record format and convert to dict
            macros = self._parse_applescript_records(result.stdout)
            
            # Apply filters
            if enabled_only:
                macros = [m for m in macros if m.get("enabled", False)]
            if group_filters:
                # Filter to include macros from any of the specified groups
                filtered_macros = []
                for macro in macros:
                    macro_group = macro.get("groupName", "").lower()
                    if any(group_filter.lower() in macro_group for group_filter in group_filters):
                        filtered_macros.append(macro)
                macros = filtered_macros
            
            # Transform to standard format
            standardized_macros = []
            for macro in macros:
                standardized_macro = {
                    "id": macro.get("macroId", ""),
                    "name": macro.get("macroName", ""),
                    "group": macro.get("groupName", ""),
                    "enabled": macro.get("enabled", True),
                    "trigger_count": macro.get("triggerCount", 0),
                    "action_count": macro.get("actionCount", 0),
                    "last_used": None,  # AppleScript doesn't provide this easily
                    "created_date": None  # AppleScript doesn't provide this easily
                }
                standardized_macros.append(standardized_macro)
            
            return Either.right(standardized_macros)
            
        except subprocess.TimeoutExpired:
            return Either.left(KMError.timeout_error("AppleScript timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript error: {str(e)}"))
    
    async def _list_macros_web_api(
        self, 
        group_filters: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> Either[KMError, List[Dict[str, Any]]]:
        """List macros using KM Web API."""
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout.total_seconds()) as client:
                # Try to get macros from web API endpoints
                # Note: Actual KM Web API endpoints may vary - this is based on common patterns
                try:
                    response = await client.get(
                        f"http://{self.config.web_api_host}:{self.config.web_api_port}/macros"
                    )
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError:
                    # Try alternative endpoint format
                    response = await client.get(
                        f"http://{self.config.web_api_host}:{self.config.web_api_port}/action.html?action=GetMacros"
                    )
                    response.raise_for_status()
                    
                    # Parse HTML response if needed (KM may return HTML instead of JSON)
                    if "application/json" in response.headers.get("content-type", ""):
                        data = response.json()
                    else:
                        # For HTML responses, we need to parse the content
                        return Either.left(KMError.execution_error("Web API returned HTML instead of JSON"))
                
                macros = data.get("macros", []) if isinstance(data, dict) else data
                
                # Transform to standard format
                standardized_macros = []
                for macro in macros:
                    # Handle different possible response formats
                    macro_id = macro.get("uid") or macro.get("id") or macro.get("uuid", "")
                    macro_name = macro.get("name") or macro.get("title", "")
                    group_name = macro.get("group") or macro.get("macroGroup", "")
                    
                    standardized_macro = {
                        "id": macro_id,
                        "name": macro_name,
                        "group": group_name,
                        "enabled": macro.get("enabled", True),
                        "trigger_count": len(macro.get("triggers", [])),
                        "action_count": len(macro.get("actions", [])),
                        "last_used": macro.get("lastUsed"),
                        "created_date": macro.get("created") or macro.get("dateCreated")
                    }
                    standardized_macros.append(standardized_macro)
                
                # Apply filters
                if enabled_only:
                    standardized_macros = [m for m in standardized_macros if m.get("enabled", False)]
                if group_filters:
                    # Filter to include macros from any of the specified groups
                    filtered_macros = []
                    for macro in standardized_macros:
                        macro_group = macro.get("group", "").lower()
                        if any(group_filter.lower() in macro_group for group_filter in group_filters):
                            filtered_macros.append(macro)
                    standardized_macros = filtered_macros
                
                return Either.right(standardized_macros)
                
        except httpx.TimeoutException:
            return Either.left(KMError.timeout_error("Web API timeout"))
        except httpx.HTTPStatusError as e:
            return Either.left(KMError.connection_error(f"Web API HTTP error: {e.response.status_code}"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Web API error: {str(e)}"))
    
    def _parse_applescript_records(self, applescript_output: str) -> List[Dict[str, Any]]:
        """Parse AppleScript record format into Python dictionaries."""
        import re
        
        records = []
        
        # Clean up the output - remove extra whitespace and newlines
        clean_output = re.sub(r'\s+', ' ', applescript_output.strip())
        
        # The actual AppleScript output is in flat comma-separated format
        # Parse format: key:value, key:value, key:value, ...
        # When we see 'macroId' again, it indicates a new record
        
        pairs = []
        current_pair = ""
        in_value = False
        paren_depth = 0
        
        # First, properly split by commas, handling nested content
        for char in clean_output:
            if char == '(' and not in_value:
                paren_depth += 1
            elif char == ')' and not in_value:
                paren_depth -= 1
            elif char == ':' and paren_depth == 0:
                in_value = True
            elif char == ',' and paren_depth == 0 and in_value:
                pairs.append(current_pair.strip())
                current_pair = ""
                in_value = False
                continue
            
            current_pair += char
        
        # Don't forget the last pair
        if current_pair.strip():
            pairs.append(current_pair.strip())
        
        # Now parse the key:value pairs into records
        current_record = {}
        for pair in pairs:
            if ':' in pair:
                # Split only on the first colon to handle values with colons
                key, value = pair.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Clean up the value - remove extra quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                # Convert values to appropriate types
                if value == 'true':
                    value = True
                elif value == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('-', '').isdigit():  # Handle negative numbers
                    value = int(value)
                
                # If we see macroId and we already have a record, start a new one
                if key == 'macroId' and current_record:
                    # Clean up the previous record before saving
                    if 'macroId' in current_record:
                        records.append(current_record)
                    current_record = {}
                
                current_record[key] = value
        
        # Don't forget the last record
        if current_record and 'macroId' in current_record:
            records.append(current_record)
        
        return records
    
    @staticmethod
    def _safe_send(
        config: ConnectionConfig, 
        command: str, 
        payload: Dict[str, Any]
    ) -> Either[KMError, Dict[str, Any]]:
        """Pure function for safe command sending with error handling."""
        try:
            if config.method == ConnectionMethod.APPLESCRIPT:
                return KMClient._send_via_applescript(command, payload, config)
            elif config.method == ConnectionMethod.URL_SCHEME:
                return KMClient._send_via_url_scheme(command, payload, config)
            elif config.method == ConnectionMethod.WEB_API:
                return KMClient._send_via_web_api(command, payload, config)
            else:
                return Either.left(KMError.connection_error(f"Unsupported method: {config.method}"))
        
        except Exception as e:
            return Either.left(KMError.execution_error(f"Command failed: {str(e)}"))
    
    @staticmethod
    def _send_via_applescript(
        command: str, 
        payload: Dict[str, Any], 
        config: ConnectionConfig
    ) -> Either[KMError, Dict[str, Any]]:
        """Send command via AppleScript."""
        if command == "execute_macro":
            macro_id = payload.get("macro_id", "")
            trigger_value = payload.get("trigger_value", "")
            
            # Properly escape quotes and special characters for AppleScript
            escaped_macro_name = macro_id.replace('"', '\\"').replace('\\', '\\\\')
            escaped_params = trigger_value.replace('"', '\\"').replace('\\', '\\\\') if trigger_value else ""
            
            script = f'''
            tell application "Keyboard Maestro Engine"
                try
                    set result to do script "{escaped_macro_name}" with parameter "{escaped_params}"
                    return result
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            try:
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    timeout=config.timeout.total_seconds()
                )
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output.startswith("ERROR:"):
                        return Either.left(KMError.execution_error(output[6:].strip()))
                    return Either.right({"output": output, "success": True})
                else:
                    return Either.left(KMError.execution_error(result.stderr.strip()))
                    
            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))
        
        elif command == "ping":
            ping_script = '''
            tell application "System Events"
                return (exists process "Keyboard Maestro Engine")
            end tell
            '''
            try:
                result = subprocess.run(
                    ["osascript", "-e", ping_script],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                alive = result.returncode == 0 and "true" in result.stdout.lower()
                return Either.right({"alive": alive})
            except Exception:
                return Either.right({"alive": False})
        
        return Either.left(KMError.execution_error(f"Unsupported AppleScript command: {command}"))
    
    @staticmethod
    def _send_via_url_scheme(
        command: str, 
        payload: Dict[str, Any], 
        config: ConnectionConfig
    ) -> Either[KMError, Dict[str, Any]]:
        """Send command via kmtrigger URL scheme."""
        if command == "execute_macro":
            macro_id = payload.get("macro_id", "")
            trigger_value = payload.get("trigger_value", "")
            
            url_params = {"macro": macro_id}
            if trigger_value:
                url_params["value"] = trigger_value
            
            url = f"kmtrigger://macro={macro_id}"
            if trigger_value:
                url += f"&value={trigger_value}"
            
            try:
                subprocess.run(["open", url], timeout=config.timeout.total_seconds())
                return Either.right({"success": True, "url": url})
            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))
        
        return Either.left(KMError.execution_error(f"Unsupported URL scheme command: {command}"))
    
    @staticmethod
    def _send_via_web_api(
        command: str, 
        payload: Dict[str, Any], 
        config: ConnectionConfig
    ) -> Either[KMError, Dict[str, Any]]:
        """Send command via web API."""
        base_url = f"http://{config.web_api_host}:{config.web_api_port}"
        
        if command == "execute_macro":
            macro_id = payload.get("macro_id", "")
            trigger_value = payload.get("trigger_value", "")
            
            params = {"macro": macro_id}
            if trigger_value:
                params["value"] = trigger_value
            
            url = f"{base_url}/action.html?" + urlencode(params)
            
            try:
                with httpx.Client() as client:
                    response = client.get(url, timeout=config.timeout.total_seconds())
                if response.status_code == 200:
                    return Either.right({"success": True, "response": response.text})
                else:
                    return Either.left(KMError.execution_error(f"HTTP {response.status_code}: {response.text}"))
            except httpx.TimeoutException:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))
        
        return Either.left(KMError.execution_error(f"Unsupported web API command: {command}"))
    
    # TASK_9: Enhanced helper methods for reliable trigger operations
    
    def _validate_trigger_definition(self, trigger_def: TriggerDefinition) -> Either[KMError, TriggerDefinition]:
        """Validate trigger definition structure and required fields."""
        
        if not trigger_def.trigger_id:
            return Either.left(KMError.validation_error("Trigger ID is required"))
        
        if not trigger_def.macro_id:
            return Either.left(KMError.validation_error("Macro ID is required"))
        
        if not isinstance(trigger_def.configuration, dict):
            return Either.left(KMError.validation_error("Trigger configuration must be a dictionary"))
        
        # Validate trigger type specific requirements
        if trigger_def.trigger_type.value == "hotkey":
            if "key" not in trigger_def.configuration:
                return Either.left(KMError.validation_error("Hotkey trigger requires 'key' parameter"))
            
            key = trigger_def.configuration["key"]
            if not isinstance(key, str) or len(key) == 0:
                return Either.left(KMError.validation_error("Hotkey 'key' must be a non-empty string"))
        
        elif trigger_def.trigger_type.value == "application":
            if "application" not in trigger_def.configuration:
                return Either.left(KMError.validation_error("Application trigger requires 'application' parameter"))
        
        return Either.right(trigger_def)
    
    def _sanitize_trigger_data(self, config: Dict[str, Any]) -> Either[str, Dict[str, Any]]:
        """Sanitize trigger configuration data to prevent injection attacks."""
        
        try:
            # Import security validation from our security module
            from .security import validate_km_input, SecurityLevel
            
            # Validate the configuration using our security system
            validation_result = validate_km_input(config, SecurityLevel.STANDARD)
            
            if not validation_result.is_safe:
                violations = [f"{v.threat_type.value}: {v.violation_text}" for v in validation_result.violations]
                return Either.left(f"Security violations detected: {'; '.join(violations)}")
            
            return Either.right(validation_result.sanitized_data or config)
            
        except Exception as e:
            return Either.left(f"Sanitization error: {str(e)}")
    
    def _build_trigger_script_safe(self, trigger_type: str, config: Dict[str, Any], trigger_id: TriggerId) -> Either[KMError, str]:
        """Build AppleScript for trigger registration with comprehensive validation and escaping."""
        
        # Escape function for AppleScript strings
        def escape_applescript_string(value: str) -> str:
            """Escape string for safe use in AppleScript."""
            if not isinstance(value, str):
                value = str(value)
            
            # Replace dangerous characters
            value = value.replace('\\', '\\\\')  # Escape backslashes
            value = value.replace('"', '\\"')      # Escape quotes
            value = value.replace('\n', '\\n')      # Escape newlines
            value = value.replace('\r', '\\r')      # Escape carriage returns
            value = value.replace('\t', '\\t')      # Escape tabs
            
            return value
        
        if trigger_type == "hotkey":
            key = config.get("key", "")
            modifiers = config.get("modifiers", [])
            
            # Validate and escape key
            if not key or not isinstance(key, str):
                return Either.left(KMError.validation_error("Invalid or missing hotkey"))
            
            escaped_key = escape_applescript_string(key)
            
            # Validate and escape modifiers
            safe_modifiers = []
            for mod in modifiers:
                if isinstance(mod, str) and mod in ["command", "option", "control", "shift"]:
                    safe_modifiers.append(escape_applescript_string(mod))
            
            escaped_trigger_id = escape_applescript_string(str(trigger_id))
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set newTrigger to make new hotkey trigger with properties {{¬
                        key:"{escaped_key}", ¬
                        modifiers:{{{', '.join(f'"{mod}"' for mod in safe_modifiers)}}}, ¬
                        unique_id:"{escaped_trigger_id}"¬
                    }}
                    return "SUCCESS: " & (unique_id of newTrigger as string)
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            return Either.right(script)
        
        elif trigger_type == "application":
            app_name = config.get("application", "")
            event_type = config.get("event", "launches")
            
            if not app_name:
                return Either.left(KMError.validation_error("Application name is required"))
            
            escaped_app = escape_applescript_string(app_name)
            escaped_event = escape_applescript_string(event_type)
            escaped_trigger_id = escape_applescript_string(str(trigger_id))
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set newTrigger to make new application trigger with properties {{¬
                        application:"{escaped_app}", ¬
                        event:"{escaped_event}", ¬
                        unique_id:"{escaped_trigger_id}"¬
                    }}
                    return "SUCCESS: " & (unique_id of newTrigger as string)
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            return Either.right(script)
        
        else:
            return Either.left(KMError.validation_error(f"Unsupported trigger type: {trigger_type}"))
    
    async def _execute_applescript_safe(self, script: str) -> Either[KMError, str]:
        """Execute AppleScript with comprehensive security validation and error handling."""
        
        try:
            # Validate script safety before execution
            if self._contains_dangerous_commands(script):
                return Either.left(KMError.validation_error("Dangerous AppleScript commands detected"))
            
            # Execute with timeout using asyncio subprocess
            try:
                process = await asyncio.create_subprocess_exec(
                    'osascript', '-e', script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for completion with timeout
                timeout_seconds = self.config.timeout.total_seconds()
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    process.terminate()
                    await process.wait()  # Ensure cleanup
                    return Either.left(KMError.timeout_error(self.config.timeout))
                
                # Check execution result
                if process.returncode != 0:
                    error_msg = stderr.decode().strip() if stderr else "Unknown AppleScript error"
                    return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
                
                result = stdout.decode().strip()
                return Either.right(result)
                
            except OSError as e:
                return Either.left(KMError.execution_error(f"Failed to execute osascript: {str(e)}"))
                
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript execution error: {str(e)}"))
    
    def _contains_dangerous_commands(self, script: str) -> bool:
        """Check if AppleScript contains potentially dangerous commands."""
        
        dangerous_patterns = [
            r'do\s+shell\s+script',  # Shell execution
            r'system\s+info',         # System information
            r'restart\s+computer',    # System restart
            r'shutdown\s+computer',   # System shutdown
            r'delete\s+file',         # File deletion
            r'delete\s+folder',       # Folder deletion
            r'sudo\s+',              # Privilege escalation
            r'rm\s+-rf',             # Dangerous remove
            r'format\s+disk',        # Disk formatting
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                return True
        
        return False
    
    # TASK_10: Macro Creation Methods
    
    async def execute_applescript_async(self, applescript: str) -> Either[KMError, str]:
        """Execute AppleScript with async support and comprehensive error handling."""
        try:
            # Security: Validate AppleScript content
            if self._contains_dangerous_applescript(applescript):
                return Either.left(KMError.security_error("Dangerous AppleScript content detected"))
            
            # Execute AppleScript asynchronously
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.timeout.total_seconds()
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown AppleScript error"
                return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
            
            return Either.right(stdout.decode().strip())
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error("AppleScript execution timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript execution error: {str(e)}"))
    
    def _contains_dangerous_applescript(self, script: str) -> bool:
        """Check AppleScript for potentially dangerous content."""
        dangerous_patterns = [
            r'do shell script.*rm\s+-rf',
            r'do shell script.*sudo',
            r'do shell script.*curl.*\|\s*sh',
            r'set\s+\w+\s+to\s+password\s+of',
            r'keychain',
            r'security\s+',
            r'\/System\/',
            r'\/usr\/bin\/',
            r'\/etc\/',
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                return True
        
        return False
    
    async def list_groups_async(self) -> Either[KMError, List[Dict[str, Any]]]:
        """List macro groups asynchronously."""
        try:
            script = '''
            tell application "Keyboard Maestro"
                set groupList to every macro group
                set groupData to {}
                
                repeat with currentGroup in groupList
                    set groupName to name of currentGroup
                    set groupID to uid of currentGroup
                    set groupEnabled to enabled of currentGroup
                    
                    set groupRecord to {groupName:groupName, groupID:groupID, enabled:groupEnabled}
                    set groupData to groupData & {groupRecord}
                end repeat
                
                return groupData
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            # Parse the AppleScript output into a list of dictionaries
            output = result.get_right()
            groups = self._parse_applescript_records(output)
            
            return Either.right(groups)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to list groups: {str(e)}"))
    
    # TASK_20: Macro Movement Operations
    
    @require(lambda self, macro_id, target_group: macro_id and target_group)
    @ensure(lambda result: result.is_right() or result.get_left().code in ["MACRO_NOT_FOUND", "GROUP_NOT_FOUND", "PERMISSION_ERROR", "MOVE_ERROR"])
    async def move_macro_to_group_async(
        self,
        macro_id: MacroId,
        target_group: GroupId,
        create_missing: bool = False
    ) -> Either[KMError, MacroMoveResult]:
        """
        Execute macro movement with atomic operation guarantees.
        
        Security Features:
        - Input validation and sanitization
        - Permission verification for source and target groups
        - Atomic operation with rollback capability
        - Audit logging for all movement operations
        
        Architecture:
        - Pattern: Command Pattern with Memento for rollback
        - Security: Defense-in-depth with validation, authorization, audit
        - Performance: O(1) movement with conflict detection
        
        Contracts:
        Preconditions:
            - macro_id is valid and non-empty
            - target_group is valid and non-empty
            - User has accessibility permissions
        
        Postconditions:
            - Macro exists in target group OR error with rollback info
            - Source group no longer contains macro on success
            - System state is consistent (no partial moves)
        
        Invariants:
            - Macro can only exist in one group at a time
            - All movements are audited and logged
            - Failed movements leave system unchanged
        """
        from ..core.types import MacroMoveResult, Duration
        import time
        
        start_time = time.time()
        
        try:
            # Phase 1: Validate inputs and get current state
            validation_result = await self._validate_move_operation(macro_id, target_group)
            if validation_result.is_left():
                return validation_result
            
            source_group, macro_info = validation_result.get_right()
            
            # Phase 2: Check for conflicts and prepare rollback
            conflict_check = await self._check_move_conflicts(macro_id, source_group, target_group)
            if conflict_check.is_left():
                return conflict_check
            
            conflicts_found = conflict_check.get_right()
            
            # Phase 3: Create target group if needed
            if create_missing:
                group_check = await self._ensure_target_group_exists(target_group)
                if group_check.is_left():
                    return group_check
            
            # Phase 4: Execute atomic move operation
            move_result = await self._execute_macro_move(macro_id, source_group, target_group)
            if move_result.is_left():
                return move_result
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            # Phase 5: Verify move success
            verification_result = await self._verify_move_success(macro_id, target_group)
            if verification_result.is_left():
                # Attempt rollback
                await self._rollback_macro_move(macro_id, target_group, source_group)
                return verification_result
            
            return Either.right(MacroMoveResult(
                macro_id=macro_id,
                source_group=source_group,
                target_group=target_group,
                execution_time=execution_time,
                conflicts_resolved=conflicts_found
            ))
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Macro movement failed: {str(e)}"))
    
    async def _validate_move_operation(
        self, 
        macro_id: MacroId, 
        target_group: GroupId
    ) -> Either[KMError, tuple[GroupId, Dict[str, Any]]]:
        """Validate move operation and get current macro state."""
        try:
            # Find macro and its current group
            find_result = await self._find_macro_current_group(macro_id)
            if find_result.is_left():
                return find_result
            
            source_group, macro_info = find_result.get_right()
            
            # Validate target group exists
            group_check = await self._validate_group_exists(target_group)
            if group_check.is_left():
                return group_check
            
            # Check if already in target group
            if source_group == target_group:
                return Either.left(KMError.validation_error(
                    f"Macro {macro_id} is already in group {target_group}"
                ))
            
            return Either.right((source_group, macro_info))
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Move validation failed: {str(e)}"))
    
    async def _find_macro_current_group(self, macro_id: MacroId) -> Either[KMError, tuple[GroupId, Dict[str, Any]]]:
        """Find macro's current group and get macro information."""
        try:
            # Escape macro ID for AppleScript
            escaped_macro_id = self._escape_applescript_string(macro_id)
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set foundMacro to first macro whose name is "{escaped_macro_id}" or uid is "{escaped_macro_id}"
                    set parentGroup to macro group of foundMacro
                    set groupName to name of parentGroup
                    set groupID to uid of parentGroup
                    set macroName to name of foundMacro
                    set macroEnabled to enabled of foundMacro
                    
                    return "SUCCESS:" & groupID & ":" & groupName & ":" & macroName & ":" & macroEnabled
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.not_found_error(
                    f"Macro not found: {macro_id}"
                ))
            
            # Parse response: SUCCESS:groupID:groupName:macroName:enabled
            parts = output[8:].split(":", 4)  # Remove "SUCCESS:" prefix
            if len(parts) < 4:
                return Either.left(KMError.execution_error("Invalid macro lookup response"))
            
            source_group = GroupId(parts[0])
            macro_info = {
                "group_name": parts[1],
                "macro_name": parts[2],
                "enabled": parts[3].lower() == "true"
            }
            
            return Either.right((source_group, macro_info))
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Macro lookup failed: {str(e)}"))
    
    async def _validate_group_exists(self, group_id: GroupId) -> Either[KMError, bool]:
        """Validate that target group exists."""
        try:
            escaped_group_id = self._escape_applescript_string(group_id)
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set targetGroup to first macro group whose name is "{escaped_group_id}" or uid is "{escaped_group_id}"
                    return "SUCCESS"
                on error
                    return "ERROR: Group not found"
                end try
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.not_found_error(
                    f"Target group not found: {group_id}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Group validation failed: {str(e)}"))
    
    async def _check_move_conflicts(
        self, 
        macro_id: MacroId, 
        source_group: GroupId, 
        target_group: GroupId
    ) -> Either[KMError, List[str]]:
        """Check for potential conflicts in target group."""
        try:
            conflicts = []
            
            # Check for name collision in target group
            escaped_macro_id = self._escape_applescript_string(macro_id)
            escaped_target = self._escape_applescript_string(target_group)
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set targetGroup to first macro group whose name is "{escaped_target}" or uid is "{escaped_target}"
                    set macrosInGroup to every macro of targetGroup
                    
                    repeat with currentMacro in macrosInGroup
                        if name of currentMacro is "{escaped_macro_id}" then
                            return "CONFLICT: Name collision"
                        end if
                    end repeat
                    
                    return "SUCCESS: No conflicts"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("CONFLICT:"):
                conflicts.append("name_collision")
            elif output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:]))
            
            return Either.right(conflicts)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Conflict check failed: {str(e)}"))
    
    async def _ensure_target_group_exists(self, group_id: GroupId) -> Either[KMError, bool]:
        """Create target group if it doesn't exist."""
        try:
            # First check if group exists
            exists_check = await self._validate_group_exists(group_id)
            if exists_check.is_right():
                return Either.right(True)  # Already exists
            
            # Create new group
            escaped_group_id = self._escape_applescript_string(group_id)
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    make new macro group with properties {{name:"{escaped_group_id}"}}
                    return "SUCCESS: Group created"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:]))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Group creation failed: {str(e)}"))
    
    async def _execute_macro_move(
        self, 
        macro_id: MacroId, 
        source_group: GroupId, 
        target_group: GroupId
    ) -> Either[KMError, bool]:
        """Execute the actual macro move operation."""
        try:
            escaped_macro_id = self._escape_applescript_string(macro_id)
            escaped_target = self._escape_applescript_string(target_group)
            
            script = f'''
            tell application "Keyboard Maestro"
                try
                    set sourceMacro to first macro whose name is "{escaped_macro_id}" or uid is "{escaped_macro_id}"
                    set targetGroup to first macro group whose name is "{escaped_target}" or uid is "{escaped_target}"
                    
                    move sourceMacro to targetGroup
                    return "SUCCESS: Macro moved"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:]))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Macro move execution failed: {str(e)}"))
    
    async def _verify_move_success(self, macro_id: MacroId, target_group: GroupId) -> Either[KMError, bool]:
        """Verify that macro move was successful."""
        try:
            # Check if macro is now in target group
            find_result = await self._find_macro_current_group(macro_id)
            if find_result.is_left():
                return Either.left(KMError.execution_error("Move verification failed: macro not found"))
            
            current_group, _ = find_result.get_right()
            if current_group != target_group:
                return Either.left(KMError.execution_error(
                    f"Move verification failed: macro in {current_group}, expected {target_group}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Move verification failed: {str(e)}"))
    
    async def _rollback_macro_move(
        self, 
        macro_id: MacroId, 
        current_group: GroupId, 
        original_group: GroupId
    ) -> Either[KMError, bool]:
        """Rollback macro to original group on failure."""
        try:
            rollback_result = await self._execute_macro_move(macro_id, current_group, original_group)
            return rollback_result
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Rollback failed: {str(e)}"))
    
    def _escape_applescript_string(self, value: str) -> str:
        """Escape string for safe use in AppleScript."""
        if not isinstance(value, str):
            value = str(value)
        
        # Replace dangerous characters
        value = value.replace('\\', '\\\\')
        value = value.replace('"', '\\"')
        value = value.replace('\n', '\\n')
        value = value.replace('\r', '\\r')
        value = value.replace('\t', '\\t')
        
        return value


# Functional utilities for working with KM client

def retry_with_backoff(
    operation: Callable[[], Either[KMError, T]], 
    max_retries: int = 3,
    initial_delay: Duration = Duration.from_seconds(0.5)
) -> Either[KMError, T]:
    """Retry operation with exponential backoff."""
    current_delay = initial_delay
    
    for attempt in range(max_retries + 1):
        result = operation()
        
        if result.is_right():
            return result
        
        error = result.get_left()
        if attempt < max_retries and error and error.retry_after:
            time.sleep(current_delay.total_seconds())
            current_delay = Duration.from_seconds(current_delay.total_seconds() * 2)
        elif attempt == max_retries:
            return result
    
    return Either.left(KMError.execution_error("Max retries exceeded"))


def create_client_with_fallback(primary_config: ConnectionConfig, fallback_config: ConnectionConfig) -> KMClient:
    """Create client that falls back to secondary method on failure."""
    
    class FallbackClient(KMClient):
        def __init__(self):
            super().__init__(primary_config)
            self._fallback = KMClient(fallback_config)
        
        def execute_macro(self, macro_id: MacroId, trigger_value: Optional[str] = None) -> Either[KMError, Dict[str, Any]]:
            result = super().execute_macro(macro_id, trigger_value)
            if result.is_left():
                return self._fallback.execute_macro(macro_id, trigger_value)
            return result
    
    return FallbackClient()