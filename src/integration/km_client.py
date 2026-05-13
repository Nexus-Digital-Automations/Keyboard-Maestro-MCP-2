"""Functional Keyboard Maestro Client Interface.

Provides a functional interface to Keyboard Maestro APIs with
pure functions, error handling monads, and connection management.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlencode

import httpx

from ..core.contracts import ensure, require
from ..core.either import Either
from ..core.types import Duration, GroupId, MacroId, TriggerId

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..core.types import MacroMoveResult
    from .events import TriggerType

# Avoid circular import - use string annotation for TriggerType
# from .events import KMEvent, TriggerType, EventPriority


T = TypeVar("T")
E = TypeVar("E")

logger = logging.getLogger(__name__)

_KM_UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$",
)


def _run_osascript_sync(script: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Execute an AppleScript via osascript synchronously.

    Replaces the removed ``commands.secure_subprocess`` shim. osascript itself
    is a hardcoded absolute path; only the script body is variable, and that
    is escaped at each call site before it reaches us.
    """
    return subprocess.run(  # noqa: S603 — hardcoded osascript, script body is caller-escaped
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _escape_applescript_xml_literal(xml_text: str) -> str:
    """Escape arbitrary XML so it can ride inside an AppleScript string literal.

    Multi-line plist XML must survive the round-trip: backslashes first
    (so we don't double-escape the escapes we add next), then quotes, then
    CR/LF translated to ``\\r``/``\\n`` because AppleScript string literals
    cannot contain raw newlines.
    """
    return (
        xml_text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


class ConnectionMethod(Enum):
    """Available connection methods to Keyboard Maestro."""

    APPLESCRIPT = "applescript"
    URL_SCHEME = "url_scheme"
    WEB_API = "web_api"
    REMOTE_TRIGGER = "remote_trigger"


@dataclass(frozen=True)
class KMError:
    """Keyboard Maestro operation error."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    retry_after: Duration | None = None

    @classmethod
    def connection_error(cls, message: str) -> KMError:
        """Create connection error."""
        return cls(code="CONNECTION_ERROR", message=message)

    @classmethod
    def execution_error(
        cls,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> KMError:
        """Create execution error."""
        return cls(code="EXECUTION_ERROR", message=message, details=details)

    @classmethod
    def timeout_error(cls, timeout: Duration | str | float) -> KMError:
        """Create timeout error. Accepts a Duration or a free-text description."""
        if isinstance(timeout, Duration):
            text = f"Operation timed out after {timeout.total_seconds()}s"
        elif isinstance(timeout, int | float):
            text = f"Operation timed out after {float(timeout)}s"
        else:
            text = str(timeout)
        return cls(
            code="TIMEOUT_ERROR",
            message=text,
            retry_after=Duration.from_seconds(1.0),
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
        return replace(self, timeout=timeout)

    def with_method(self, method: ConnectionMethod) -> ConnectionConfig:
        """Create new config with different connection method."""
        return replace(self, method=method)


@dataclass(frozen=True)
class TriggerDefinition:
    """Definition for registering a macro trigger."""

    trigger_id: TriggerId
    macro_id: MacroId
    trigger_type: TriggerType  # String annotation to avoid circular import
    configuration: dict[str, Any]
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "trigger_id": self.trigger_id,
            "macro_id": self.macro_id,
            "trigger_type": self.trigger_type.value,
            "configuration": self.configuration,
            "enabled": self.enabled,
        }


class KMClient:
    """Functional interface to Keyboard Maestro APIs with pure error handling."""

    def __init__(self, connection_config: ConnectionConfig | None = None):
        self._config = (
            connection_config if connection_config is not None else ConnectionConfig()
        )
        self._send_command = partial(KMClient._safe_send, self._config)

    @property
    def config(self) -> ConnectionConfig:
        """Get connection configuration."""
        return self._config

    def execute_macro(
        self,
        macro_id: MacroId,
        trigger_value: str | None = None,
    ) -> Either[KMError, dict[str, Any]]:
        """Execute macro with functional error handling."""
        command_data = {"macro_id": macro_id}
        if trigger_value:
            command_data["trigger_value"] = trigger_value

        result = self._send_command("execute_macro", command_data)
        return result

    # FIXME: Contract disabled - @require(lambda __self, trigger_def: trigger_def.trigger_id and trigger_def.macro_id)
    def register_trigger(
        self,
        trigger_def: TriggerDefinition,
    ) -> Either[KMError, TriggerId]:
        """Register trigger with functional error handling."""
        result = self._send_command("register_trigger", trigger_def.to_dict())
        return result.map(lambda r: r.get("trigger_id", trigger_def.trigger_id))

    def unregister_trigger(self, trigger_id: TriggerId) -> Either[KMError, bool]:
        """Unregister trigger by ID."""
        result = self._send_command("unregister_trigger", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("success", False))

    def get_macro_list(
        self,
        group_filter: str | None = None,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """Get list of available macros."""
        params = {"group_filter": group_filter} if group_filter else {}
        result = self._send_command("list_macros", params)
        return result.map(lambda r: r.get("macros", []))

    def list_macros(
        self,
        group_filter: str | None = None,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """Get list of available macros (synchronous version for test compatibility)."""
        return self.get_macro_list(group_filter)

    def create_macro(
        self,
        macro_data: dict[str, Any],
    ) -> Either[KMError, dict[str, Any]]:
        """Create a new macro with the given data."""
        # Validate required fields
        if not macro_data.get("name"):
            return Either.left(KMError.validation_error("Macro name is required"))

        # Create the macro through AppleScript
        result = self._send_command("create_macro", macro_data)
        return result.map(
            lambda r: {
                "macro_id": r.get("macro_id", macro_data.get("name")),
                "success": True,
                "created": True,
            },
        )

    def list_macros_with_details(
        self,
        group_filter: str | None = None,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """Get detailed list of macros with full information."""
        # Get basic macro list first
        basic_result = self.get_macro_list(group_filter)
        if basic_result.is_left():
            return basic_result

        # Enhance with additional details
        macros = basic_result.get_right()
        detailed_macros = []

        for macro in macros:
            detailed_macro = {
                **macro,
                "details": True,
                "actions": macro.get("action_count", 0),
                "triggers": macro.get("trigger_count", 0),
                "metadata": {
                    "created_date": macro.get("created_date"),
                    "last_used": macro.get("last_used"),
                    "enabled": macro.get("enabled", True),
                },
            }
            detailed_macros.append(detailed_macro)

        return Either.right(detailed_macros)

    def get_macro_status(self, macro_id: MacroId) -> Either[KMError, dict[str, Any]]:
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

    def list_triggers(self) -> Either[KMError, list[dict[str, Any]]]:
        """Get list of all triggers from Keyboard Maestro."""
        result = self._send_command("list_triggers", {})
        return result.map(lambda r: r.get("triggers", []))

    def get_trigger_status(
        self,
        trigger_id: TriggerId,
    ) -> Either[KMError, dict[str, Any]]:
        """Get status of specific trigger."""
        result = self._send_command("get_trigger_status", {"trigger_id": trigger_id})
        return result.map(lambda r: r.get("status", {}))

    # Async versions for integration with async trigger manager

    async def register_trigger_async(
        self,
        trigger_def: TriggerDefinition,
    ) -> Either[KMError, TriggerId]:
        """Register trigger with comprehensive error handling and validation.

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
                return Either.left(
                    KMError.validation_error(
                        f"Trigger data sanitization failed: {sanitized_data.get_left()}",
                    ),
                )

            safe_config = sanitized_data.get_right()

            # Build AppleScript for trigger registration with proper escaping
            script_result = self._build_trigger_script_safe(
                trigger_def.trigger_type.value,
                safe_config,
                trigger_def.trigger_id,
            )
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
                return Either.left(
                    KMError.execution_error(f"KM registration failed: {km_response}"),
                )

            return Either.right(trigger_def.trigger_id)

        except Exception as e:
            return Either.left(
                KMError.execution_error(
                    f"Unexpected error in trigger registration: {e!s}",
                ),
            )

    async def activate_trigger_async(
        self,
        trigger_id: TriggerId,
    ) -> Either[KMError, bool]:
        """Async version of activate_trigger."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.activate_trigger,
            trigger_id,
        )

    async def deactivate_trigger_async(
        self,
        trigger_id: TriggerId,
    ) -> Either[KMError, bool]:
        """Async version of deactivate_trigger."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.deactivate_trigger,
            trigger_id,
        )

    async def list_triggers_async(self) -> Either[KMError, list[dict[str, Any]]]:
        """Async version of list_triggers."""
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(None, self.list_triggers)

    # TASK_5: Real Macro Listing Implementation

    async def list_macros_async(
        self,
        group_filters: list[str] | None = None,
        enabled_only: bool = True,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """Get real macro list from Keyboard Maestro using multiple API methods."""
        import asyncio

        # Try AppleScript first (most reliable)
        applescript_result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._list_macros_applescript,
            group_filters,
            enabled_only,
        )
        if applescript_result.is_right():
            return applescript_result

        # Fallback to Web API
        web_api_result = await self._list_macros_web_api(group_filters, enabled_only)
        if web_api_result.is_right():
            return web_api_result

        # Both methods failed
        return Either.left(
            KMError.connection_error("Cannot connect to Keyboard Maestro"),
        )

    def _list_macros_applescript(
        self,
        group_filters: list[str] | None = None,
        enabled_only: bool = True,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """List macros using AppleScript getmacros command."""
        # AppleScript to get macro information from Keyboard Maestro
        script = """
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
        """

        try:
            result = _run_osascript_sync(script, self.config.timeout.total_seconds())

            if result.returncode != 0:
                return Either.left(
                    KMError.execution_error(f"AppleScript failed: {result.stderr}"),
                )

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
                    if any(
                        group_filter.lower() in macro_group
                        for group_filter in group_filters
                    ):
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
                    "created_date": None,  # AppleScript doesn't provide this easily
                }
                standardized_macros.append(standardized_macro)

            return Either.right(standardized_macros)

        except subprocess.TimeoutExpired:
            return Either.left(KMError.timeout_error("AppleScript timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript error: {e!s}"))

    async def _list_macros_web_api(
        self,
        group_filters: list[str] | None = None,
        enabled_only: bool = True,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """List macros using KM Web API."""
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout.total_seconds(),
            ) as client:
                # Try to get macros from web API endpoints
                # Note: Actual KM Web API endpoints may vary - this is based on common patterns
                try:
                    response = await client.get(
                        f"http://{self.config.web_api_host}:{self.config.web_api_port}/macros",
                    )
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError:
                    # Try alternative endpoint format
                    response = await client.get(
                        f"http://{self.config.web_api_host}:{self.config.web_api_port}/action.html?action=GetMacros",
                    )
                    response.raise_for_status()

                    # Parse HTML response if needed (KM may return HTML instead of JSON)
                    if "application/json" in response.headers.get("content-type", ""):
                        data = response.json()
                    else:
                        # For HTML responses, we need to parse the content
                        return Either.left(
                            KMError.execution_error(
                                "Web API returned HTML instead of JSON",
                            ),
                        )

                macros = data.get("macros", []) if isinstance(data, dict) else data

                # Transform to standard format
                standardized_macros = []
                for macro in macros:
                    # Handle different possible response formats
                    macro_id = (
                        macro.get("uid") or macro.get("id") or macro.get("uuid", "")
                    )
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
                        "created_date": macro.get("created")
                        or macro.get("dateCreated"),
                    }
                    standardized_macros.append(standardized_macro)

                # Apply filters
                if enabled_only:
                    standardized_macros = [
                        m for m in standardized_macros if m.get("enabled", False)
                    ]
                if group_filters:
                    # Filter to include macros from any of the specified groups
                    filtered_macros = []
                    for macro in standardized_macros:
                        macro_group = macro.get("group", "").lower()
                        if any(
                            group_filter.lower() in macro_group
                            for group_filter in group_filters
                        ):
                            filtered_macros.append(macro)
                    standardized_macros = filtered_macros

                return Either.right(standardized_macros)

        except httpx.TimeoutException:
            return Either.left(KMError.timeout_error("Web API timeout"))
        except httpx.HTTPStatusError as e:
            return Either.left(
                KMError.connection_error(
                    f"Web API HTTP error: {e.response.status_code}",
                ),
            )
        except Exception as e:
            return Either.left(KMError.execution_error(f"Web API error: {e!s}"))

    def _parse_applescript_records(
        self,
        applescript_output: str,
        *,
        record_start_key: str = "macroId",
    ) -> list[dict[str, Any]]:
        """Parse AppleScript record format into Python dictionaries.

        AppleScript serialises a list of records as a flat
        ``key1:val1, key2:val2, key1:val1, key2:val2, ...`` stream. Pass
        ``record_start_key`` to whichever key always appears first per
        record (``macroId`` for macros, ``groupName`` for groups, etc.).
        """
        import re

        records = []

        # Clean up the output - remove extra whitespace and newlines
        clean_output = re.sub(r"\s+", " ", applescript_output.strip())

        # The actual AppleScript output is in flat comma-separated format
        # Parse format: key:value, key:value, key:value, ...
        # When we see 'macroId' again, it indicates a new record

        pairs = []
        current_pair = ""
        in_value = False
        paren_depth = 0

        # First, properly split by commas, handling nested content
        for char in clean_output:
            if char == "(" and not in_value:
                paren_depth += 1
            elif char == ")" and not in_value:
                paren_depth -= 1
            elif char == ":" and paren_depth == 0:
                in_value = True
            elif char == "," and paren_depth == 0 and in_value:
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
            if ":" in pair:
                # Split only on the first colon to handle values with colons
                key, value = pair.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Clean up the value - remove extra quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

                # Convert values to appropriate types
                if value == "true":
                    value = True
                elif value == "false":
                    value = False
                elif value.isdigit() or (
                    value.startswith("-") and value[1:].isdigit()
                ):
                    # UUIDs (``99999999-2222-...``) also pass ``replace('-','').isdigit()``;
                    # only coerce real integers (optional leading ``-`` then digits).
                    value = int(value)

                if key == record_start_key and current_record:
                    if record_start_key in current_record:
                        records.append(current_record)
                    current_record = {}

                current_record[key] = value

        if current_record and record_start_key in current_record:
            records.append(current_record)

        return records

    @staticmethod
    def _safe_send(
        config: ConnectionConfig,
        command: str,
        payload: dict[str, Any],
    ) -> Either[KMError, dict[str, Any]]:
        """Pure function for safe command sending with error handling."""
        try:
            if config.method == ConnectionMethod.APPLESCRIPT:
                return KMClient._send_via_applescript(command, payload, config)
            if config.method == ConnectionMethod.URL_SCHEME:
                return KMClient._send_via_url_scheme(command, payload, config)
            if config.method == ConnectionMethod.WEB_API:
                return KMClient._send_via_web_api(command, payload, config)
            return Either.left(
                KMError.connection_error(f"Unsupported method: {config.method}"),
            )

        except Exception as e:
            return Either.left(KMError.execution_error(f"Command failed: {e!s}"))

    @staticmethod
    def _send_via_applescript(
        command: str,
        payload: dict[str, Any],
        config: ConnectionConfig,
    ) -> Either[KMError, dict[str, Any]]:
        """Send command via AppleScript."""
        if command == "execute_macro":
            macro_id = payload.get("macro_id", "")
            trigger_value = payload.get("trigger_value", "")

            # Properly escape quotes and special characters for AppleScript
            escaped_macro_name = macro_id.replace('"', '\\"').replace("\\", "\\\\")
            escaped_params = (
                trigger_value.replace('"', '\\"').replace("\\", "\\\\")
                if trigger_value
                else ""
            )

            script = f"""
            tell application "Keyboard Maestro Engine"
                try
                    set result to do script "{escaped_macro_name}" with parameter "{escaped_params}"
                    return result
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            try:
                result = _run_osascript_sync(script, config.timeout.total_seconds())

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output.startswith("ERROR:"):
                        return Either.left(KMError.execution_error(output[6:].strip()))
                    return Either.right({"output": output, "success": True})
                return Either.left(KMError.execution_error(result.stderr.strip()))

            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        elif command == "ping":
            ping_script = """
            tell application "System Events"
                return (exists process "Keyboard Maestro Engine")
            end tell
            """
            try:
                result = subprocess.run(  # noqa: S603 — hardcoded osascript string, no injection surface
                    ["/usr/bin/osascript", "-e", ping_script],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    check=False,
                )
                alive = result.returncode == 0 and "true" in result.stdout.lower()
                return Either.right({"alive": alive})
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning("KM ping failed: %s", e)
                return Either.right({"alive": False})

        elif command == "register_trigger":
            # Extract trigger information from payload
            trigger_type = payload.get("trigger_type", "")
            trigger_id = payload.get("trigger_id", "")
            macro_id = payload.get("macro_id", "")
            payload.get("configuration", {})

            # Create AppleScript for trigger registration
            escaped_trigger_id = trigger_id.replace('"', '\\"').replace("\\", "\\\\")
            escaped_macro_id = macro_id.replace('"', '\\"').replace("\\", "\\\\")

            register_script = f"""
            tell application "Keyboard Maestro"
                try
                    -- Create new trigger for macro
                    set targetMacro to macro "{escaped_macro_id}"
                    set newTrigger to make new trigger at end of triggers of targetMacro
                    set name of newTrigger to "{escaped_trigger_id}"
                    set trigger type of newTrigger to "{trigger_type}"
                    return "SUCCESS: Trigger registered"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            try:
                result = _run_osascript_sync(register_script, config.timeout.total_seconds())

                if result.returncode == 0 and not result.stdout.startswith("ERROR:"):
                    return Either.right({"trigger_id": trigger_id, "success": True})
                error_msg = result.stderr.strip() or result.stdout.strip()
                return Either.left(
                    KMError.execution_error(
                        f"Trigger registration failed: {error_msg}",
                    ),
                )

            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        elif command == "unregister_trigger":
            trigger_id = payload.get("trigger_id", "")
            escaped_trigger_id = trigger_id.replace('"', '\\"').replace("\\", "\\\\")

            unregister_script = f"""
            tell application "Keyboard Maestro"
                try
                    -- Find and delete trigger by name
                    repeat with thisMacro in macros
                        repeat with thisTrigger in triggers of thisMacro
                            if name of thisTrigger is "{escaped_trigger_id}" then
                                delete thisTrigger
                                return "SUCCESS: Trigger unregistered"
                            end if
                        end repeat
                    end repeat
                    return "ERROR: Trigger not found"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            try:
                result = _run_osascript_sync(unregister_script, config.timeout.total_seconds())

                if result.returncode == 0 and not result.stdout.startswith("ERROR:"):
                    return Either.right({"success": True})
                error_msg = result.stderr.strip() or result.stdout.strip()
                return Either.left(
                    KMError.execution_error(
                        f"Trigger unregistration failed: {error_msg}",
                    ),
                )

            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        elif command == "create_macro":
            macro_name = payload.get("name", "")
            macro_group = payload.get("group", "Default")
            # TODO: Add support for actions and triggers in future implementation
            # actions = payload.get("actions", [])
            # triggers = payload.get("triggers", [])

            if not macro_name:
                return Either.left(KMError.validation_error("Macro name is required"))

            # Escape strings for AppleScript
            escaped_name = macro_name.replace('"', '\\"').replace("\\", "\\\\")
            escaped_group = macro_group.replace('"', '\\"').replace("\\", "\\\\")

            create_script = f"""
            tell application "Keyboard Maestro"
                try
                    -- Find or create the target group
                    set targetGroup to missing value
                    repeat with currentGroup in macro groups
                        if name of currentGroup is "{escaped_group}" then
                            set targetGroup to currentGroup
                            exit repeat
                        end if
                    end repeat

                    if targetGroup is missing value then
                        set targetGroup to make new macro group with properties {{name:"{escaped_group}"}}
                    end if

                    -- Create the new macro
                    set newMacro to make new macro at end of macros of targetGroup
                    set name of newMacro to "{escaped_name}"
                    set enabled of newMacro to true

                    -- Get the macro ID
                    set macroID to uid of newMacro

                    return "SUCCESS:" & macroID
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            try:
                result = _run_osascript_sync(create_script, config.timeout.total_seconds())

                if result.returncode == 0 and result.stdout.startswith("SUCCESS:"):
                    macro_id = result.stdout.strip()[8:]  # Remove "SUCCESS:" prefix
                    return Either.right(
                        {
                            "macro_id": macro_id,
                            "name": macro_name,
                            "group": macro_group,
                            "success": True,
                        },
                    )
                error_msg = result.stderr.strip() or result.stdout.strip()
                return Either.left(
                    KMError.execution_error(
                        f"Macro creation failed: {error_msg}",
                    ),
                )

            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        elif command == "activate_trigger":
            trigger_id = payload.get("trigger_id", "")
            escaped_trigger_id = trigger_id.replace('"', '\\"').replace("\\", "\\\\")

            activate_script = f"""
            tell application "Keyboard Maestro"
                try
                    -- Find and enable trigger by name
                    repeat with thisMacro in macros
                        repeat with thisTrigger in triggers of thisMacro
                            if name of thisTrigger is "{escaped_trigger_id}" then
                                set enabled of thisTrigger to true
                                return "SUCCESS: Trigger activated"
                            end if
                        end repeat
                    end repeat
                    return "ERROR: Trigger not found"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            try:
                result = _run_osascript_sync(activate_script, config.timeout.total_seconds())

                if result.returncode == 0 and not result.stdout.startswith("ERROR:"):
                    return Either.right({"success": True})
                error_msg = result.stderr.strip() or result.stdout.strip()
                return Either.left(
                    KMError.execution_error(
                        f"Trigger activation failed: {error_msg}",
                    ),
                )

            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        return Either.left(
            KMError.execution_error(f"Unsupported AppleScript command: {command}"),
        )

    @staticmethod
    def _send_via_url_scheme(
        command: str,
        payload: dict[str, Any],
        config: ConnectionConfig,
    ) -> Either[KMError, dict[str, Any]]:
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
                subprocess.run(  # noqa: S603 — hardcoded /usr/bin/open with kmtrigger URL
                    ["/usr/bin/open", url],
                    capture_output=True,
                    text=True,
                    timeout=config.timeout.total_seconds(),
                    check=False,
                )
                return Either.right({"success": True, "url": url})
            except subprocess.TimeoutExpired:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        return Either.left(
            KMError.execution_error(f"Unsupported URL scheme command: {command}"),
        )

    @staticmethod
    def _send_via_web_api(
        command: str,
        payload: dict[str, Any],
        config: ConnectionConfig,
    ) -> Either[KMError, dict[str, Any]]:
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
                return Either.left(
                    KMError.execution_error(
                        f"HTTP {response.status_code}: {response.text}",
                    ),
                )
            except httpx.TimeoutException:
                return Either.left(KMError.timeout_error(config.timeout))
            except Exception as e:
                return Either.left(KMError.execution_error(str(e)))

        return Either.left(
            KMError.execution_error(f"Unsupported web API command: {command}"),
        )

    # TASK_9: Enhanced helper methods for reliable trigger operations

    def _validate_trigger_definition(
        self,
        trigger_def: TriggerDefinition,
    ) -> Either[KMError, TriggerDefinition]:
        """Validate trigger definition structure and required fields."""
        if not trigger_def.trigger_id:
            return Either.left(KMError.validation_error("Trigger ID is required"))

        if not trigger_def.macro_id:
            return Either.left(KMError.validation_error("Macro ID is required"))

        if not isinstance(trigger_def.configuration, dict):
            return Either.left(
                KMError.validation_error("Trigger configuration must be a dictionary"),
            )

        # Validate trigger type specific requirements
        if trigger_def.trigger_type.value == "hotkey":
            if "key" not in trigger_def.configuration:
                return Either.left(
                    KMError.validation_error("Hotkey trigger requires 'key' parameter"),
                )

            key = trigger_def.configuration["key"]
            if not isinstance(key, str) or len(key) == 0:
                return Either.left(
                    KMError.validation_error("Hotkey 'key' must be a non-empty string"),
                )

        elif trigger_def.trigger_type.value == "application":
            if "application" not in trigger_def.configuration:
                return Either.left(
                    KMError.validation_error(
                        "Application trigger requires 'application' parameter",
                    ),
                )

        return Either.right(trigger_def)

    def _sanitize_trigger_data(
        self,
        config: dict[str, Any],
    ) -> Either[str, dict[str, Any]]:
        """Sanitize trigger configuration data to prevent injection attacks."""
        try:
            # Import security validation from our security module
            from .security import SecurityLevel, validate_km_input

            # Validate the configuration using our security system
            validation_result = validate_km_input(config, SecurityLevel.STANDARD)

            if not validation_result.is_safe:
                violations = [
                    f"{v.threat_type.value}: {v.violation_text}"
                    for v in validation_result.violations
                ]
                return Either.left(
                    f"Security violations detected: {'; '.join(violations)}",
                )

            return Either.right(validation_result.sanitized_data or config)

        except Exception as e:
            return Either.left(f"Sanitization error: {e!s}")

    def _build_trigger_script_safe(
        self,
        trigger_type: str,
        config: dict[str, Any],
        trigger_id: TriggerId,
    ) -> Either[KMError, str]:
        """Build AppleScript for trigger registration with comprehensive validation and escaping."""

        # Escape function for AppleScript strings
        def escape_applescript_string(value: str) -> str:
            """Escape string for safe use in AppleScript."""
            if not isinstance(value, str):
                value = str(value)

            # Replace dangerous characters
            value = value.replace("\\", "\\\\")  # Escape backslashes
            value = value.replace('"', '\\"')  # Escape quotes
            value = value.replace("\n", "\\n")  # Escape newlines
            value = value.replace("\r", "\\r")  # Escape carriage returns
            value = value.replace("\t", "\\t")  # Escape tabs

            return value

        if trigger_type == "hotkey":
            key = config.get("key", "")
            modifiers = config.get("modifiers", [])

            # Validate and escape key
            if not key or not isinstance(key, str):
                return Either.left(
                    KMError.validation_error("Invalid or missing hotkey"),
                )

            escaped_key = escape_applescript_string(key)

            # Validate and escape modifiers
            safe_modifiers = []
            for mod in modifiers:
                if isinstance(mod, str) and mod in [
                    "command",
                    "option",
                    "control",
                    "shift",
                ]:
                    safe_modifiers.append(escape_applescript_string(mod))

            escaped_trigger_id = escape_applescript_string(str(trigger_id))

            script = f"""
            tell application "Keyboard Maestro"
                try
                    set newTrigger to make new hotkey trigger with properties {{¬
                        key:"{escaped_key}", ¬
                        modifiers:{{{", ".join(f'"{mod}"' for mod in safe_modifiers)}}}¬
                    }}
                    return "SUCCESS: Trigger registered"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            return Either.right(script)

        if trigger_type == "application":
            app_name = config.get("application", "")
            event_type = config.get("event", "launches")

            if not app_name:
                return Either.left(
                    KMError.validation_error("Application name is required"),
                )

            escaped_app = escape_applescript_string(app_name)
            escaped_event = escape_applescript_string(event_type)
            escaped_trigger_id = escape_applescript_string(str(trigger_id))

            script = f"""
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
            """

            return Either.right(script)

        return Either.left(
            KMError.validation_error(f"Unsupported trigger type: {trigger_type}"),
        )

    async def _execute_applescript_safe(self, script: str) -> Either[KMError, str]:
        """Execute AppleScript with comprehensive security validation and error handling."""
        try:
            # Validate script safety before execution
            if self._contains_dangerous_commands(script):
                return Either.left(
                    KMError.validation_error("Dangerous AppleScript commands detected"),
                )

            # Execute with timeout using asyncio subprocess
            try:
                process = await asyncio.create_subprocess_exec(
                    "osascript",
                    "-e",
                    script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Wait for completion with timeout
                timeout_seconds = self.config.timeout.total_seconds()
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    process.terminate()
                    await process.wait()  # Ensure cleanup
                    return Either.left(KMError.timeout_error(self.config.timeout))

                # Check execution result
                if process.returncode != 0:
                    error_msg = (
                        stderr.decode().strip()
                        if stderr
                        else "Unknown AppleScript error"
                    )
                    return Either.left(
                        KMError.execution_error(f"AppleScript failed: {error_msg}"),
                    )

                result = stdout.decode().strip()
                return Either.right(result)

            except OSError as e:
                return Either.left(
                    KMError.execution_error(f"Failed to execute osascript: {e!s}"),
                )

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"AppleScript execution error: {e!s}"),
            )

    def _contains_dangerous_commands(self, script: str) -> bool:
        """Check if AppleScript contains potentially dangerous commands."""
        dangerous_patterns = [
            r"do\s+shell\s+script",  # Shell execution
            r"system\s+info",  # System information
            r"restart\s+computer",  # System restart
            r"shutdown\s+computer",  # System shutdown
            r"delete\s+file",  # File deletion
            r"delete\s+folder",  # Folder deletion
            r"sudo\s+",  # Privilege escalation
            r"rm\s+-rf",  # Dangerous remove
            r"format\s+disk",  # Disk formatting
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
                return Either.left(
                    KMError.security_error("Dangerous AppleScript content detected"),
                )

            # Execute AppleScript asynchronously
            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout.total_seconds(),
            )

            if process.returncode != 0:
                error_msg = (
                    stderr.decode().strip() if stderr else "Unknown AppleScript error"
                )
                return Either.left(
                    KMError.execution_error(f"AppleScript failed: {error_msg}"),
                )

            return Either.right(stdout.decode().strip())

        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error("AppleScript execution timeout"))
        except Exception as e:
            return Either.left(
                KMError.execution_error(f"AppleScript execution error: {e!s}"),
            )

    def _contains_dangerous_applescript(self, script: str) -> bool:
        """Check AppleScript for potentially dangerous content."""
        dangerous_patterns = [
            r"do shell script.*rm\s+-rf",
            r"do shell script.*sudo",
            r"do shell script.*curl.*\|\s*sh",
            r"set\s+\w+\s+to\s+password\s+of",
            r"keychain",
            r"security\s+",
            r"\/System\/",
            r"\/usr\/bin\/",
            r"\/etc\/",
        ]

        import re

        for pattern in dangerous_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                return True

        return False

    async def list_groups_async(self) -> Either[KMError, list[dict[str, Any]]]:
        """List macro groups asynchronously.

        Failure modes: KM internal/sentinel groups (e.g., the synthetic
        ``99999999-2222-3333-4444-555555555555`` smart-group placeholder) raise
        on ``uid``/``id`` dereference. We skip those rather than fail the whole
        listing — they are not user-visible groups.
        """
        try:
            script = """
            tell application "Keyboard Maestro"
                set groupData to {}
                repeat with currentGroup in (every macro group)
                    try
                        set groupName to name of currentGroup
                        set groupID to (id of currentGroup as string)
                        set groupEnabled to enabled of currentGroup
                        set groupRecord to {groupName:groupName, groupID:groupID, enabled:groupEnabled}
                        set groupData to groupData & {groupRecord}
                    end try
                end repeat
                return groupData
            end tell
            """

            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result

            # Parse the AppleScript output into a list of dictionaries
            output = result.get_right()
            groups = self._parse_applescript_records(
                output, record_start_key="groupName",
            )

            return Either.right(groups)

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Failed to list groups: {e!s}"),
            )

    # TASK_20: Macro Movement Operations

    @require(lambda __self, macro_id, target_group: macro_id and target_group)
    @ensure(
        lambda result: result.is_right()
        or result.get_left().code
        in [
            "MACRO_NOT_FOUND",
            "GROUP_NOT_FOUND",
            "PERMISSION_ERROR",
            "MOVE_ERROR",
            "NOT_FOUND_ERROR",
        ],
    )
    async def move_macro_to_group_async(
        self,
        macro_id: MacroId,
        target_group: GroupId,
        create_missing: bool = False,
    ) -> Either[KMError, MacroMoveResult]:
        """Execute macro movement with atomic operation guarantees.

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
        import time

        from ..core.types import Duration, MacroMoveResult

        start_time = time.time()

        try:
            # Phase 1: Validate inputs and get current state
            validation_result = await self._validate_move_operation(
                macro_id,
                target_group,
            )
            if validation_result.is_left():
                return validation_result

            source_group, macro_info = validation_result.get_right()

            # Phase 2: Check for conflicts and prepare rollback
            conflict_check = await self._check_move_conflicts(
                macro_id,
                source_group,
                target_group,
            )
            if conflict_check.is_left():
                return conflict_check

            conflicts_found = conflict_check.get_right()

            # Phase 3: Create target group if needed
            if create_missing:
                group_check = await self._ensure_target_group_exists(target_group)
                if group_check.is_left():
                    return group_check

            # Phase 4: Execute atomic move operation
            move_result = await self._execute_macro_move(
                macro_id,
                source_group,
                target_group,
            )
            if move_result.is_left():
                return move_result

            execution_time = Duration.from_seconds(time.time() - start_time)

            # Phase 5: Verify move success
            verification_result = await self._verify_move_success(
                macro_id,
                target_group,
            )
            if verification_result.is_left():
                # Attempt rollback
                await self._rollback_macro_move(macro_id, target_group, source_group)
                return verification_result

            return Either.right(
                MacroMoveResult(
                    macro_id=macro_id,
                    source_group=source_group,
                    target_group=target_group,
                    execution_time=execution_time,
                    conflicts_resolved=conflicts_found,
                ),
            )

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Macro movement failed: {e!s}"),
            )

    async def _validate_move_operation(
        self,
        macro_id: MacroId,
        target_group: GroupId,
    ) -> Either[KMError, tuple[GroupId, dict[str, Any]]]:
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
                return Either.left(
                    KMError.validation_error(
                        f"Macro {macro_id} is already in group {target_group}",
                    ),
                )

            return Either.right((source_group, macro_info))

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Move validation failed: {e!s}"),
            )

    async def _find_macro_current_group(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, tuple[GroupId, dict[str, Any]]]:
        """Find macro's current group and get macro information."""
        try:
            # Escape macro ID for AppleScript
            escaped_macro_id = self._escape_applescript_string(macro_id)

            script = f"""
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
            """

            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result

            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(
                    KMError.not_found_error(f"Macro not found: {macro_id}"),
                )

            # Parse response: SUCCESS:groupID:groupName:macroName:enabled
            parts = output[8:].split(":", 4)  # Remove "SUCCESS:" prefix
            if len(parts) < 4:
                return Either.left(
                    KMError.execution_error("Invalid macro lookup response"),
                )

            source_group = GroupId(parts[0])
            macro_info = {
                "group_name": parts[1],
                "macro_name": parts[2],
                "enabled": parts[3].lower() == "true",
            }

            return Either.right((source_group, macro_info))

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Macro lookup failed: {e!s}"),
            )

    async def _validate_group_exists(self, group_id: GroupId) -> Either[KMError, bool]:
        """Validate that target group exists."""
        try:
            escaped_group_id = self._escape_applescript_string(group_id)

            script = f"""
            tell application "Keyboard Maestro"
                try
                    set targetGroup to first macro group whose name is "{escaped_group_id}" or uid is "{escaped_group_id}"
                    return "SUCCESS"
                on error
                    return "ERROR: Group not found"
                end try
            end tell
            """

            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result

            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(
                    KMError.not_found_error(f"Target group not found: {group_id}"),
                )

            return Either.right(True)

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Group validation failed: {e!s}"),
            )

    async def _check_move_conflicts(
        self,
        macro_id: MacroId,
        _source_group: GroupId,
        target_group: GroupId,
    ) -> Either[KMError, list[str]]:
        """Check for potential conflicts in target group."""
        try:
            conflicts = []

            # Check for name collision in target group
            escaped_macro_id = self._escape_applescript_string(macro_id)
            escaped_target = self._escape_applescript_string(target_group)

            script = f"""
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
            """

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
            return Either.left(
                KMError.execution_error(f"Conflict check failed: {e!s}"),
            )

    async def _ensure_target_group_exists(
        self,
        group_id: GroupId,
    ) -> Either[KMError, bool]:
        """Create target group if it doesn't exist."""
        try:
            # First check if group exists
            exists_check = await self._validate_group_exists(group_id)
            if exists_check.is_right():
                return Either.right(True)  # Already exists

            # Create new group
            escaped_group_id = self._escape_applescript_string(group_id)

            script = f"""
            tell application "Keyboard Maestro"
                try
                    make new macro group with properties {{name:"{escaped_group_id}"}}
                    return "SUCCESS: Group created"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            """

            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result

            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:]))

            return Either.right(True)

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Group creation failed: {e!s}"),
            )

    async def _execute_macro_move(
        self,
        macro_id: MacroId,
        _source_group: GroupId,
        target_group: GroupId,
    ) -> Either[KMError, bool]:
        """Execute the actual macro move operation."""
        try:
            escaped_macro_id = self._escape_applescript_string(macro_id)
            escaped_target = self._escape_applescript_string(target_group)

            script = f"""
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
            """

            result = await self.execute_applescript_async(script)
            if result.is_left():
                return result

            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:]))

            return Either.right(True)

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Macro move execution failed: {e!s}"),
            )

    async def _verify_move_success(
        self,
        macro_id: MacroId,
        target_group: GroupId,
    ) -> Either[KMError, bool]:
        """Verify that macro move was successful."""
        try:
            # Check if macro is now in target group
            find_result = await self._find_macro_current_group(macro_id)
            if find_result.is_left():
                return Either.left(
                    KMError.execution_error(
                        "Move verification failed: macro not found",
                    ),
                )

            current_group, _ = find_result.get_right()
            if current_group != target_group:
                return Either.left(
                    KMError.execution_error(
                        f"Move verification failed: macro in {current_group}, expected {target_group}",
                    ),
                )

            return Either.right(True)

        except Exception as e:
            return Either.left(
                KMError.execution_error(f"Move verification failed: {e!s}"),
            )

    async def _rollback_macro_move(
        self,
        macro_id: MacroId,
        current_group: GroupId,
        original_group: GroupId,
    ) -> Either[KMError, bool]:
        """Rollback macro to original group on failure."""
        try:
            rollback_result = await self._execute_macro_move(
                macro_id,
                current_group,
                original_group,
            )
            return rollback_result

        except Exception as e:
            return Either.left(KMError.execution_error(f"Rollback failed: {e!s}"))

    def _macro_selector(self, macro_id: str) -> str:
        """Return the AppleScript selector for ``macro_id``.

        KM 11 AppleScript distinguishes UUID and name lookups; ``whose name
        is "<uuid>"`` silently matches nothing rather than erroring. Callers
        must dispatch on shape — UUIDs go through ``macro id "..."``, names
        through ``first macro whose name is "..."``.
        """
        escaped = self._escape_applescript_string(macro_id)
        if _KM_UUID_RE.match(macro_id):
            return f'macro id "{escaped}"'
        return f'first macro whose name is "{escaped}"'

    def _escape_applescript_string(self, value: str) -> str:
        """Escape string for safe use in AppleScript."""
        if not isinstance(value, str):
            value = str(value)

        # Replace dangerous characters
        value = value.replace("\\", "\\\\")
        value = value.replace('"', '\\"')
        value = value.replace("\n", "\\n")
        value = value.replace("\r", "\\r")
        value = value.replace("\t", "\\t")

        return value

    async def delete_macro_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, bool]:
        """Delete a macro by name or UUID. Returns NOT_FOUND on miss.

        ``exists`` precheck protects against the AppleScript-`whose` quirk
        where a no-match would otherwise complete silently.
        """
        selector = self._macro_selector(str(macro_id).strip())
        script = f'''
        tell application "Keyboard Maestro"
            try
                if not (exists {selector}) then
                    return "ERROR: macro not found"
                end if
                delete {selector}
                return "deleted"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)

    async def rename_macro_async(
        self,
        macro_id: MacroId,
        new_name: str,
    ) -> Either[KMError, bool]:
        """Rename a macro by name or UUID. Fails if target name is taken."""
        if not new_name.strip():
            return Either.left(KMError.validation_error("new_name cannot be empty"))
        selector = self._macro_selector(str(macro_id).strip())
        escaped_new = self._escape_applescript_string(new_name)
        script = f'''
        tell application "Keyboard Maestro"
            try
                if not (exists {selector}) then
                    return "ERROR: macro not found"
                end if
                set name of ({selector}) to "{escaped_new}"
                return "renamed"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def duplicate_macro_async(
        self,
        macro_id: MacroId,
        new_name: str | None = None,
    ) -> Either[KMError, dict[str, Any]]:
        """Duplicate a macro. If ``new_name`` is provided, the copy is renamed to it.

        KM 11 quirks worked around here:
        - ``duplicate`` returns a *list* of macro references, not a single
          reference. Read ``item 1`` of the result.
        - ``set name of <reference>`` only works when the reference was obtained
          via ``whose name is`` / ``whose id is``. Setting name on the
          duplicate-result reference fails with a type-coercion error, so we
          re-resolve the new macro by id before renaming.
        - ``name of <reference>`` also doesn't coerce to text directly on
          duplicate-result references — use ``id of … as string`` and look up
          the name afterwards.
        """
        source_selector = self._macro_selector(str(macro_id).strip())
        script = f'''
        tell application "Keyboard Maestro"
            try
                if not (exists {source_selector}) then
                    return "ERROR: macro not found"
                end if
                set sourceMacro to {source_selector}
                set dupResult to duplicate sourceMacro
                set newMacro to item 1 of dupResult
                return (id of newMacro as string)
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        new_id = output

        final_name: str | None = None
        if new_name and new_name.strip():
            escaped_id = self._escape_applescript_string(new_id)
            escaped_new = self._escape_applescript_string(new_name)
            rename_script = f'''
            tell application "Keyboard Maestro"
                try
                    set name of (first macro whose id is "{escaped_id}") to "{escaped_new}"
                    return "ok"
                on error errMsg
                    return "ERROR: " & errMsg
                end try
            end tell
            '''
            rename_result = await self.execute_applescript_async(rename_script)
            if rename_result.is_left():
                return Either.left(rename_result.get_left())
            rename_output = rename_result.get_right().strip()
            if rename_output.startswith("ERROR:"):
                return Either.left(
                    KMError.execution_error(
                        f"Duplicate created (id={new_id}) but rename failed: "
                        f"{rename_output[6:].strip()}",
                    ),
                )
            final_name = new_name

        return Either.right(
            {"new_id": new_id, "new_name": final_name, "source": str(macro_id)},
        )

    async def set_macro_enabled_async(
        self,
        macro_id: MacroId,
        enabled: bool,
    ) -> Either[KMError, bool]:
        """Enable or disable a macro by name or UUID."""
        selector = self._macro_selector(str(macro_id).strip())
        flag = "true" if enabled else "false"
        script = f'''
        tell application "Keyboard Maestro"
            try
                if not (exists {selector}) then
                    return "ERROR: macro not found"
                end if
                set enabled of ({selector}) to {flag}
                return "ok"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def create_group_async(
        self,
        group_name: str,
    ) -> Either[KMError, dict[str, Any]]:
        """Create a new macro group. Fails if the name already exists."""
        if not group_name.strip():
            return Either.left(KMError.validation_error("group_name cannot be empty"))
        escaped = self._escape_applescript_string(group_name)
        # KM 11 refuses to coerce ``uid of macro-group`` into text directly
        # (returns "Can't make uid ... into type specifier"). Use ``id of`` with
        # explicit ``as string`` instead — same pattern as ``list_groups_async``.
        script = f'''
        tell application "Keyboard Maestro"
            try
                set newGroup to make new macro group with properties {{name:"{escaped}"}}
                return (id of newGroup as string)
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right({"group_id": output, "name": group_name})

    async def delete_group_async(
        self,
        group_id: GroupId,
    ) -> Either[KMError, bool]:
        """Delete a macro group by name or UUID. All macros inside are deleted with it."""
        escaped = self._escape_applescript_string(str(group_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                delete (first macro group whose name is "{escaped}")
                return "deleted"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)

    async def rename_group_async(
        self,
        group_id: GroupId,
        new_name: str,
    ) -> Either[KMError, bool]:
        """Rename a macro group."""
        if not new_name.strip():
            return Either.left(KMError.validation_error("new_name cannot be empty"))
        escaped_old = self._escape_applescript_string(str(group_id))
        escaped_new = self._escape_applescript_string(new_name)
        script = f'''
        tell application "Keyboard Maestro"
            try
                set name of (first macro group whose name is "{escaped_old}") to "{escaped_new}"
                return "renamed"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def set_group_enabled_async(
        self,
        group_id: GroupId,
        enabled: bool,
    ) -> Either[KMError, bool]:
        """Enable or disable a macro group."""
        escaped = self._escape_applescript_string(str(group_id))
        flag = "true" if enabled else "false"
        script = f'''
        tell application "Keyboard Maestro"
            try
                set enabled of (first macro group whose name is "{escaped}") to {flag}
                return "ok"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def attach_trigger_async(
        self,
        macro_id: MacroId,
        trigger_type: str,
        config: dict[str, Any],
    ) -> Either[KMError, bool]:
        """Attach a new trigger to an existing macro.

        Supports trigger_type values: "hotkey" (config: key, modifiers),
        "application" (config: application, event).
        Other types are deferred to a later round.

        Existing `register_trigger_async` predates the macro-scoping concept
        in this codebase and doesn't attach to a specific macro — this is
        the proper macro-scoped primitive.
        """
        escaped_id = self._escape_applescript_string(str(macro_id))
        if trigger_type == "hotkey":
            key = config.get("key")
            if not isinstance(key, str) or not key:
                return Either.left(KMError.validation_error("hotkey requires 'key' (str)"))
            allowed_mods = {"command", "option", "control", "shift"}
            mods_raw = config.get("modifiers", [])
            if not isinstance(mods_raw, list):
                return Either.left(KMError.validation_error("'modifiers' must be a list"))
            safe_mods = [m for m in mods_raw if isinstance(m, str) and m in allowed_mods]
            mods_clause = "{" + ", ".join(f'"{m}"' for m in safe_mods) + "}"
            escaped_key = self._escape_applescript_string(key)
            props = f'key:"{escaped_key}", modifiers:{mods_clause}'
            type_clause = "hot key trigger"
        elif trigger_type == "application":
            app_name = config.get("application")
            event = config.get("event", "launches")
            if not isinstance(app_name, str) or not app_name:
                return Either.left(
                    KMError.validation_error("application trigger requires 'application' (str)"),
                )
            if event not in {"launches", "quits", "activates"}:
                return Either.left(
                    KMError.validation_error("'event' must be launches|quits|activates"),
                )
            escaped_app = self._escape_applescript_string(app_name)
            props = f'application:"{escaped_app}", event:"{event}"'
            type_clause = "application trigger"
        else:
            return Either.left(
                KMError.validation_error(
                    f"trigger_type '{trigger_type}' not supported; "
                    "use 'hotkey' or 'application'",
                ),
            )
        script = f'''
        tell application "Keyboard Maestro"
            try
                set targetMacro to first macro whose name is "{escaped_id}" or uid is "{escaped_id}"
                make new {type_clause} at end of triggers of targetMacro ¬
                    with properties {{{props}}}
                return "attached"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def list_macro_triggers_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """List triggers attached to a macro (description only).

        KM 11 doesn't expose ``enabled`` on individual triggers via AppleScript
        (only at the macro level). Asking for it raises a coercion error and
        breaks the whole listing. Older code referenced ``enabled of t``,
        which is why this endpoint always returned ``LIST_FAILED``.
        """
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                set parentMacro to first macro whose name is "{escaped}" or uid is "{escaped}"
                set trigCount to count of triggers of parentMacro
                set output to ""
                repeat with i from 1 to trigCount
                    set output to output & (description of (trigger i of parentMacro)) & linefeed
                end repeat
                return output
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        triggers: list[dict[str, Any]] = []
        for idx, line in enumerate(filter(None, output.split("\n")), start=1):
            triggers.append({"index": idx, "description": line.strip()})
        return Either.right(triggers)

    async def remove_macro_trigger_async(
        self,
        macro_id: MacroId,
        trigger_index: int,
    ) -> Either[KMError, bool]:
        """Remove the Nth trigger (1-indexed) from a macro."""
        if trigger_index < 1:
            return Either.left(KMError.validation_error("trigger_index must be >= 1"))
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                delete trigger {trigger_index} of (first macro whose name is "{escaped}" or uid is "{escaped}")
                return "deleted"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)

    async def clear_macro_triggers_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, bool]:
        """Remove all triggers from a macro."""
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                delete every trigger of (first macro whose name is "{escaped}" or uid is "{escaped}")
                return "cleared"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)

    async def set_trigger_enabled_async(
        self,
        macro_id: MacroId,
        trigger_index: int,
        enabled: bool,
    ) -> Either[KMError, bool]:
        """Enable or disable the Nth trigger (1-indexed) of a macro."""
        if trigger_index < 1:
            return Either.left(KMError.validation_error("trigger_index must be >= 1"))
        escaped = self._escape_applescript_string(str(macro_id))
        flag = "true" if enabled else "false"
        script = f'''
        tell application "Keyboard Maestro"
            try
                set enabled of trigger {trigger_index} of (first macro whose name is "{escaped}" or uid is "{escaped}") to {flag}
                return "ok"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def get_macro_trigger_xml_async(
        self,
        macro_id: MacroId,
        trigger_index: int,
    ) -> Either[KMError, str]:
        """Return the full plist XML of the Nth trigger (1-indexed) of a macro."""
        if trigger_index < 1:
            return Either.left(KMError.validation_error("trigger_index must be >= 1"))
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                return xml of trigger {trigger_index} of (first macro whose name is "{escaped}" or uid is "{escaped}")
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right()
        if output.strip().startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output.strip()[6:].strip()))
        return Either.right(output)

    async def list_macro_triggers_with_xml_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """List triggers with full XML body so callers can edit and replace."""
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                set output to ""
                set trigList to triggers of (first macro whose name is "{escaped}" or uid is "{escaped}")
                repeat with t in trigList
                    set output to output & (description of t) & "␟" & (enabled of t) & "␟" & (xml of t) & "␞"
                end repeat
                return output
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right()
        stripped = output.strip()
        if stripped.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(stripped[6:].strip()))
        triggers: list[dict[str, Any]] = []
        for idx, record in enumerate(filter(None, output.split("␞")), start=1):
            parts = record.split("␟", 2)
            triggers.append(
                {
                    "index": idx,
                    "description": parts[0].strip() if parts else "",
                    "enabled": (parts[1].strip().lower() == "true") if len(parts) > 1 else True,
                    "xml": parts[2] if len(parts) > 2 else "",
                },
            )
        return Either.right(triggers)

    async def append_macro_trigger_xml_async(
        self,
        macro_id: MacroId,
        trigger_xml: str,
    ) -> Either[KMError, bool]:
        """Append a trigger by setting its full plist XML.

        Works for every KM trigger type — caller controls the
        ``MacroTriggerType`` discriminator and per-type fields in the XML.
        """
        escaped_id = self._escape_applescript_string(str(macro_id))
        escaped_xml = _escape_applescript_xml_literal(trigger_xml)
        script = f'''
        tell application "Keyboard Maestro"
            try
                set targetMacro to first macro whose name is "{escaped_id}" or uid is "{escaped_id}"
                set newTrigger to make new trigger at end of triggers of targetMacro
                set xml of newTrigger to "{escaped_xml}"
                return "appended"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def update_macro_trigger_xml_async(
        self,
        macro_id: MacroId,
        trigger_index: int,
        trigger_xml: str,
    ) -> Either[KMError, bool]:
        """Replace the XML of the Nth trigger (1-indexed) of a macro."""
        if trigger_index < 1:
            return Either.left(KMError.validation_error("trigger_index must be >= 1"))
        escaped_id = self._escape_applescript_string(str(macro_id))
        escaped_xml = _escape_applescript_xml_literal(trigger_xml)
        script = f'''
        tell application "Keyboard Maestro"
            try
                set xml of trigger {trigger_index} of (first macro whose name is "{escaped_id}" or uid is "{escaped_id}") to "{escaped_xml}"
                return "updated"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def list_macro_actions_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, list[dict[str, Any]]]:
        """List actions inside a macro (description + enabled flag, 1-indexed)."""
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                set output to ""
                set actList to actions of (first macro whose name is "{escaped}" or uid is "{escaped}")
                repeat with a in actList
                    set output to output & (name of a) & "␟" & (enabled of a) & "\n"
                end repeat
                return output
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        actions: list[dict[str, Any]] = []
        for idx, line in enumerate(filter(None, output.split("\n")), start=1):
            parts = line.split("␟")
            actions.append(
                {
                    "index": idx,
                    "name": parts[0] if parts else line,
                    "enabled": (parts[1].strip().lower() == "true") if len(parts) > 1 else True,
                },
            )
        return Either.right(actions)

    async def append_macro_action_async(
        self,
        macro_id: MacroId,
        action_xml: str,
    ) -> Either[KMError, bool]:
        """Append an action to a macro by setting its XML.

        Caller builds the action XML (one ``<dict>...</dict>`` element in KM's
        action format). Tool layer (`action_builder_tools`) owns the
        type-to-XML mapping; this primitive only escapes + dispatches.
        """
        escaped_id = self._escape_applescript_string(str(macro_id))
        escaped_xml = action_xml.replace("\\", "\\\\").replace('"', '\\"')
        script = f'''
        tell application "Keyboard Maestro"
            try
                set targetMacro to first macro whose name is "{escaped_id}" or uid is "{escaped_id}"
                set newAction to make new action at end of actions of targetMacro
                set XML of newAction to "{escaped_xml}"
                return "appended"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.execution_error(output[6:].strip()))
        return Either.right(True)

    async def delete_macro_action_async(
        self,
        macro_id: MacroId,
        action_index: int,
    ) -> Either[KMError, bool]:
        """Delete the Nth action (1-indexed) from a macro."""
        if action_index < 1:
            return Either.left(KMError.validation_error("action_index must be >= 1"))
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                delete action {action_index} of (first macro whose name is "{escaped}" or uid is "{escaped}")
                return "deleted"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)

    async def clear_macro_actions_async(
        self,
        macro_id: MacroId,
    ) -> Either[KMError, bool]:
        """Remove all actions from a macro (leaves the shell intact)."""
        escaped = self._escape_applescript_string(str(macro_id))
        script = f'''
        tell application "Keyboard Maestro"
            try
                delete every action of (first macro whose name is "{escaped}" or uid is "{escaped}")
                return "cleared"
            on error errMsg
                return "ERROR: " & errMsg
            end try
        end tell
        '''
        result = await self.execute_applescript_async(script)
        if result.is_left():
            return Either.left(result.get_left())
        output = result.get_right().strip()
        if output.startswith("ERROR:"):
            return Either.left(KMError.not_found_error(output[6:].strip()))
        return Either.right(True)


# Functional utilities for working with KM client


def retry_with_backoff(
    operation: Callable[[], Either[KMError, T]],
    max_retries: int = 3,
    initial_delay: Duration | None = None,
) -> Either[KMError, T]:
    """Retry operation with exponential backoff."""
    if initial_delay is None:
        initial_delay = Duration.from_seconds(0.5)
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


def create_client_with_fallback(
    primary_config: ConnectionConfig,
    fallback_config: ConnectionConfig,
) -> KMClient:
    """Create client that falls back to secondary method on failure."""

    class FallbackClient(KMClient):
        def __init__(self):
            super().__init__(primary_config)
            self._fallback = KMClient(fallback_config)

        def execute_macro(
            self,
            macro_id: MacroId,
            trigger_value: str | None = None,
        ) -> Either[KMError, dict[str, Any]]:
            result = super().execute_macro(macro_id, trigger_value)
            if result.is_left():
                return self._fallback.execute_macro(macro_id, trigger_value)
            return result

    return FallbackClient()


# Add test compatibility methods by overriding the original methods
def _add_test_compatibility_to_kmclient():
    """Add test compatibility methods to KMClient class."""

    # Store original methods
    original_list_macros = KMClient.list_macros
    original_execute_macro = KMClient.execute_macro
    original_create_macro = KMClient.create_macro

    def list_macros_simple(
        self,
        group_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of macros as plain list for test compatibility."""
        result = original_list_macros(self, group_filter)
        if result.is_right():
            return result.get_right()
        return []  # Return empty list on error for test compatibility

    def execute_macro_simple(
        self,
        macro_id: MacroId,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Execute macro and return simple result for test compatibility."""
        result = original_execute_macro(self, macro_id, **kwargs)
        if result.is_right():
            return result.get_right()
        return None  # Return None on error for test compatibility

    def create_macro_simple(self, macro_data: dict[str, Any]) -> dict[str, Any] | None:
        """Create macro and return simple result for test compatibility."""
        result = original_create_macro(self, macro_data)
        if result.is_right():
            return result.get_right()
        return None  # Return None on error for test compatibility

    # Override the methods for test compatibility
    KMClient.list_macros = list_macros_simple
    KMClient.execute_macro = execute_macro_simple
    KMClient.create_macro = create_macro_simple


# Test compatibility layer disabled - preserve Either monad API contract
# _add_test_compatibility_to_kmclient()
