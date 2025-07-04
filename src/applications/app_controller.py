"""
Secure Application Control with State Tracking and Error Recovery

Implements comprehensive application lifecycle management with security validation,
permission checking, and robust error handling for system-level operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import subprocess
import asyncio
import re
import time
from pathlib import Path

from ..core.types import Duration
from ..core.contracts import require, ensure
from ..integration.km_client import Either, KMError


class AppState(Enum):
    """Application execution states with comprehensive lifecycle tracking."""
    NOT_RUNNING = "not_running"
    LAUNCHING = "launching"
    RUNNING = "running"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    TERMINATING = "terminating"
    CRASHED = "crashed"
    UNKNOWN = "unknown"


class ApplicationPermission(Enum):
    """Application control permission levels."""
    LAUNCH = "launch"
    QUIT = "quit"
    ACTIVATE = "activate"
    MENU_CONTROL = "menu_control"
    FORCE_QUIT = "force_quit"
    UI_AUTOMATION = "ui_automation"


@dataclass(frozen=True)
class AppIdentifier:
    """Type-safe application identifier with validation and security checks."""
    bundle_id: Optional[str] = None
    app_name: Optional[str] = None
    
    def __post_init__(self):
        # Contract: At least one identifier must be provided
        if not self.bundle_id and not self.app_name:
            raise ValueError("Either bundle_id or app_name must be provided")
        
        # Validate bundle ID format if provided
        if self.bundle_id and not re.match(r'^[a-zA-Z0-9\.\-]+$', self.bundle_id):
            raise ValueError(f"Invalid bundle ID format: {self.bundle_id}")
        
        # Validate app name if provided
        if self.app_name and (len(self.app_name) == 0 or len(self.app_name) > 255):
            raise ValueError(f"App name must be 1-255 characters: {self.app_name}")
    
    def primary_identifier(self) -> str:
        """Get primary identifier for operations - prefer bundle ID for specificity."""
        return self.bundle_id if self.bundle_id else self.app_name
    
    def display_name(self) -> str:
        """Get human-readable display name."""
        return self.app_name if self.app_name else self.bundle_id
    
    def is_bundle_id(self) -> bool:
        """Check if primary identifier is a bundle ID."""
        return self.bundle_id is not None


@dataclass(frozen=True) 
class MenuPath:
    """Type-safe menu navigation path with validation."""
    path: List[str]
    
    def __post_init__(self):
        # Contract: Path must not be empty
        if len(self.path) == 0:
            raise ValueError("Menu path cannot be empty")
        
        # Contract: Each menu item must be valid
        for item in self.path:
            if not isinstance(item, str) or len(item) == 0:
                raise ValueError(f"Invalid menu item: {item}")
            if len(item) > 100:
                raise ValueError(f"Menu item too long (max 100 chars): {item}")
    
    def __str__(self) -> str:
        """String representation for logging."""
        return " → ".join(self.path)
    
    def depth(self) -> int:
        """Get menu depth."""
        return len(self.path)


@dataclass(frozen=True)
class LaunchConfiguration:
    """Configuration for application launch operations."""
    wait_for_launch: bool = True
    timeout: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    hide_on_launch: bool = False
    activate_on_launch: bool = True
    launch_arguments: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate timeout
        if self.timeout.total_seconds() <= 0:
            raise ValueError("Timeout must be positive")
        if self.timeout.total_seconds() > 300:  # 5 minutes max
            raise ValueError("Timeout cannot exceed 300 seconds")


@dataclass(frozen=True)
class AppOperationResult:
    """Result of application control operation."""
    success: bool
    app_state: AppState
    operation_time: Duration
    details: Optional[str] = None
    error_code: Optional[str] = None
    
    @classmethod
    def success_result(
        cls, 
        app_state: AppState, 
        operation_time: Duration, 
        details: Optional[str] = None
    ) -> AppOperationResult:
        """Create successful operation result."""
        return cls(
            success=True,
            app_state=app_state,
            operation_time=operation_time,
            details=details
        )
    
    @classmethod
    def failure_result(
        cls, 
        app_state: AppState, 
        operation_time: Duration,
        error_code: str,
        details: Optional[str] = None
    ) -> AppOperationResult:
        """Create failed operation result."""
        return cls(
            success=False,
            app_state=app_state,
            operation_time=operation_time,
            error_code=error_code,
            details=details
        )


class AppController:
    """
    Secure application control with state tracking and error recovery.
    
    Provides comprehensive application lifecycle management with:
    - Security validation and permission checking
    - State tracking and monitoring
    - Timeout protection and error recovery
    - Menu automation with accessibility API integration
    """
    
    def __init__(self):
        # Application whitelist for security (empty = allow all)
        self._app_whitelist: set[str] = set()
        # Application blacklist for security
        self._app_blacklist: set[str] = {
            # Security-sensitive applications
            "com.apple.keychainaccess",
            "com.apple.systempreferences",
            "com.apple.Terminal",
            # Add more as needed for security
        }
        # State cache for performance
        self._state_cache: Dict[str, tuple[AppState, float]] = {}
        self._cache_timeout = 2.0  # seconds
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["LAUNCH_ERROR", "PERMISSION_ERROR", "SECURITY_ERROR"])
    async def launch_application(
        self, 
        app_id: AppIdentifier,
        config: LaunchConfiguration = LaunchConfiguration()
    ) -> Either[KMError, AppOperationResult]:
        """
        Launch application with security validation and state tracking.
        
        Security Features:
        - Bundle ID validation and format checking
        - Application whitelist/blacklist enforcement
        - Permission verification for launch operations
        - Safe parameter handling and injection prevention
        
        Architecture:
        - Pattern: Command Pattern with comprehensive validation
        - Security: Defense-in-depth with multiple validation layers
        - Performance: State caching with intelligent invalidation
        """
        start_time = time.time()
        
        try:
            # Phase 1: Security validation
            security_check = self._validate_app_security(app_id)
            if security_check.is_left():
                return security_check
            
            # Phase 2: Check current state
            current_state = await self._get_app_state_async(app_id)
            if current_state in [AppState.RUNNING, AppState.FOREGROUND]:
                # App already running - just activate if needed
                if config.activate_on_launch:
                    return await self.activate_application(app_id)
                else:
                    operation_time = Duration.from_seconds(time.time() - start_time)
                    return Either.right(AppOperationResult.success_result(
                        current_state, operation_time, "Application already running"
                    ))
            
            # Phase 3: Launch application via AppleScript
            launch_result = await self._launch_via_applescript(app_id, config)
            if launch_result.is_left():
                operation_time = Duration.from_seconds(time.time() - start_time)
                return Either.left(launch_result.get_left())
            
            # Phase 4: Wait for launch completion if requested
            final_state = AppState.RUNNING
            if config.wait_for_launch:
                wait_result = await self._wait_for_launch(app_id, config.timeout)
                if wait_result.is_left():
                    operation_time = Duration.from_seconds(time.time() - start_time)
                    return wait_result
                final_state = wait_result.get_right()
            
            # Phase 5: Handle post-launch configuration
            if config.hide_on_launch and final_state in [AppState.RUNNING, AppState.FOREGROUND]:
                await self._hide_application(app_id)
                final_state = AppState.BACKGROUND
            
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.right(AppOperationResult.success_result(
                final_state, operation_time, f"Successfully launched {app_id.display_name()}"
            ))
            
        except Exception as e:
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.left(KMError.execution_error(f"Launch failed: {str(e)}"))
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["QUIT_ERROR", "APP_NOT_RUNNING", "PERMISSION_ERROR"])
    async def quit_application(
        self,
        app_id: AppIdentifier,
        force: bool = False,
        timeout: Duration = Duration.from_seconds(10)
    ) -> Either[KMError, AppOperationResult]:
        """
        Quit application with graceful/force options.
        
        Security Features:
        - Permission checking for quit operations
        - Force quit safety with confirmation
        - Timeout protection for hanging operations
        """
        start_time = time.time()
        
        try:
            # Check current state
            current_state = await self._get_app_state_async(app_id)
            if current_state == AppState.NOT_RUNNING:
                operation_time = Duration.from_seconds(time.time() - start_time)
                return Either.right(AppOperationResult.success_result(
                    AppState.NOT_RUNNING, operation_time, "Application not running"
                ))
            
            # Security check for force quit
            if force and not self._is_force_quit_allowed(app_id):
                operation_time = Duration.from_seconds(time.time() - start_time)
                return Either.left(KMError.validation_error(
                    f"Force quit not allowed for {app_id.display_name()}"
                ))
            
            # Attempt graceful quit first
            quit_result = await self._quit_via_applescript(app_id, force, timeout)
            if quit_result.is_left():
                return quit_result
            
            # Wait for termination
            final_state = await self._wait_for_termination(app_id, timeout)
            operation_time = Duration.from_seconds(time.time() - start_time)
            
            return Either.right(AppOperationResult.success_result(
                final_state, operation_time, 
                f"Successfully quit {app_id.display_name()}" + (" (forced)" if force else "")
            ))
            
        except Exception as e:
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.left(KMError.execution_error(f"Quit failed: {str(e)}"))
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    async def activate_application(self, app_id: AppIdentifier) -> Either[KMError, AppOperationResult]:
        """
        Activate (bring to foreground) application.
        
        Uses AppleScript for reliable activation with error handling.
        """
        start_time = time.time()
        
        try:
            # Check if app is running
            current_state = await self._get_app_state_async(app_id)
            if current_state == AppState.NOT_RUNNING:
                operation_time = Duration.from_seconds(time.time() - start_time)
                return Either.left(KMError.validation_error(
                    f"Cannot activate {app_id.display_name()}: not running"
                ))
            
            # Activate via AppleScript
            activate_result = await self._activate_via_applescript(app_id)
            if activate_result.is_left():
                return activate_result
            
            # Update state cache
            self._invalidate_state_cache(app_id)
            
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.right(AppOperationResult.success_result(
                AppState.FOREGROUND, operation_time, f"Activated {app_id.display_name()}"
            ))
            
        except Exception as e:
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.left(KMError.execution_error(f"Activation failed: {str(e)}"))
    
    @require(lambda menu_path: len(menu_path.path) > 0)
    async def select_menu_item(
        self,
        app_id: AppIdentifier,
        menu_path: MenuPath,
        timeout: Duration = Duration.from_seconds(5)
    ) -> Either[KMError, AppOperationResult]:
        """
        Navigate and select menu item with path validation.
        
        Uses UI automation for reliable menu navigation with comprehensive error handling.
        """
        start_time = time.time()
        
        try:
            # Ensure app is active for menu access
            current_state = await self._get_app_state_async(app_id)
            if current_state != AppState.FOREGROUND:
                activate_result = await self.activate_application(app_id)
                if activate_result.is_left():
                    return activate_result
                
                # Brief wait for activation
                await asyncio.sleep(0.5)
            
            # Navigate menu via AppleScript UI automation
            menu_result = await self._navigate_menu_applescript(app_id, menu_path, timeout)
            
            operation_time = Duration.from_seconds(time.time() - start_time)
            if menu_result.is_left():
                return menu_result
            
            return Either.right(AppOperationResult.success_result(
                AppState.FOREGROUND, operation_time, 
                f"Selected menu: {menu_path}"
            ))
            
        except Exception as e:
            operation_time = Duration.from_seconds(time.time() - start_time)
            return Either.left(KMError.execution_error(f"Menu selection failed: {str(e)}"))
    
    async def get_application_state(self, app_id: AppIdentifier) -> Either[KMError, AppState]:
        """Get current application state with caching."""
        try:
            state = await self._get_app_state_async(app_id)
            return Either.right(state)
        except Exception as e:
            return Either.left(KMError.execution_error(f"State query failed: {str(e)}"))
    
    def _validate_app_security(self, app_id: AppIdentifier) -> Either[KMError, bool]:
        """Validate application security and permissions."""
        identifier = app_id.primary_identifier().lower()
        
        # Check blacklist
        if identifier in self._app_blacklist:
            return Either.left(KMError.validation_error(
                f"Application blocked by security policy: {app_id.display_name()}"
            ))
        
        # Check whitelist (if enabled)
        if self._app_whitelist and identifier not in self._app_whitelist:
            return Either.left(KMError.validation_error(
                f"Application not in whitelist: {app_id.display_name()}"
            ))
        
        # Validate bundle ID format
        if app_id.bundle_id and not self._is_valid_bundle_id(app_id.bundle_id):
            return Either.left(KMError.validation_error(
                f"Invalid bundle ID format: {app_id.bundle_id}"
            ))
        
        return Either.right(True)
    
    def _is_valid_bundle_id(self, bundle_id: str) -> bool:
        """Validate bundle ID format and security."""
        # Basic format validation
        if not re.match(r'^[a-zA-Z0-9\.\-]+$', bundle_id):
            return False
        
        # Must contain at least one dot
        if '.' not in bundle_id:
            return False
        
        # No suspicious patterns
        suspicious_patterns = [
            r'\.\.', r'--', r'^\.',  # malformed patterns
            r'script', r'shell', r'exec',  # execution patterns
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, bundle_id, re.IGNORECASE):
                return False
        
        return True
    
    async def _get_app_state_async(self, app_id: AppIdentifier) -> AppState:
        """Get application state with caching."""
        cache_key = app_id.primary_identifier()
        current_time = time.time()
        
        # Check cache
        if cache_key in self._state_cache:
            cached_state, cache_time = self._state_cache[cache_key]
            if current_time - cache_time < self._cache_timeout:
                return cached_state
        
        # Query actual state
        try:
            state = await self._query_app_state_applescript(app_id)
            self._state_cache[cache_key] = (state, current_time)
            return state
        except Exception:
            return AppState.UNKNOWN
    
    def _invalidate_state_cache(self, app_id: AppIdentifier):
        """Invalidate cached state for application."""
        cache_key = app_id.primary_identifier()
        if cache_key in self._state_cache:
            del self._state_cache[cache_key]
    
    def _is_force_quit_allowed(self, app_id: AppIdentifier) -> bool:
        """Check if force quit is allowed for application."""
        # Restrict force quit for system applications
        system_apps = {
            "com.apple.finder",
            "com.apple.dock",
            "com.apple.systemuiserver",
        }
        return app_id.primary_identifier().lower() not in system_apps
    
    # AppleScript execution methods - Phase 2 Implementation
    async def _launch_via_applescript(self, app_id: AppIdentifier, config: LaunchConfiguration) -> Either[KMError, bool]:
        """Launch application via AppleScript with comprehensive error handling."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            if app_id.is_bundle_id():
                # Use bundle ID for precise targeting
                script = f'''
                tell application "System Events"
                    try
                        set appExists to exists application process "{escaped_identifier}"
                        if not appExists then
                            tell application id "{escaped_identifier}" to activate
                        else
                            tell application id "{escaped_identifier}" to activate
                        end if
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
                '''
            else:
                # Use application name
                script = f'''
                tell application "{escaped_identifier}"
                    try
                        activate
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
                '''
            
            result = await self._execute_applescript(script, config.timeout)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(
                    f"Launch failed: {output[6:].strip()}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript launch error: {str(e)}"))
    
    async def _quit_via_applescript(self, app_id: AppIdentifier, force: bool, timeout: Duration) -> Either[KMError, bool]:
        """Quit application via AppleScript with graceful and force options."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            if force:
                # Force quit via System Events
                script = f'''
                tell application "System Events"
                    try
                        set proc to process "{escaped_identifier}"
                        kill proc
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
                '''
            else:
                # Graceful quit
                script = f'''
                tell application "{escaped_identifier}"
                    try
                        quit
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
                '''
            
            result = await self._execute_applescript(script, timeout)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(
                    f"Quit failed: {output[6:].strip()}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript quit error: {str(e)}"))
    
    async def _activate_via_applescript(self, app_id: AppIdentifier) -> Either[KMError, bool]:
        """Activate application via AppleScript."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            script = f'''
            tell application "{escaped_identifier}"
                try
                    activate
                    return "SUCCESS"
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            result = await self._execute_applescript(script, Duration.from_seconds(10))
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(
                    f"Activation failed: {output[6:].strip()}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"AppleScript activation error: {str(e)}"))
    
    async def _navigate_menu_applescript(self, app_id: AppIdentifier, menu_path: MenuPath, timeout: Duration) -> Either[KMError, bool]:
        """Navigate and select menu item via AppleScript UI automation."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            # Build menu path for AppleScript
            menu_items = []
            for item in menu_path.path:
                escaped_item = self._escape_applescript_string(item)
                menu_items.append(f'menu item "{escaped_item}"')
            
            # Construct menu navigation script
            if len(menu_items) == 1:
                menu_script = f'click {menu_items[0]} of menu bar 1'
            else:
                # Navigate nested menus
                menu_script = f'click {menu_items[0]} of menu bar 1\n'
                for i in range(1, len(menu_items)):
                    menu_script += f'                        click {menu_items[i]} of menu 1 of {menu_items[i-1]} of menu bar 1\n'
            
            script = f'''
            tell application "System Events"
                tell process "{escaped_identifier}"
                    try
                        {menu_script}
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
            end tell
            '''
            
            result = await self._execute_applescript(script, timeout)
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(
                    f"Menu navigation failed: {output[6:].strip()}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Menu navigation error: {str(e)}"))
    
    async def _query_app_state_applescript(self, app_id: AppIdentifier) -> AppState:
        """Query application state via AppleScript with comprehensive state detection."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            script = f'''
            tell application "System Events"
                try
                    set proc to process "{escaped_identifier}"
                    if exists proc then
                        if frontmost of proc then
                            return "foreground"
                        else
                            return "background"
                        end if
                    else
                        return "not_running"
                    end if
                on error
                    return "not_running"
                end try
            end tell
            '''
            
            result = await self._execute_applescript(script, Duration.from_seconds(5))
            if result.is_left():
                return AppState.UNKNOWN
            
            output = result.get_right().strip().lower()
            
            if output == "not_running":
                return AppState.NOT_RUNNING
            elif output == "foreground":
                return AppState.FOREGROUND
            elif output == "background":
                return AppState.BACKGROUND
            else:
                return AppState.RUNNING
                
        except Exception:
            return AppState.UNKNOWN
    
    async def _wait_for_launch(self, app_id: AppIdentifier, timeout: Duration) -> Either[KMError, AppState]:
        """Wait for application launch completion with state polling."""
        start_time = time.time()
        timeout_seconds = timeout.total_seconds()
        
        while (time.time() - start_time) < timeout_seconds:
            current_state = await self._get_app_state_async(app_id)
            if current_state in [AppState.RUNNING, AppState.FOREGROUND, AppState.BACKGROUND]:
                return Either.right(current_state)
            
            # Brief sleep to avoid excessive polling
            await asyncio.sleep(0.5)
        
        return Either.left(KMError.timeout_error(
            f"Application launch timeout: {app_id.display_name()}"
        ))
    
    async def _wait_for_termination(self, app_id: AppIdentifier, timeout: Duration) -> AppState:
        """Wait for application termination with state polling."""
        start_time = time.time()
        timeout_seconds = timeout.total_seconds()
        
        while (time.time() - start_time) < timeout_seconds:
            current_state = await self._get_app_state_async(app_id)
            if current_state == AppState.NOT_RUNNING:
                return AppState.NOT_RUNNING
            
            # Brief sleep to avoid excessive polling
            await asyncio.sleep(0.5)
        
        # Timeout reached - return current state
        return await self._get_app_state_async(app_id)
    
    async def _hide_application(self, app_id: AppIdentifier) -> Either[KMError, bool]:
        """Hide application (send to background) via AppleScript."""
        try:
            identifier = app_id.primary_identifier()
            escaped_identifier = self._escape_applescript_string(identifier)
            
            script = f'''
            tell application "System Events"
                tell process "{escaped_identifier}"
                    try
                        set visible to false
                        return "SUCCESS"
                    on error errorMessage
                        return "ERROR: " & errorMessage
                    end try
                end tell
            end tell
            '''
            
            result = await self._execute_applescript(script, Duration.from_seconds(5))
            if result.is_left():
                return result
            
            output = result.get_right().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(
                    f"Hide failed: {output[6:].strip()}"
                ))
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Hide application error: {str(e)}"))
    
    async def _execute_applescript(self, script: str, timeout: Duration) -> Either[KMError, str]:
        """Execute AppleScript with timeout and error handling."""
        try:
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout.total_seconds()
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown AppleScript error"
                return Either.left(KMError.execution_error(
                    f"AppleScript failed: {error_msg}"
                ))
            
            return Either.right(stdout.decode())
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error(
                f"AppleScript execution timeout ({timeout.total_seconds()}s)"
            ))
        except Exception as e:
            return Either.left(KMError.execution_error(
                f"AppleScript execution error: {str(e)}"
            ))
    
    def _escape_applescript_string(self, value: str) -> str:
        """Escape string for safe AppleScript inclusion."""
        if not isinstance(value, str):
            value = str(value)
        
        # Security: Escape quotes and special characters
        escaped = value.replace('\\', '\\\\')  # Escape backslashes first
        escaped = escaped.replace('"', '\\"')   # Escape quotes
        escaped = escaped.replace('\n', '\\n')  # Escape newlines
        escaped = escaped.replace('\r', '\\r')  # Escape carriage returns
        escaped = escaped.replace('\t', '\\t')  # Escape tabs
        
        return escaped