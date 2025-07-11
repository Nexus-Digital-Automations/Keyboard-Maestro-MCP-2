# TASK_12: km_app_control - Application Management

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: HIGH | **Duration**: 3 hours
**Technique Focus**: System Integration + Permission Management + Error Recovery + State Tracking
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_10 (km_create_macro foundation)
**Blocking**: TASK_13 (file operations), TASK_16 (window management)

## 📖 Required Reading (Complete before starting)
- [ ] **development/protocols/KM_MCP.md**: km_app_control specification (lines 731-743)
- [ ] **src/integration/km_client.py**: AppleScript patterns for system interaction
- [ ] **src/core/types.py**: Permission system and branded types
- [ ] **macOS Application Control**: Understanding bundle IDs, launch services, app states
- [ ] **tests/TESTING.md**: Current test framework and system integration testing

## 🎯 Implementation Overview
Create a comprehensive application control system that enables AI assistants to launch, quit, activate applications, automate menu selections, and interact with UI elements while maintaining security boundaries and providing robust error handling for system-level operations.

<thinking>
Application control is fundamental for system automation:
1. Security Critical: Must validate app bundle IDs to prevent launching malicious apps
2. State Management: Track application states (running, foreground, background)
3. Menu Automation: Navigate complex menu structures safely
4. Error Recovery: Handle app launch failures, permission issues, timeouts
5. Permission Model: Respect system security and user permissions
6. Integration: Work with existing macro creation for app-specific workflows
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Application Control Infrastructure
- [ ] **Application types**: Define AppIdentifier, AppState, MenuPath, LaunchConfiguration types
- [ ] **Security validation**: Bundle ID validation, application whitelist/blacklist support
- [ ] **State tracking**: Monitor application launch, quit, activation states
- [ ] **Permission management**: Verify system permissions for app control operations

### Phase 2: AppleScript & System Integration
- [ ] **App lifecycle control**: Launch, quit, activate, hide applications via AppleScript
- [ ] **Menu automation**: Navigate menu hierarchies with path-based selection
- [ ] **UI interaction**: Basic UI element interaction through accessibility APIs
- [ ] **Error handling**: Comprehensive error recovery for system-level failures

### Phase 3: Advanced Features & Safety
- [ ] **Bundle ID resolution**: Convert app names to valid bundle identifiers
- [ ] **Force quit handling**: Safe force termination with user confirmation
- [ ] **Multi-instance support**: Handle apps with multiple windows/instances
- [ ] **Timeout management**: Prevent hanging operations with configurable timeouts

### Phase 4: MCP Tool Integration
- [ ] **Tool implementation**: km_app_control MCP tool with operation modes
- [ ] **Operation types**: launch, quit, activate, menu_select, force_quit
- [ ] **Response formatting**: Application state information and operation results
- [ ] **Testing integration**: System integration tests with real applications

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/applications/app_controller.py - Core Application Control
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import subprocess

from ..core.types import Duration
from ..core.contracts import require, ensure

class AppState(Enum):
    """Application execution states."""
    NOT_RUNNING = "not_running"
    LAUNCHING = "launching"
    RUNNING = "running"
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    TERMINATING = "terminating"

@dataclass(frozen=True)
class AppIdentifier:
    """Type-safe application identifier."""
    bundle_id: Optional[str] = None
    app_name: Optional[str] = None
    
    @require(lambda self: self.bundle_id is not None or self.app_name is not None)
    @require(lambda self: not self.bundle_id or re.match(r'^[a-zA-Z0-9\.\-]+$', self.bundle_id))
    def __post_init__(self):
        pass
    
    def primary_identifier(self) -> str:
        """Get primary identifier for operations."""
        return self.bundle_id if self.bundle_id else self.app_name

@dataclass(frozen=True) 
class MenuPath:
    """Type-safe menu navigation path."""
    path: List[str]
    
    @require(lambda self: len(self.path) > 0)
    @require(lambda self: all(len(item) > 0 and len(item) <= 100 for item in self.path))
    def __post_init__(self):
        pass

class AppController:
    """Secure application control with state tracking and error recovery."""
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["LAUNCH_ERROR", "PERMISSION_ERROR"])
    async def launch_application(
        self, 
        app_id: AppIdentifier,
        wait_for_launch: bool = True,
        timeout: Duration = Duration.from_seconds(30)
    ) -> Either[KMError, AppState]:
        """Launch application with security validation and state tracking."""
        pass
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["QUIT_ERROR", "APP_NOT_RUNNING"])
    async def quit_application(
        self,
        app_id: AppIdentifier,
        force: bool = False,
        timeout: Duration = Duration.from_seconds(10)
    ) -> Either[KMError, bool]:
        """Quit application with graceful/force options."""
        pass
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    async def activate_application(self, app_id: AppIdentifier) -> Either[KMError, AppState]:
        """Activate (bring to foreground) application."""
        pass
    
    @require(lambda menu_path: len(menu_path.path) > 0)
    async def select_menu_item(
        self,
        app_id: AppIdentifier,
        menu_path: MenuPath,
        timeout: Duration = Duration.from_seconds(5)
    ) -> Either[KMError, bool]:
        """Navigate and select menu item with path validation."""
        pass
    
    def _validate_bundle_id(self, bundle_id: str) -> bool:
        """Validate bundle ID format and security."""
        pass
    
    def _get_app_state(self, app_id: AppIdentifier) -> AppState:
        """Get current application state."""
        pass
    
    def _is_application_safe(self, app_id: AppIdentifier) -> bool:
        """Security check for application launch permissions."""
        pass
```

#### src/applications/menu_navigator.py - Menu Automation System
```python
from typing import List, Dict, Any, Optional

class MenuNavigator:
    """Safe menu navigation with accessibility API integration."""
    
    @require(lambda app_id: app_id.primary_identifier() != "")
    @require(lambda path: len(path) > 0)
    async def navigate_menu(
        self,
        app_id: AppIdentifier,
        menu_path: List[str],
        timeout: Duration = Duration.from_seconds(5)
    ) -> Either[KMError, bool]:
        """Navigate menu hierarchy with error recovery."""
        pass
    
    def _find_menu_item(self, menu_name: str, parent_menu: Any) -> Optional[Any]:
        """Find menu item in menu structure."""
        pass
    
    def _click_menu_item(self, menu_item: Any) -> bool:
        """Click menu item with accessibility API."""
        pass
    
    def _validate_menu_path(self, path: List[str]) -> bool:
        """Validate menu path for security and format."""
        pass
```

#### src/server/tools/app_control_tools.py - MCP Tool Implementation
```python
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
        description="Menu path for menu_select operation",
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
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive application control with security and error handling.
    
    Operations:
    - launch: Start application with launch validation
    - quit: Terminate application (graceful or forced)
    - activate: Bring application to foreground
    - menu_select: Navigate and select menu items
    - get_state: Query current application state
    
    Security Features:
    - Bundle ID validation and application whitelist checking
    - Permission verification for system-level operations
    - Safe menu navigation with path validation
    - Timeout protection for hanging operations
    
    Returns operation results with application state and timing information.
    """
    if ctx:
        await ctx.info(f"Performing app control operation: {operation} on {app_identifier}")
    
    try:
        app_controller = AppController()
        
        # Parse application identifier
        app_id = AppIdentifier(
            bundle_id=app_identifier if "." in app_identifier else None,
            app_name=app_identifier if "." not in app_identifier else None
        )
        
        if operation == "launch":
            # Launch application with validation
            pass
        elif operation == "quit":
            # Quit application (graceful or force)
            pass
        elif operation == "activate":
            # Activate application
            pass
        elif operation == "menu_select":
            # Navigate and select menu item
            pass
        elif operation == "get_state":
            # Get application state
            pass
            
    except Exception as e:
        # Comprehensive error handling
        pass
```

### Files to Enhance:

#### src/core/types.py - Add Application Types
```python
# Add to existing types
AppId = NewType('AppId', str)
BundleId = NewType('BundleId', str)
MenuItemId = NewType('MenuItemId', str)

class ApplicationPermission(Enum):
    """Application control permission levels."""
    LAUNCH = "launch"
    QUIT = "quit"
    ACTIVATE = "activate"
    MENU_CONTROL = "menu_control"
    FORCE_QUIT = "force_quit"
    UI_AUTOMATION = "ui_automation"
```

#### src/integration/km_client.py - Add Application Methods
```python
def launch_app_applescript(self, app_identifier: str) -> Either[KMError, bool]:
    """Launch application via AppleScript."""
    script = f'''
    tell application "System Events"
        try
            tell application "{app_identifier}" to activate
            return "SUCCESS"
        on error errorMessage
            return "ERROR: " & errorMessage
        end try
    end tell
    '''
    pass

def quit_app_applescript(self, app_identifier: str, force: bool = False) -> Either[KMError, bool]:
    """Quit application via AppleScript."""
    pass

def select_menu_applescript(self, app_identifier: str, menu_path: List[str]) -> Either[KMError, bool]:
    """Select menu item via AppleScript UI automation."""
    pass
```

## 🏗️ Modularity Strategy
- **src/applications/**: New directory for application control (<250 lines each)
- **app_controller.py**: Core control operations and security (240 lines)
- **menu_navigator.py**: Menu automation and UI interaction (180 lines)
- **src/server/tools/app_control_tools.py**: MCP tool implementation (220 lines)
- **Enhance existing files**: Minimal additions to types.py and km_client.py

## 🔒 Security Implementation
1. **Bundle ID Validation**: Verify bundle IDs match expected format and whitelist
2. **Application Whitelist**: Optional whitelist of approved applications for launch
3. **Permission Checking**: Verify system permissions for app control operations
4. **Safe Menu Navigation**: Validate menu paths and prevent UI injection attacks
5. **Timeout Protection**: Prevent hanging operations from blocking the system
6. **Force Quit Safety**: Require explicit confirmation for force termination

## 📊 Performance Targets
- **App Launch**: <5 seconds for typical applications
- **App Quit**: <3 seconds for graceful quit, <1 second for force quit
- **Menu Navigation**: <2 seconds for simple menu paths
- **State Query**: <500ms for application state checking
- **Activation**: <1 second for application activation

## ✅ Success Criteria
- [ ] All advanced techniques implemented (system integration, permission management, error recovery)
- [ ] Complete security validation with bundle ID verification and permissions
- [ ] Support for launch, quit, activate, and menu automation operations
- [ ] Real system integration with macOS applications (no mock data)
- [ ] Comprehensive error handling with timeout protection and recovery
- [ ] Property-based testing covers all application control scenarios
- [ ] Performance meets sub-5-second targets for most operations
- [ ] Integration with existing MCP framework and security model
- [ ] TESTING.md updated with application control system tests
- [ ] Full documentation with security guidelines and permissions

## 🎨 Usage Examples

### Basic Application Control
```python
# Launch application
result = await client.call_tool("km_app_control", {
    "operation": "launch",
    "app_identifier": "com.apple.TextEdit",
    "wait_for_completion": True
})

# Quit application gracefully
result = await client.call_tool("km_app_control", {
    "operation": "quit",
    "app_identifier": "TextEdit",
    "force_quit": False
})
```

### Advanced Menu Automation
```python
# Navigate complex menu structure
result = await client.call_tool("km_app_control", {
    "operation": "menu_select",
    "app_identifier": "com.apple.finder",
    "menu_path": ["File", "New Folder"],
    "timeout_seconds": 10
})

# Check application state
result = await client.call_tool("km_app_control", {
    "operation": "get_state",
    "app_identifier": "Safari"
})
```

## 🧪 Testing Strategy
- **Property-Based Testing**: Random application operations with various identifiers
- **Security Testing**: Attempt to launch unauthorized applications, test permission boundaries
- **Integration Testing**: Real application launches, quits, and menu operations
- **Performance Testing**: Measure operation timings and timeout handling
- **Error Recovery Testing**: Test handling of missing apps, permission failures
- **Menu Navigation Testing**: Complex menu paths and edge cases