# TASK_10: km_macro_manager - Comprehensive Macro Creation, Editing & Debugging

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: HIGH | **Duration**: 6 hours
**Technique Focus**: Design by Contract + Type Safety + Security Validation + Interactive Debugging + Template Engine
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 📋 **MERGED FUNCTIONALITY**
This task combines macro creation (original TASK_10) with interactive editing and debugging (TASK_28) for a comprehensive macro management system.

## 🚦 Status & Assignment
**Status**: COMPLETED ✅
**Assigned**: Agent_1 (Agent_ADDER+)
**Dependencies**: TASK_1-9 (Foundation completed) ✅
**Blocking**: TASK_11, TASK_12, TASK_14, TASK_15, TASK_17, TASK_18, TASK_19 → NOW UNBLOCKED

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_create_macro specification (lines 296-311)
- [x] **src/integration/km_client.py**: Current KM client and AppleScript integration patterns
- [x] **src/core/types.py**: Branded types for MacroId, GroupId, and validation
- [x] **src/server/tools/core_tools.py**: Existing tool implementation patterns
- [x] **tests/TESTING.md**: Current test framework and requirements

## 🎯 Implementation Overview
Create a comprehensive macro management platform that combines macro creation, editing, debugging, and validation capabilities. This unified system enables AI assistants to:

1. **Create Macros**: Programmatically create new macros from scratch with templates
2. **Edit Macros**: Modify existing macros with interactive debugging
3. **Debug Macros**: Step-through execution with variable monitoring
4. **Validate Macros**: Real-time validation and error detection

<thinking>
Macro management is the foundation tool that unlocks AI automation capabilities:
1. Security Critical: Must validate all macro components to prevent malicious automation
2. Template System: Common patterns like "Open App", "Text Expansion", "File Processing"
3. Interactive Editing: Real-time macro modification with debugging capabilities
4. Debugging Engine: Step-through execution with breakpoints and variable inspection
5. Validation Engine: Ensure created/modified macros are syntactically correct and safe
6. Integration: Work with existing KM AppleScript API for macro operations
7. Error Recovery: Rollback on creation/modification failures, provide detailed error messages
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Macro Management Infrastructure ✅ COMPLETED
- [x] **Macro creation types**: Define MacroTemplate, MacroBuilder, CreationRequest types
- [x] **Macro editing types**: Define MacroEditor, EditSession, ModificationRequest types
- [x] **Debugging types**: Define DebugSession, Breakpoint, ExecutionState types
- [x] **Security validation**: Input sanitization, permission checking, safety validation
- [x] **Template system**: Pre-built templates for common automation patterns
- [x] **Builder pattern**: Fluent API for programmatic macro construction

### Phase 2: AppleScript Integration ✅ COMPLETED
- [x] **KM creation API**: AppleScript commands for macro creation via "tell application Keyboard Maestro"
- [x] **KM editing API**: AppleScript commands for macro modification and debugging
- [x] **Debug integration**: Real-time execution monitoring and step-through control
- [x] **Group management**: Integrate with macro groups, handle group selection/creation
- [x] **Property setting**: Configure macro name, enabled state, color, notes
- [x] **Error handling**: Comprehensive error recovery and rollback mechanisms

### Phase 3: Template Engine & Validation ✅ COMPLETED
- [x] **Template library**: Common templates (hotkey actions, app automation, text processing)
- [x] **Validation engine**: Syntax checking, dependency validation, safety verification
- [x] **Conflict detection**: Check for naming conflicts, hotkey conflicts
- [x] **Preview system**: Generate macro preview before creation/modification
- [x] **Interactive debugging**: Step-through execution with breakpoints and variable monitoring
- [x] **Edit validation**: Real-time validation during macro editing and modification

### Phase 4: MCP Tool Integration ✅ COMPLETED
- [x] **Tool implementation**: km_macro_manager MCP tool with creation, editing, and debugging
- [x] **Parameter validation**: Comprehensive input validation with branded types
- [x] **Response formatting**: Standardized success/error responses with metadata
- [x] **Debug session management**: Interactive debugging session lifecycle management
- [x] **Testing integration**: Property-based tests and integration tests

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/creation/macro_manager.py - Comprehensive Macro Management Engine
```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..core.types import MacroId, GroupId, TriggerId
from ..core.contracts import require, ensure

class MacroTemplate(Enum):
    """Pre-built macro templates for common patterns."""
    HOTKEY_ACTION = "hotkey_action"
    APP_LAUNCHER = "app_launcher" 
    TEXT_EXPANSION = "text_expansion"
    FILE_PROCESSOR = "file_processor"
    WINDOW_MANAGER = "window_manager"
    CUSTOM = "custom"

@dataclass(frozen=True)
class MacroCreationRequest:
    """Type-safe macro creation request."""
    name: str
    template: MacroTemplate
    group_id: Optional[GroupId] = None
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.name) > 0 and len(self.name) <= 255)
    @require(lambda self: self.name.isascii())  # Security: ASCII only
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class MacroEditRequest:
    """Type-safe macro editing request."""
    macro_id: MacroId
    operation: str  # "modify", "debug", "validate", "preview"
    changes: Dict[str, Any] = field(default_factory=dict)
    debug_options: Optional[Dict[str, Any]] = None
    
    @require(lambda self: len(self.operation) > 0)
    @require(lambda self: self.operation in ["modify", "debug", "validate", "preview"])
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class DebugSession:
    """Interactive debugging session state."""
    session_id: str
    macro_id: MacroId
    breakpoints: List[int] = field(default_factory=list)
    current_step: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)
    
    @require(lambda self: len(self.session_id) > 0)
    def __post_init__(self):
        pass

class MacroManager:
    """Comprehensive macro management with creation, editing, and debugging."""
    
    def __init__(self):
        self.debug_sessions: Dict[str, DebugSession] = {}
        self.macro_builder = MacroBuilder()
        self.macro_editor = MacroEditor()
    
    @require(lambda request: request.template in MacroTemplate)
    @ensure(lambda result: result.is_right() or result.get_left().code == "VALIDATION_ERROR")
    async def create_macro(self, request: MacroCreationRequest) -> Either[KMError, MacroId]:
        """Create macro with comprehensive validation and error handling."""
        return await self.macro_builder.create_macro(request)
    
    @require(lambda request: request.operation in ["modify", "debug", "validate", "preview"])
    @ensure(lambda result: result.is_right() or result.get_left().code in ["EDIT_ERROR", "DEBUG_ERROR"])
    async def edit_macro(self, request: MacroEditRequest) -> Either[KMError, Dict[str, Any]]:
        """Edit macro with interactive debugging and validation."""
        if request.operation == "debug":
            return await self._start_debug_session(request)
        elif request.operation == "modify":
            return await self.macro_editor.modify_macro(request.macro_id, request.changes)
        elif request.operation == "validate":
            return await self.macro_editor.validate_macro(request.macro_id)
        elif request.operation == "preview":
            return await self.macro_editor.preview_changes(request.macro_id, request.changes)
    
    async def _start_debug_session(self, request: MacroEditRequest) -> Either[KMError, Dict[str, Any]]:
        """Start interactive debugging session."""
        session_id = f"debug_{request.macro_id}_{int(time.time())}"
        
        debug_session = DebugSession(
            session_id=session_id,
            macro_id=request.macro_id,
            breakpoints=request.debug_options.get("breakpoints", []) if request.debug_options else []
        )
        
        self.debug_sessions[session_id] = debug_session
        
        return Either.right({
            "session_id": session_id,
            "status": "debug_started",
            "breakpoints": debug_session.breakpoints,
            "available_commands": ["step", "continue", "inspect", "set_breakpoint", "stop"]
        })
    
    def _validate_security(self, request: MacroCreationRequest) -> bool:
        """Validate macro creation request for security compliance."""
        return self.macro_builder._validate_security(request)
    
    def _generate_applescript(self, request: MacroCreationRequest) -> str:
        """Generate AppleScript for macro creation."""
        return self.macro_builder._generate_applescript(request)

class MacroBuilder:
    """Fluent builder for macro creation with security validation."""
    
    @require(lambda request: request.template in MacroTemplate)
    @ensure(lambda result: result.is_right() or result.get_left().code == "VALIDATION_ERROR")
    async def create_macro(self, request: MacroCreationRequest) -> Either[KMError, MacroId]:
        """Create macro with comprehensive validation and error handling."""
        pass
    
    def _validate_security(self, request: MacroCreationRequest) -> bool:
        """Validate macro creation request for security compliance."""
        pass
    
    def _generate_applescript(self, request: MacroCreationRequest) -> str:
        """Generate AppleScript for macro creation."""
        pass

class MacroEditor:
    """Interactive macro editing with real-time validation."""
    
    async def modify_macro(self, macro_id: MacroId, changes: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Modify existing macro with validation."""
        pass
    
    async def validate_macro(self, macro_id: MacroId) -> Either[KMError, Dict[str, Any]]:
        """Validate macro syntax and dependencies."""
        pass
    
    async def preview_changes(self, macro_id: MacroId, changes: Dict[str, Any]) -> Either[KMError, Dict[str, Any]]:
        """Preview macro changes without applying."""
        pass
```

#### src/creation/templates.py - Macro Template System
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class MacroTemplateGenerator(ABC):
    """Abstract base for macro template generators."""
    
    @abstractmethod
    def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action configurations for this template."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate template-specific parameters."""
        pass

class HotkeyActionTemplate(MacroTemplateGenerator):
    """Template for hotkey-triggered actions."""
    
    def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hotkey action configuration with security validation."""
        pass

class AppLauncherTemplate(MacroTemplateGenerator):
    """Template for application launcher macros."""
    
    def generate_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate app launcher configuration with bundle ID validation."""
        pass
```

#### src/server/tools/macro_management_tools.py - MCP Tool Implementation
```python
async def km_macro_manager(
    operation: Annotated[str, Field(
        description="Operation type: create|edit|debug|validate|preview",
        pattern=r"^(create|edit|debug|validate|preview)$"
    )],
    macro_identifier: Annotated[Optional[str], Field(
        default=None,
        description="Macro ID for edit/debug operations",
        max_length=255
    )] = None,
    name: Annotated[Optional[str], Field(
        default=None,
        description="Macro name for creation (1-255 ASCII characters)",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\s\-\.]+$"  # Security: Restricted character set
    )] = None,
    template: Annotated[Optional[str], Field(
        default=None,
        description="Macro template type for creation",
        pattern=r"^(hotkey_action|app_launcher|text_expansion|file_processor|window_manager|custom)$"
    )] = None,
    group_name: Annotated[Optional[str], Field(
        default=None,
        description="Target macro group name",
        max_length=255
    )] = None,
    enabled: Annotated[bool, Field(
        default=True,
        description="Initial enabled state"
    )] = True,
    parameters: Annotated[Dict[str, Any], Field(
        default_factory=dict,
        description="Operation-specific parameters (template params for create, changes for edit, debug options)"
    )] = {},
    debug_options: Annotated[Optional[Dict[str, Any]], Field(
        default=None,
        description="Debugging options: breakpoints, step_mode, variable_watch"
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive macro management: create, edit, debug, validate, and preview macros.
    
    Operations:
    - create: Create new macro with templates (hotkey_action, app_launcher, text_expansion, etc.)
    - edit: Modify existing macro with interactive editing
    - debug: Start debugging session with step-through execution
    - validate: Validate macro syntax and dependencies
    - preview: Preview changes without applying
    
    Templates for creation:
    - hotkey_action: Hotkey-triggered actions
    - app_launcher: Application launching macros
    - text_expansion: Text expansion snippets
    - file_processor: File processing workflows
    - window_manager: Window manipulation macros
    - custom: Custom macro with user-defined actions
    
    Returns operation results with status, data, and metadata.
    """
    if ctx:
        await ctx.info(f"Managing macro: operation='{operation}', macro='{macro_identifier}'")
    
    try:
        macro_manager = MacroManager()
        
        if operation == "create":
            if not name or not template:
                raise ValidationError("Name and template required for macro creation")
            
            creation_request = MacroCreationRequest(
                name=name,
                template=MacroTemplate(template),
                group_id=parameters.get("group_id"),
                enabled=parameters.get("enabled", True),
                parameters=parameters
            )
            
            result = await macro_manager.create_macro(creation_request)
            
            if result.is_left():
                error = result.get_left()
                raise ExecutionError(f"Macro creation failed: {error.message}")
            
            macro_id = result.get_right()
            return {
                "success": True,
                "operation": "create",
                "macro_created": {
                    "macro_id": str(macro_id),
                    "name": name,
                    "template": template,
                    "enabled": creation_request.enabled
                }
            }
            
        elif operation in ["edit", "debug", "validate", "preview"]:
            if not macro_identifier:
                raise ValidationError(f"Macro identifier required for {operation} operation")
            
            edit_request = MacroEditRequest(
                macro_id=MacroId(macro_identifier),
                operation=operation,
                changes=parameters,
                debug_options=debug_options
            )
            
            result = await macro_manager.edit_macro(edit_request)
            
            if result.is_left():
                error = result.get_left()
                raise ExecutionError(f"Macro {operation} failed: {error.message}")
            
            operation_result = result.get_right()
            return {
                "success": True,
                "operation": operation,
                "macro_id": macro_identifier,
                "result": operation_result
            }
        
        else:
            raise ValidationError(f"Unknown operation: {operation}")
            
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "MACRO_MANAGEMENT_ERROR",
                "message": str(e),
                "operation": operation,
                "macro_id": macro_identifier
            }
        }
```

### Files to Enhance:

#### src/core/types.py - Add Creation Types
```python
# Add to existing types
TemplateId = NewType('TemplateId', str)
CreationToken = NewType('CreationToken', str)

class MacroCreationStatus(Enum):
    """Macro creation status tracking."""
    VALIDATING = "validating"
    CREATING = "creating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
```

#### src/integration/km_client.py - Add Creation Methods
```python
def create_macro_applescript(
    self, 
    name: str, 
    group_id: Optional[str] = None,
    actions: List[Dict[str, Any]] = None,
    triggers: List[Dict[str, Any]] = None
) -> Either[KMError, str]:
    """Create macro via AppleScript with comprehensive error handling."""
    pass

def validate_macro_creation(self, request: MacroCreationRequest) -> bool:
    """Validate macro creation request against KM constraints."""
    pass
```

## 🏗️ Modularity Strategy
- **src/creation/**: New directory for macro creation functionality (<250 lines each)
- **macro_builder.py**: Core creation engine and validation (240 lines)
- **templates.py**: Template system with generators (220 lines)
- **src/server/tools/creation_tools.py**: MCP tool implementation (180 lines)
- **Enhance existing files**: Minimal additions to types.py and km_client.py

## 🔒 Security Implementation
1. **Input Sanitization**: ASCII-only names, restricted character sets, length limits
2. **Template Validation**: Each template validates its specific parameters
3. **AppleScript Security**: Escape all user inputs, validate generated AppleScript
4. **Permission Checking**: Verify user has macro creation permissions
5. **Rollback Safety**: Automatic rollback on creation failures

## 📊 Performance Targets
- **Validation Time**: <100ms for parameter validation
- **Creation Time**: <2 seconds for simple templates
- **Complex Templates**: <5 seconds for multi-action templates
- **Memory Usage**: <10MB for template processing

## ✅ Success Criteria
- [x] All advanced techniques implemented (contracts, defensive programming, type safety)
- [x] Complete security validation with injection prevention
- [x] Macro creation with template system supporting 5+ common automation patterns
- [x] Interactive macro editing with real-time modification capabilities
- [x] Step-through debugging with breakpoints and variable monitoring
- [x] Real-time validation during editing and debugging
- [x] Preview system for changes before applying
- [x] Real Keyboard Maestro integration (no mock data)
- [x] Comprehensive error handling with rollback capability
- [x] Property-based testing covers all creation and editing scenarios
- [x] Performance meets sub-5-second targets for all operations
- [x] Integration with existing MCP framework complete
- [ ] TESTING.md updated with comprehensive macro management test status
- [x] Full documentation with creation, editing, and debugging examples

## 🎨 Usage Examples

### Macro Creation
```python
# Create simple hotkey macro
result = await client.call_tool("km_macro_manager", {
    "operation": "create",
    "name": "Quick Notes",
    "template": "hotkey_action",
    "parameters": {
        "hotkey": "Cmd+Shift+N",
        "action": "open_app",
        "app_name": "Notes"
    }
})

# Create file processing macro
result = await client.call_tool("km_macro_manager", {
    "operation": "create",
    "name": "Process Screenshots",
    "template": "file_processor",
    "parameters": {
        "group_name": "File Automation",
        "watch_folder": "~/Desktop",
        "file_pattern": "*.png",
        "action_chain": ["resize", "optimize", "move"],
        "destination": "~/Documents/Screenshots"
    }
})
```

### Interactive Macro Editing
```python
# Start debugging session
result = await client.call_tool("km_macro_manager", {
    "operation": "debug",
    "macro_identifier": "550e8400-e29b-41d4-a716-446655440000",
    "debug_options": {
        "breakpoints": [1, 5, 10],
        "step_mode": "manual",
        "variable_watch": ["counter", "filename"]
    }
})

# Modify existing macro
result = await client.call_tool("km_macro_manager", {
    "operation": "edit",
    "macro_identifier": "My Macro",
    "parameters": {
        "enabled": False,
        "add_action": {
            "type": "pause",
            "duration": 2.0,
            "position": 3
        }
    }
})

# Validate macro before deployment
result = await client.call_tool("km_macro_manager", {
    "operation": "validate",
    "macro_identifier": "Production Workflow"
})

# Preview changes without applying
result = await client.call_tool("km_macro_manager", {
    "operation": "preview",
    "macro_identifier": "Test Macro",
    "parameters": {
        "rename": "Updated Test Macro",
        "modify_action": {
            "action_id": 2,
            "new_text": "Updated text content"
        }
    }
})
```

## 🧪 Testing Strategy
- **Property-Based Testing**: Validate creation with random valid inputs
- **Security Testing**: Attempt injection attacks, malformed inputs
- **Integration Testing**: Test with real Keyboard Maestro instances
- **Template Testing**: Verify each template generates valid macros
- **Error Recovery Testing**: Test rollback scenarios and error handling