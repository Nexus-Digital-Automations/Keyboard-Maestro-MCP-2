# TASK_28: km_macro_editor - Interactive Macro Modification and Debugging

**Created By**: Agent_1 (Advanced Macro Creation Enhancement) | **Priority**: HIGH | **Duration**: 5 hours
**Technique Focus**: Design by Contract + Type Safety + Defensive Programming + Property-Based Testing
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_21 (km_add_condition), TASK_22 (km_control_flow), TASK_23 (km_create_trigger_advanced)
**Blocking**: Advanced macro editing workflows requiring interactive modification

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Interactive tool specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Macro modification and debugging capabilities
- [ ] **Foundation Architecture**: src/server/tools/creation_tools.py - Existing macro creation patterns
- [ ] **Condition System**: development/tasks/TASK_21.md - Conditional logic integration
- [ ] **Control Flow**: development/tasks/TASK_22.md - Control flow modification patterns
- [ ] **Type System**: src/core/types.py - Branded types for macro components
- [ ] **Testing Framework**: tests/TESTING.md - Property-based testing requirements

## 🎯 Problem Analysis
**Classification**: Missing Critical Functionality
**Gap Identified**: No interactive macro editing capabilities - only basic creation and execution
**Impact**: AI limited to creating new macros - cannot modify, debug, or refine existing workflows

<thinking>
Root Cause Analysis:
1. Current tools focus on macro creation but not modification
2. No debugging capabilities for complex macros with conditions and control flow
3. Missing interactive editing for refining macro behavior
4. No visual inspection of macro structure and logic
5. Cannot modify existing macros without recreation
6. Essential for iterative macro development and refinement
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Macro editor types**: Define branded types for all editing operations
- [ ] **Validation framework**: Input sanitization and security boundaries for editing
- [ ] **State management**: Track macro modification history and rollback capabilities

### Phase 2: Core Editing Operations
- [ ] **Action modification**: Insert, update, delete, reorder actions in macros
- [ ] **Condition editing**: Modify conditional logic with visual validation
- [ ] **Trigger management**: Add, remove, and modify macro triggers
- [ ] **Property editing**: Update macro name, group, enabled state, notes

### Phase 3: Interactive Debugging
- [ ] **Macro inspection**: Detailed view of macro structure and components
- [ ] **Step-through debugging**: Execute macro step-by-step with breakpoints
- [ ] **Variable monitoring**: Watch KM variables during macro execution
- [ ] **Error analysis**: Detailed error reporting and suggested fixes

### Phase 4: Advanced Features
- [ ] **Macro comparison**: Diff two macros showing differences
- [ ] **Version history**: Track macro changes with rollback capability
- [ ] **Performance analysis**: Timing analysis and optimization suggestions
- [ ] **Validation engine**: Comprehensive macro validation and health checks

### Phase 5: Integration & Testing
- [ ] **TESTING.md update**: Real-time test status and coverage tracking
- [ ] **Security validation**: Prevent unauthorized macro modifications
- [ ] **Property-based tests**: Hypothesis validation for all editing operations
- [ ] **Integration tests**: Verify compatibility with existing macro tools

## 🔧 Implementation Files & Specifications
```
src/server/tools/macro_editor_tools.py          # Main macro editor tool implementation
src/core/macro_editor.py                        # Macro editing type definitions and operations
src/integration/km_macro_editor.py              # KM-specific editor integration
src/debugging/macro_debugger.py                 # Interactive debugging functionality
tests/tools/test_macro_editor_tools.py          # Unit and integration tests
tests/property_tests/test_macro_editor.py       # Property-based editor validation
```

### km_macro_editor Tool Specification
```python
@mcp.tool()
async def km_macro_editor(
    macro_identifier: str,                       # Target macro (name or UUID)
    operation: str,                             # inspect|modify|debug|compare|validate
    modification_spec: Optional[Dict] = None,    # Detailed modification instructions
    debug_options: Optional[Dict] = None,        # Debugging configuration
    comparison_target: Optional[str] = None,     # For macro comparison
    validation_level: str = "standard",         # standard|strict|comprehensive
    create_backup: bool = True,                 # Backup before modifications
    ctx = None
) -> Dict[str, Any]:
```

### Macro Editing Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum

class EditOperation(Enum):
    """Supported macro editing operations."""
    INSPECT = "inspect"
    MODIFY_ACTION = "modify_action"
    ADD_ACTION = "add_action"
    DELETE_ACTION = "delete_action"
    REORDER_ACTIONS = "reorder_actions"
    MODIFY_CONDITION = "modify_condition"
    ADD_TRIGGER = "add_trigger"
    REMOVE_TRIGGER = "remove_trigger"
    UPDATE_PROPERTIES = "update_properties"
    DEBUG_EXECUTE = "debug_execute"
    COMPARE_MACROS = "compare_macros"
    VALIDATE_MACRO = "validate_macro"

@dataclass(frozen=True)
class MacroModification:
    """Type-safe macro modification specification."""
    operation: EditOperation
    target_element: Optional[str] = None        # Action UUID, trigger ID, etc.
    new_value: Optional[Dict[str, Any]] = None  # New configuration
    position: Optional[int] = None              # For reordering or insertion
    
    @require(lambda self: self.operation in EditOperation)
    @require(lambda self: self.position is None or self.position >= 0)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class MacroInspection:
    """Comprehensive macro inspection result."""
    macro_id: str
    macro_name: str
    enabled: bool
    group_name: str
    action_count: int
    trigger_count: int
    condition_count: int
    actions: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    variables_used: Set[str]
    estimated_execution_time: float
    complexity_score: int
    health_score: int

@dataclass(frozen=True)
class DebugSession:
    """Interactive debugging session configuration."""
    macro_id: str
    breakpoints: Set[str] = field(default_factory=set)  # Action UUIDs
    watch_variables: Set[str] = field(default_factory=set)  # Variable names
    step_mode: bool = False
    timeout_seconds: int = 60
    
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class MacroComparison:
    """Result of comparing two macros."""
    macro1_id: str
    macro2_id: str
    differences: List[Dict[str, Any]]
    similarity_score: float
    recommendation: str

class MacroEditor:
    """Fluent API for macro editing operations."""
    
    def __init__(self, macro_id: str):
        self.macro_id = macro_id
        self._modifications: List[MacroModification] = []
    
    def add_action(self, action_type: str, config: Dict, position: Optional[int] = None) -> 'MacroEditor':
        """Add new action to macro."""
        mod = MacroModification(
            operation=EditOperation.ADD_ACTION,
            new_value={"type": action_type, "config": config},
            position=position
        )
        self._modifications.append(mod)
        return self
    
    def modify_action(self, action_id: str, new_config: Dict) -> 'MacroEditor':
        """Modify existing action."""
        mod = MacroModification(
            operation=EditOperation.MODIFY_ACTION,
            target_element=action_id,
            new_value=new_config
        )
        self._modifications.append(mod)
        return self
    
    def delete_action(self, action_id: str) -> 'MacroEditor':
        """Delete action from macro."""
        mod = MacroModification(
            operation=EditOperation.DELETE_ACTION,
            target_element=action_id
        )
        self._modifications.append(mod)
        return self
    
    def reorder_actions(self, new_order: List[str]) -> 'MacroEditor':
        """Reorder actions in macro."""
        mod = MacroModification(
            operation=EditOperation.REORDER_ACTIONS,
            new_value={"action_order": new_order}
        )
        self._modifications.append(mod)
        return self
    
    def add_condition(self, condition_type: str, config: Dict) -> 'MacroEditor':
        """Add conditional logic to macro."""
        mod = MacroModification(
            operation=EditOperation.MODIFY_CONDITION,
            new_value={"type": condition_type, "config": config}
        )
        self._modifications.append(mod)
        return self
    
    def add_trigger(self, trigger_type: str, config: Dict) -> 'MacroEditor':
        """Add trigger to macro."""
        mod = MacroModification(
            operation=EditOperation.ADD_TRIGGER,
            new_value={"type": trigger_type, "config": config}
        )
        self._modifications.append(mod)
        return self
    
    def update_properties(self, properties: Dict[str, Any]) -> 'MacroEditor':
        """Update macro properties."""
        mod = MacroModification(
            operation=EditOperation.UPDATE_PROPERTIES,
            new_value=properties
        )
        self._modifications.append(mod)
        return self
    
    def get_modifications(self) -> List[MacroModification]:
        """Get all pending modifications."""
        return self._modifications.copy()
```

## 🔒 Security Implementation
```python
class MacroEditorValidator:
    """Security-first macro editing validation."""
    
    @staticmethod
    def validate_modification_permissions(macro_id: str, operation: EditOperation) -> Either[SecurityError, None]:
        """Validate user has permission to modify macro."""
        # Check if macro exists and is editable
        if not macro_exists(macro_id):
            return Either.left(SecurityError("Macro not found"))
        
        # Check for read-only macros
        if is_system_macro(macro_id):
            return Either.left(SecurityError("Cannot modify system macro"))
        
        # Validate operation type
        dangerous_operations = [EditOperation.DEBUG_EXECUTE]
        if operation in dangerous_operations:
            if not has_debug_permission():
                return Either.left(SecurityError("Debug permission required"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_action_modification(action_config: Dict) -> Either[SecurityError, Dict]:
        """Prevent malicious action modifications."""
        # Sanitize script content
        if "script" in action_config:
            script = action_config["script"]
            if contains_dangerous_patterns(script):
                return Either.left(SecurityError("Dangerous script content detected"))
        
        # Validate file paths
        if "file_path" in action_config:
            path_result = validate_file_path(action_config["file_path"])
            if path_result.is_left():
                return path_result
        
        # Limit action complexity
        if calculate_action_complexity(action_config) > 100:
            return Either.left(SecurityError("Action too complex"))
        
        return Either.right(action_config)
    
    @staticmethod
    def validate_debug_session(debug_config: Dict) -> Either[SecurityError, None]:
        """Prevent abuse of debugging capabilities."""
        # Limit breakpoint count
        breakpoints = debug_config.get("breakpoints", [])
        if len(breakpoints) > 50:
            return Either.left(SecurityError("Too many breakpoints"))
        
        # Validate timeout
        timeout = debug_config.get("timeout_seconds", 60)
        if timeout > 300:
            return Either.left(SecurityError("Debug timeout too long"))
        
        # Check watch variable count
        watch_vars = debug_config.get("watch_variables", [])
        if len(watch_vars) > 20:
            return Either.left(SecurityError("Too many watch variables"))
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_macro_identifier_properties(macro_id):
    """Property: All valid macro identifiers should be accepted."""
    if is_valid_macro_id(macro_id):
        editor = MacroEditor(macro_id)
        assert editor.macro_id == macro_id
    else:
        with pytest.raises(ValueError):
            MacroEditor(macro_id)

@given(st.lists(st.dictionaries(st.text(), st.text()), min_size=1, max_size=10))
def test_action_modification_properties(action_configs):
    """Property: Action modifications should preserve macro structure."""
    editor = MacroEditor("test_macro")
    for i, config in enumerate(action_configs):
        editor.add_action("test_action", config, position=i)
    
    modifications = editor.get_modifications()
    assert len(modifications) == len(action_configs)
    assert all(mod.operation == EditOperation.ADD_ACTION for mod in modifications)

@given(st.integers(min_value=1, max_value=50))
def test_debug_breakpoint_properties(breakpoint_count):
    """Property: Debug sessions should handle reasonable breakpoint counts."""
    breakpoints = {f"action_{i}" for i in range(breakpoint_count)}
    
    if breakpoint_count <= 50:
        debug_session = DebugSession("test_macro", breakpoints=breakpoints)
        assert len(debug_session.breakpoints) == breakpoint_count
    else:
        validation_result = MacroEditorValidator.validate_debug_session({
            "breakpoints": list(breakpoints)
        })
        assert validation_result.is_left()
```

## 🏗️ Modularity Strategy
- **macro_editor_tools.py**: Main MCP tool interface (<250 lines)
- **macro_editor.py**: Type definitions, operations, and validation (<300 lines)
- **km_macro_editor.py**: KM integration and AppleScript generation (<250 lines)
- **macro_debugger.py**: Interactive debugging functionality (<200 lines)

## 📋 Interactive Editing Examples

### Basic Action Modification
```python
# Example: Add notification action to existing macro
editor = MacroEditor("Daily Backup Macro")
editor.add_action("display_notification", {
    "title": "Backup Complete",
    "subtitle": "Daily backup finished successfully",
    "sound": "Glass"
}, position=0)  # Insert at beginning

result = await km_macro_editor(
    macro_identifier="Daily Backup Macro",
    operation="modify",
    modification_spec=editor.get_modifications()
)
```

### Interactive Debugging Session
```python
# Example: Debug macro with step-through execution
debug_session = DebugSession(
    macro_id="Complex Workflow",
    breakpoints={"action_uuid_1", "action_uuid_5"},
    watch_variables={"ProcessedFiles", "ErrorCount"},
    step_mode=True
)

result = await km_macro_editor(
    macro_identifier="Complex Workflow",
    operation="debug",
    debug_options=debug_session.__dict__
)
```

### Macro Comparison and Analysis
```python
# Example: Compare two similar macros
result = await km_macro_editor(
    macro_identifier="Backup Macro V1",
    operation="compare",
    comparison_target="Backup Macro V2",
    validation_level="comprehensive"
)

# Access comparison results
differences = result["data"]["comparison"]["differences"]
similarity = result["data"]["comparison"]["similarity_score"]
```

## ✅ Success Criteria
- Complete macro editing implementation with all core operations (inspect, modify, debug, compare)
- Interactive debugging with breakpoints, variable watching, and step-through execution
- Comprehensive security validation prevents unauthorized modifications and malicious content
- Property-based tests validate behavior across all editing scenarios
- Integration with condition system (TASK_21) and control flow (TASK_22) for complex editing
- Performance: <100ms for inspection, <500ms for modifications, <2s for debugging setup
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and functionality tests passing
- Tool enables AI to iteratively refine and perfect macro workflows through interactive editing

## 🔄 Integration Points
- **TASK_21 (km_add_condition)**: Edit conditional logic within macros
- **TASK_22 (km_control_flow)**: Modify control flow structures
- **TASK_23 (km_create_trigger_advanced)**: Add/remove advanced triggers
- **TASK_10 (km_create_macro)**: Edit newly created macros
- **All Existing Tools**: Enhance any macro created by existing tools
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This enables iterative macro development beyond simple creation
- Essential for refining and perfecting AI-created automation workflows
- Interactive debugging enables understanding macro behavior and troubleshooting
- Security is critical - editing can modify existing user workflows
- Must maintain functional programming patterns for testability and reliability
- Success here transforms macro development from one-shot creation to iterative refinement