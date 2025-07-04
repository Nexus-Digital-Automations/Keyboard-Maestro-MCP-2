# TASK_29: km_action_sequence_builder - Drag-and-Drop Action Composition

**Created By**: Agent_1 (Advanced Macro Creation Enhancement) | **Priority**: HIGH | **Duration**: 4 hours
**Technique Focus**: Builder Pattern + Functional Programming + Type Safety + Property-Based Testing
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_21 (km_add_condition), TASK_22 (km_control_flow), TASK_28 (km_macro_editor)
**Blocking**: Visual macro composition workflows requiring intuitive action sequencing

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Builder pattern specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Action types and composition patterns
- [ ] **Foundation Architecture**: src/server/tools/creation_tools.py - Existing action creation patterns
- [ ] **Builder Patterns**: src/actions/action_builder.py - Action builder implementation
- [ ] **Type System**: src/core/types.py - Branded types for actions and sequences
- [ ] **Testing Framework**: tests/TESTING.md - Property-based testing requirements

## 🎯 Problem Analysis
**Classification**: Missing User Experience Enhancement
**Gap Identified**: No intuitive visual composition interface for building complex action sequences
**Impact**: AI must understand low-level action configuration - cannot provide user-friendly macro building

<thinking>
Root Cause Analysis:
1. Current tools require detailed knowledge of KM action XML structure
2. No abstraction layer for composing complex action sequences
3. Missing visual/intuitive workflow for building sophisticated macros
4. Cannot easily reorder, group, or organize actions in logical workflows
5. No drag-and-drop style composition that mirrors KM Editor experience
6. Essential for making AI macro creation accessible and intuitive
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Action sequence types**: Define branded types for action sequences and compositions
- [ ] **Builder framework**: Fluent API for intuitive action sequence construction
- [ ] **Validation system**: Action compatibility and dependency validation

### Phase 2: Core Sequence Building
- [ ] **Action catalog**: Complete registry of available KM actions with metadata
- [ ] **Sequence composition**: Drag-and-drop style action ordering and grouping
- [ ] **Action templates**: Pre-configured action templates for common operations
- [ ] **Dependency resolution**: Automatic variable and resource dependency management

### Phase 3: Advanced Composition Features
- [ ] **Action groups**: Logical grouping of related actions with collapse/expand
- [ ] **Parallel execution**: Define actions that can run concurrently
- [ ] **Error handling**: Built-in error recovery and retry logic for action sequences
- [ ] **Performance optimization**: Automatic action ordering for optimal execution

### Phase 4: Visual Workflow Features
- [ ] **Sequence visualization**: Text-based representation of action workflow
- [ ] **Action relationships**: Visualize data flow and dependencies between actions
- [ ] **Validation feedback**: Real-time validation with helpful error messages
- [ ] **Sequence preview**: Preview generated macro before creation

### Phase 5: Integration & Testing
- [ ] **TESTING.md update**: Real-time test status and coverage tracking
- [ ] **Security validation**: Prevent malicious action sequences and resource abuse
- [ ] **Property-based tests**: Hypothesis validation for all composition operations
- [ ] **Integration tests**: Verify compatibility with macro editor and other tools

## 🔧 Implementation Files & Specifications
```
src/server/tools/action_sequence_tools.py       # Main action sequence builder tool
src/core/action_sequence.py                     # Action sequence type definitions
src/composition/sequence_builder.py             # Fluent builder API implementation
src/composition/action_catalog.py               # Complete KM action registry
tests/tools/test_action_sequence_tools.py       # Unit and integration tests
tests/property_tests/test_action_sequence.py    # Property-based composition validation
```

### km_action_sequence_builder Tool Specification
```python
@mcp.tool()
async def km_action_sequence_builder(
    operation: str,                             # create|modify|validate|preview|catalog
    sequence_name: Optional[str] = None,        # Name for the action sequence
    actions: Optional[List[Dict]] = None,       # Action specifications
    execution_mode: str = "sequential",         # sequential|parallel|conditional
    error_handling: str = "continue",           # continue|stop|retry
    optimization_level: str = "standard",       # none|standard|aggressive
    validation_strict: bool = True,             # Enable strict validation
    generate_preview: bool = False,             # Generate workflow preview
    ctx = None
) -> Dict[str, Any]:
```

### Action Sequence Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum

class ActionCategory(Enum):
    """KM Action categories for organization."""
    TEXT_MANIPULATION = "text_manipulation"
    FILE_OPERATIONS = "file_operations"
    APPLICATION_CONTROL = "application_control"
    SYSTEM_INTERACTION = "system_interaction"
    VARIABLE_MANAGEMENT = "variable_management"
    CONTROL_FLOW = "control_flow"
    USER_INTERFACE = "user_interface"
    NETWORK_OPERATIONS = "network_operations"
    MEDIA_PROCESSING = "media_processing"
    AUTOMATION_TOOLS = "automation_tools"

class ExecutionMode(Enum):
    """Action execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    BACKGROUND = "background"

@dataclass(frozen=True)
class ActionSpec:
    """Type-safe action specification."""
    action_type: str
    action_id: str
    category: ActionCategory
    config: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    timeout_seconds: int = 30
    retry_count: int = 0
    
    @require(lambda self: len(self.action_type) > 0)
    @require(lambda self: len(self.action_id) > 0)
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    @require(lambda self: self.retry_count >= 0 and self.retry_count <= 5)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class ActionSequence:
    """Complete action sequence specification."""
    sequence_name: str
    actions: List[ActionSpec]
    execution_mode: ExecutionMode
    error_handling: str
    estimated_duration: float
    complexity_score: int
    
    @require(lambda self: len(self.sequence_name) > 0)
    @require(lambda self: len(self.actions) > 0)
    @require(lambda self: len(self.actions) <= 100)  # Reasonable limit
    def __post_init__(self):
        pass
    
    def get_action_count(self) -> int:
        return len(self.actions)
    
    def get_categories_used(self) -> Set[ActionCategory]:
        return {action.category for action in self.actions}
    
    def get_dependencies(self) -> Set[str]:
        return set().union(*(action.dependencies for action in self.actions))

@dataclass(frozen=True)
class ActionTemplate:
    """Reusable action template."""
    template_name: str
    description: str
    action_type: str
    default_config: Dict[str, Any]
    required_parameters: Set[str]
    optional_parameters: Set[str]
    category: ActionCategory
    
    @require(lambda self: len(self.template_name) > 0)
    @require(lambda self: len(self.action_type) > 0)
    def __post_init__(self):
        pass

class ActionSequenceBuilder:
    """Fluent API for building action sequences."""
    
    def __init__(self, sequence_name: str):
        self.sequence_name = sequence_name
        self._actions: List[ActionSpec] = []
        self._execution_mode = ExecutionMode.SEQUENTIAL
        self._error_handling = "continue"
    
    def add_text_action(self, action_type: str, text: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add text manipulation action."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"text_{len(self._actions)}",
            category=ActionCategory.TEXT_MANIPULATION,
            config={"text": text, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_file_action(self, action_type: str, file_path: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add file operation action."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"file_{len(self._actions)}",
            category=ActionCategory.FILE_OPERATIONS,
            config={"file_path": file_path, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_app_action(self, action_type: str, app_name: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add application control action."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"app_{len(self._actions)}",
            category=ActionCategory.APPLICATION_CONTROL,
            config={"application": app_name, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_variable_action(self, action_type: str, variable_name: str, value: str = None, **kwargs) -> 'ActionSequenceBuilder':
        """Add variable management action."""
        config = {"variable_name": variable_name, **kwargs}
        if value is not None:
            config["value"] = value
        
        action = ActionSpec(
            action_type=action_type,
            action_id=f"var_{len(self._actions)}",
            category=ActionCategory.VARIABLE_MANAGEMENT,
            config=config
        )
        self._actions.append(action)
        return self
    
    def add_condition(self, condition_type: str, condition_config: Dict) -> 'ActionSequenceBuilder':
        """Add conditional logic action."""
        action = ActionSpec(
            action_type=f"condition_{condition_type}",
            action_id=f"cond_{len(self._actions)}",
            category=ActionCategory.CONTROL_FLOW,
            config=condition_config
        )
        self._actions.append(action)
        return self
    
    def add_custom_action(self, action_spec: ActionSpec) -> 'ActionSequenceBuilder':
        """Add custom action specification."""
        self._actions.append(action_spec)
        return self
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'ActionSequenceBuilder':
        """Set sequence execution mode."""
        self._execution_mode = mode
        return self
    
    def set_error_handling(self, handling: str) -> 'ActionSequenceBuilder':
        """Set error handling strategy."""
        self._error_handling = handling
        return self
    
    def insert_action(self, position: int, action_spec: ActionSpec) -> 'ActionSequenceBuilder':
        """Insert action at specific position."""
        if 0 <= position <= len(self._actions):
            self._actions.insert(position, action_spec)
        return self
    
    def remove_action(self, action_id: str) -> 'ActionSequenceBuilder':
        """Remove action by ID."""
        self._actions = [a for a in self._actions if a.action_id != action_id]
        return self
    
    def reorder_actions(self, new_order: List[str]) -> 'ActionSequenceBuilder':
        """Reorder actions by ID list."""
        id_to_action = {a.action_id: a for a in self._actions}
        self._actions = [id_to_action[aid] for aid in new_order if aid in id_to_action]
        return self
    
    def build(self) -> ActionSequence:
        """Build the final action sequence."""
        estimated_duration = sum(action.timeout_seconds for action in self._actions)
        complexity_score = self._calculate_complexity()
        
        return ActionSequence(
            sequence_name=self.sequence_name,
            actions=self._actions.copy(),
            execution_mode=self._execution_mode,
            error_handling=self._error_handling,
            estimated_duration=estimated_duration,
            complexity_score=complexity_score
        )
    
    def _calculate_complexity(self) -> int:
        """Calculate sequence complexity score."""
        base_score = len(self._actions)
        category_variety = len({a.category for a in self._actions})
        dependency_count = sum(len(a.dependencies) for a in self._actions)
        
        return base_score + (category_variety * 2) + dependency_count
```

## 🔒 Security Implementation
```python
class ActionSequenceValidator:
    """Security-first action sequence validation."""
    
    @staticmethod
    def validate_sequence_limits(sequence: ActionSequence) -> Either[SecurityError, None]:
        """Validate sequence doesn't exceed resource limits."""
        # Check action count limits
        if len(sequence.actions) > 100:
            return Either.left(SecurityError("Too many actions in sequence"))
        
        # Check estimated duration
        if sequence.estimated_duration > 1800:  # 30 minutes
            return Either.left(SecurityError("Sequence duration too long"))
        
        # Check complexity limits
        if sequence.complexity_score > 200:
            return Either.left(SecurityError("Sequence too complex"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_action_config(action: ActionSpec) -> Either[SecurityError, ActionSpec]:
        """Validate individual action configuration."""
        # Sanitize file paths
        if "file_path" in action.config:
            path_result = validate_file_path(action.config["file_path"])
            if path_result.is_left():
                return Either.left(SecurityError("Invalid file path in action"))
        
        # Validate script content
        if "script" in action.config:
            script = action.config["script"]
            if contains_dangerous_patterns(script):
                return Either.left(SecurityError("Dangerous script in action"))
        
        # Check timeout limits
        if action.timeout_seconds > 300:
            return Either.left(SecurityError("Action timeout too long"))
        
        return Either.right(action)
    
    @staticmethod
    def validate_dependencies(sequence: ActionSequence) -> Either[SecurityError, None]:
        """Validate action dependencies are resolvable."""
        available_outputs = set()
        
        for action in sequence.actions:
            # Check if dependencies are available
            missing_deps = action.dependencies - available_outputs
            if missing_deps:
                return Either.left(SecurityError(f"Unresolved dependencies: {missing_deps}"))
            
            # Add this action's outputs
            available_outputs.update(action.outputs)
        
        return Either.right(None)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=50))
def test_sequence_name_properties(sequence_name):
    """Property: All valid sequence names should be accepted."""
    if is_valid_sequence_name(sequence_name):
        builder = ActionSequenceBuilder(sequence_name)
        assert builder.sequence_name == sequence_name
    else:
        with pytest.raises(ValueError):
            ActionSequenceBuilder(sequence_name)

@given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
def test_text_action_properties(text_values):
    """Property: Text actions should handle various text inputs."""
    builder = ActionSequenceBuilder("test_sequence")
    for text in text_values:
        builder.add_text_action("type_text", text)
    
    sequence = builder.build()
    assert len(sequence.actions) == len(text_values)
    assert all(action.category == ActionCategory.TEXT_MANIPULATION for action in sequence.actions)

@given(st.integers(min_value=0, max_value=100))
def test_action_count_limits(action_count):
    """Property: Action sequences should respect count limits."""
    builder = ActionSequenceBuilder("test_sequence")
    for i in range(action_count):
        builder.add_text_action("type_text", f"text_{i}")
    
    if action_count <= 100:
        sequence = builder.build()
        assert len(sequence.actions) == action_count
        validation_result = ActionSequenceValidator.validate_sequence_limits(sequence)
        assert validation_result.is_right()
    else:
        # Should be rejected by validation
        sequence = builder.build()
        validation_result = ActionSequenceValidator.validate_sequence_limits(sequence)
        assert validation_result.is_left()
```

## 🏗️ Modularity Strategy
- **action_sequence_tools.py**: Main MCP tool interface (<250 lines)
- **action_sequence.py**: Type definitions and core logic (<300 lines)
- **sequence_builder.py**: Fluent builder API implementation (<250 lines)
- **action_catalog.py**: Complete KM action registry (<400 lines)

## 📋 Action Sequence Examples

### Text Processing Workflow
```python
# Example: Build a text processing sequence
builder = ActionSequenceBuilder("Text Processing Workflow")
sequence = builder \
    .add_text_action("get_clipboard", "") \
    .add_text_action("search_replace", "oldtext", replacement="newtext", use_regex=False) \
    .add_text_action("change_case", "title_case") \
    .add_variable_action("set_variable", "ProcessedText", "%LastResult%") \
    .add_text_action("set_clipboard", "%Variable%ProcessedText%") \
    .set_execution_mode(ExecutionMode.SEQUENTIAL) \
    .set_error_handling("continue") \
    .build()

result = await km_action_sequence_builder(
    operation="create",
    sequence_name="Text Processing Workflow",
    actions=[action.__dict__ for action in sequence.actions]
)
```

### File Processing Pipeline
```python
# Example: Build a file processing pipeline
builder = ActionSequenceBuilder("Document Processing Pipeline")
sequence = builder \
    .add_file_action("select_files", "~/Documents/*.pdf") \
    .add_variable_action("set_variable", "FileCount", "%FileCount%") \
    .add_condition("for_each", {"collection": "SelectedFiles", "variable": "CurrentFile"}) \
    .add_file_action("open_file", "%Variable%CurrentFile%") \
    .add_text_action("extract_text", "") \
    .add_file_action("save_text", "~/ProcessedText/%FileName%.txt") \
    .add_app_action("close_window", "Preview") \
    .set_execution_mode(ExecutionMode.SEQUENTIAL) \
    .build()
```

### Application Automation Sequence
```python
# Example: Build application automation sequence
builder = ActionSequenceBuilder("Email Processing Automation")
sequence = builder \
    .add_app_action("activate_application", "Mail") \
    .add_app_action("select_menu", "Mailbox", "Get All New Mail") \
    .add_variable_action("set_variable", "NewEmailCount", "%EmailCount%") \
    .add_condition("if_then", {"condition": "%Variable%NewEmailCount% > 0"}) \
    .add_app_action("select_menu", "File", "Export", "As PDF") \
    .add_file_action("save_file", "~/Email Backups/%LongDate%.pdf") \
    .set_error_handling("retry") \
    .build()
```

## ✅ Success Criteria
- Complete action sequence builder with intuitive fluent API
- Comprehensive action catalog covering all major KM action types
- Advanced composition features (grouping, dependencies, parallel execution)
- Comprehensive security validation prevents malicious sequences and resource abuse
- Property-based tests validate behavior across all composition scenarios
- Integration with condition system (TASK_21) and control flow (TASK_22)
- Performance: <100ms sequence building, <500ms validation, <1s preview generation
- Documentation: Complete API documentation with examples and best practices
- TESTING.md shows 95%+ test coverage with all security and functionality tests passing
- Tool enables AI to build sophisticated macro workflows through intuitive composition

## 🔄 Integration Points
- **TASK_21 (km_add_condition)**: Include conditional logic in action sequences
- **TASK_22 (km_control_flow)**: Integrate control flow structures
- **TASK_28 (km_macro_editor)**: Edit sequences built with this tool
- **TASK_10 (km_create_macro)**: Generate macros from action sequences
- **All Existing Tools**: Create sequences using any available action type
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This provides the intuitive "drag-and-drop" style macro building experience
- Essential for making AI macro creation accessible to users of all skill levels
- Builder pattern enables complex composition while maintaining simplicity
- Security is critical - sequences can combine multiple potentially dangerous actions
- Must maintain functional programming patterns for testability and composability
- Success here enables sophisticated workflow creation through simple, intuitive composition