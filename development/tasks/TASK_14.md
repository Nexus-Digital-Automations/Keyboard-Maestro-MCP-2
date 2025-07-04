# TASK_14: km_action_builder - Advanced Action Construction & Sequence Composition

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 6 hours
**Technique Focus**: Builder Pattern + Functional Programming + XML Security + Contract Validation + Component Assembly
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 📋 **MERGED FUNCTIONALITY**
This task combines programmatic action construction (original TASK_14) with intuitive action sequence composition (TASK_29) for comprehensive macro building capabilities.

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_7
**Dependencies**: TASK_10 (km_create_macro foundation)
**Blocking**: TASK_15 (hotkey triggers use action building)

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_add_action specification (lines 657-672)
- [x] **src/creation/**: Macro creation patterns from TASK_10 implementation
- [x] **Keyboard Maestro Action XML**: Understanding action XML structure and validation
- [x] **src/core/types.py**: Branded types and validation patterns
- [x] **tests/TESTING.md**: Current testing framework and XML security requirements

## 🎯 Implementation Overview
Create a comprehensive action building and sequence composition system that enables AI assistants to:

1. **Programmatic Action Construction**: Add individual actions to existing macros with XML generation
2. **Intuitive Sequence Composition**: Build complex action sequences using fluent API
3. **Visual Workflow Building**: Drag-and-drop style action composition and ordering
4. **Advanced Composition Features**: Action grouping, dependencies, parallel execution

<thinking>
Action building and sequence composition is the foundation for sophisticated macro construction:
1. Security Critical: Must validate and sanitize all action XML to prevent injection
2. Builder Pattern: Fluent API for constructing complex action sequences with drag-and-drop style composition
3. XML Generation: Safe XML generation with proper escaping and validation
4. Action Registry: Support for 300+ Keyboard Maestro action types with comprehensive catalog
5. Sequence Composition: Intuitive workflow building with visual composition and dependency management
6. Validation Engine: Ensure action configurations and sequences are syntactically correct
7. Integration: Work with macro creation system for complete workflow building
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Action Building Infrastructure
- [ ] **Action types**: Define ActionType, ActionConfiguration, ActionBuilder, ActionSequence types
- [ ] **Sequence types**: Define ActionSpec, ActionSequence, ActionTemplate for composition
- [ ] **Security validation**: XML sanitization, action parameter validation, injection prevention
- [ ] **Builder pattern**: Fluent API for constructing individual actions and complete sequences
- [ ] **Action registry**: Comprehensive mapping of KM action types with metadata catalog

### Phase 2: XML Generation & Sequence Composition
- [ ] **XML generation**: Safe XML construction with proper escaping and formatting
- [ ] **Action templates**: Pre-built templates for common action types and sequences
- [ ] **Sequence composition**: Drag-and-drop style action ordering and grouping
- [ ] **Parameter validation**: Type-specific validation for action parameters and sequences
- [ ] **XML security**: Prevent XML injection and validate generated XML structure

### Phase 3: Advanced Composition Features
- [ ] **Core actions**: Text manipulation, system control, application actions
- [ ] **Flow control**: If/then/else, loops, switch statements, pause actions
- [ ] **Variable actions**: Set, get, calculate, increment/decrement variables
- [ ] **Advanced actions**: AppleScript, shell scripts, web requests
- [ ] **Action groups**: Logical grouping of related actions with collapse/expand
- [ ] **Parallel execution**: Define actions that can run concurrently
- [ ] **Dependency resolution**: Automatic variable and resource dependency management

### Phase 4: MCP Tool Integration
- [ ] **Tool implementation**: km_action_builder MCP tool with action and sequence support
- [ ] **Position management**: Insert actions at specific positions in macro
- [ ] **Sequence management**: Create, modify, validate, and preview action sequences
- [ ] **Response formatting**: Action/sequence results with XML preview and validation status
- [ ] **Testing integration**: Property-based tests for all action types and sequences

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/actions/action_builder.py - Core Action Building System
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from ..core.types import MacroId, Duration
from ..core.contracts import require, ensure

class ActionCategory(Enum):
    """Action categories for organization and validation."""
    TEXT = "text"
    APPLICATION = "application"
    FILE = "file"
    SYSTEM = "system"
    VARIABLE = "variable"
    CONTROL = "control"
    INTERFACE = "interface"
    WEB = "web"
    CALCULATION = "calculation"

@dataclass(frozen=True)
class ActionType:
    """Type-safe action type definition."""
    identifier: str
    category: ActionCategory
    required_params: List[str]
    optional_params: List[str] = field(default_factory=list)
    
    @require(lambda self: len(self.identifier) > 0 and self.identifier.replace("_", "").replace(" ", "").isalnum())
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class ActionConfiguration:
    """Type-safe action configuration with validation."""
    action_type: ActionType
    parameters: Dict[str, Any]
    position: Optional[int] = None
    enabled: bool = True
    timeout: Optional[Duration] = None
    
    @require(lambda self: all(param in self.parameters for param in self.action_type.required_params))
    @require(lambda self: all(isinstance(v, (str, int, float, bool)) for v in self.parameters.values()))
    def __post_init__(self):
        pass
    
    def validate_parameters(self) -> bool:
        """Validate action parameters against type requirements."""
        # Check required parameters are present
        for param in self.action_type.required_params:
            if param not in self.parameters:
                return False
        
        # Validate parameter types and formats
        return self._validate_parameter_types()
    
    def _validate_parameter_types(self) -> bool:
        """Validate parameter types for security and correctness."""
        # Implement type-specific validation
        return True

@dataclass(frozen=True)
class ActionSpec:
    """Type-safe action specification for sequence building."""
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
    execution_mode: str  # "sequential", "parallel", "conditional"
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

class ActionSequenceBuilder:
    """Fluent API for building complete action sequences."""
    
    def __init__(self, sequence_name: str):
        self.sequence_name = sequence_name
        self._actions: List[ActionSpec] = []
        self._execution_mode = "sequential"
        self._error_handling = "continue"
    
    def add_text_action(self, action_type: str, text: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add text manipulation action to sequence."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"text_{len(self._actions)}",
            category=ActionCategory.TEXT,
            config={"text": text, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_file_action(self, action_type: str, file_path: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add file operation action to sequence."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"file_{len(self._actions)}",
            category=ActionCategory.FILE,
            config={"file_path": file_path, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_app_action(self, action_type: str, app_name: str, **kwargs) -> 'ActionSequenceBuilder':
        """Add application control action to sequence."""
        action = ActionSpec(
            action_type=action_type,
            action_id=f"app_{len(self._actions)}",
            category=ActionCategory.APPLICATION,
            config={"application": app_name, **kwargs}
        )
        self._actions.append(action)
        return self
    
    def add_variable_action(self, action_type: str, variable_name: str, value: str = None, **kwargs) -> 'ActionSequenceBuilder':
        """Add variable management action to sequence."""
        config = {"variable_name": variable_name, **kwargs}
        if value is not None:
            config["value"] = value
        
        action = ActionSpec(
            action_type=action_type,
            action_id=f"var_{len(self._actions)}",
            category=ActionCategory.VARIABLE,
            config=config
        )
        self._actions.append(action)
        return self
    
    def add_condition(self, condition_type: str, condition_config: Dict) -> 'ActionSequenceBuilder':
        """Add conditional logic action to sequence."""
        action = ActionSpec(
            action_type=f"condition_{condition_type}",
            action_id=f"cond_{len(self._actions)}",
            category=ActionCategory.CONTROL,
            config=condition_config
        )
        self._actions.append(action)
        return self
    
    def set_execution_mode(self, mode: str) -> 'ActionSequenceBuilder':
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
        """Reorder actions by ID list (drag-and-drop style)."""
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

class ActionBuilder:
    """Fluent builder for constructing individual actions and complete sequences."""
    
    def __init__(self):
        self.actions: List[ActionConfiguration] = []
        self.action_registry = ActionRegistry()
        self.sequence_builders: Dict[str, ActionSequenceBuilder] = {}
    
    @require(lambda action_type: action_type.identifier != "")
    def add_action(self, action_type: str, parameters: Dict[str, Any], **kwargs) -> 'ActionBuilder':
        """Add action to sequence with parameter validation."""
        action_def = self.action_registry.get_action_type(action_type)
        if not action_def:
            raise ValueError(f"Unknown action type: {action_type}")
        
        config = ActionConfiguration(
            action_type=action_def,
            parameters=parameters,
            **kwargs
        )
        
        if not config.validate_parameters():
            raise ValueError(f"Invalid parameters for action {action_type}")
        
        self.actions.append(config)
        return self
    
    def add_text_action(self, text: str, **kwargs) -> 'ActionBuilder':
        """Convenience method for adding text input actions."""
        return self.add_action("Type a String", {"text": text}, **kwargs)
    
    def add_pause_action(self, duration: Duration, **kwargs) -> 'ActionBuilder':
        """Convenience method for adding pause actions."""
        return self.add_action("Pause", {"duration": duration.total_seconds()}, **kwargs)
    
    def add_if_action(self, condition: Dict[str, Any], **kwargs) -> 'ActionBuilder':
        """Convenience method for adding conditional actions."""
        return self.add_action("If Then Else", {"condition": condition}, **kwargs)
    
    def create_sequence_builder(self, sequence_name: str) -> ActionSequenceBuilder:
        """Create new action sequence builder."""
        builder = ActionSequenceBuilder(sequence_name)
        self.sequence_builders[sequence_name] = builder
        return builder
    
    def get_sequence_builder(self, sequence_name: str) -> Optional[ActionSequenceBuilder]:
        """Get existing sequence builder."""
        return self.sequence_builders.get(sequence_name)
    
    def build_sequence_xml(self, sequence: ActionSequence) -> Either[KMError, str]:
        """Generate XML for complete action sequence."""
        try:
            root = ET.Element("actions")
            root.set("sequence_name", sequence.sequence_name)
            root.set("execution_mode", sequence.execution_mode)
            
            for i, action_spec in enumerate(sequence.actions):
                action_elem = self._generate_sequence_action_xml(action_spec, i)
                root.append(action_elem)
            
            xml_string = ET.tostring(root, encoding='unicode')
            if not self._validate_xml_security(xml_string):
                return Either.left(KMError.validation_error("Generated sequence XML failed security validation"))
            
            return Either.right(xml_string)
        except Exception as e:
            return Either.left(KMError.execution_error(f"Sequence XML generation failed: {str(e)}"))
    
    def _generate_sequence_action_xml(self, action_spec: ActionSpec, index: int) -> ET.Element:
        """Generate XML element for sequence action."""
        action_elem = ET.Element("action")
        action_elem.set("type", action_spec.action_type)
        action_elem.set("id", action_spec.action_id)
        action_elem.set("category", action_spec.category.value)
        
        # Add configuration parameters
        for param_name, param_value in action_spec.config.items():
            param_elem = ET.SubElement(action_elem, param_name)
            param_elem.text = escape(str(param_value))
        
        # Add dependencies and outputs
        if action_spec.dependencies:
            deps_elem = ET.SubElement(action_elem, "dependencies")
            deps_elem.text = ",".join(action_spec.dependencies)
        
        if action_spec.outputs:
            outputs_elem = ET.SubElement(action_elem, "outputs")
            outputs_elem.text = ",".join(action_spec.outputs)
        
        return action_elem
    
    @ensure(lambda result: result.is_right() or result.get_left().code == "XML_GENERATION_ERROR")
    def build_xml(self) -> Either[KMError, str]:
        """Generate XML for all actions with security validation."""
        try:
            root = ET.Element("actions")
            
            for i, action in enumerate(self.actions):
                action_elem = self._generate_action_xml(action, i)
                root.append(action_elem)
            
            # Validate generated XML
            xml_string = ET.tostring(root, encoding='unicode')
            if not self._validate_xml_security(xml_string):
                return Either.left(KMError.validation_error("Generated XML failed security validation"))
            
            return Either.right(xml_string)
        except Exception as e:
            return Either.left(KMError.execution_error(f"XML generation failed: {str(e)}"))
    
    def _generate_action_xml(self, action: ActionConfiguration, index: int) -> ET.Element:
        """Generate XML element for single action with proper escaping."""
        action_elem = ET.Element("action")
        action_elem.set("type", action.action_type.identifier)
        action_elem.set("id", str(index))
        
        # Add parameters with proper escaping
        for param_name, param_value in action.parameters.items():
            param_elem = ET.SubElement(action_elem, param_name)
            param_elem.text = escape(str(param_value))
        
        return action_elem
    
    def _validate_xml_security(self, xml_string: str) -> bool:
        """Validate XML for security issues."""
        # Check for XML injection patterns
        dangerous_patterns = [
            "<!DOCTYPE", "<!ENTITY", "<?xml", "<![CDATA[",
            "javascript:", "vbscript:", "data:", "file:"
        ]
        
        xml_lower = xml_string.lower()
        return not any(pattern in xml_lower for pattern in dangerous_patterns)
```

#### src/actions/action_registry.py - Action Type Registry
```python
from typing import Dict, Optional, List

class ActionRegistry:
    """Registry of supported Keyboard Maestro action types with validation."""
    
    def __init__(self):
        self._actions: Dict[str, ActionType] = {}
        self._initialize_core_actions()
    
    def _initialize_core_actions(self):
        """Initialize registry with core Keyboard Maestro actions."""
        # Text actions
        self.register_action(ActionType(
            identifier="Type a String",
            category=ActionCategory.TEXT,
            required_params=["text"],
            optional_params=["by_typing", "by_pasting"]
        ))
        
        # System actions
        self.register_action(ActionType(
            identifier="Pause",
            category=ActionCategory.SYSTEM,
            required_params=["duration"],
            optional_params=["unit"]
        ))
        
        # Application actions
        self.register_action(ActionType(
            identifier="Activate a Specific Application",
            category=ActionCategory.APPLICATION,
            required_params=["application"],
            optional_params=["bring_all_windows"]
        ))
        
        # Control flow actions
        self.register_action(ActionType(
            identifier="If Then Else",
            category=ActionCategory.CONTROL,
            required_params=["condition"],
            optional_params=["else_condition"]
        ))
        
        # Variable actions
        self.register_action(ActionType(
            identifier="Set Variable to Text",
            category=ActionCategory.VARIABLE,
            required_params=["variable", "text"],
            optional_params=["append", "trim"]
        ))
        
        # Add more action types...
    
    def register_action(self, action_type: ActionType):
        """Register new action type in registry."""
        self._actions[action_type.identifier] = action_type
    
    def get_action_type(self, identifier: str) -> Optional[ActionType]:
        """Get action type by identifier."""
        return self._actions.get(identifier)
    
    def get_actions_by_category(self, category: ActionCategory) -> List[ActionType]:
        """Get all actions in specific category."""
        return [action for action in self._actions.values() if action.category == category]
    
    def list_all_actions(self) -> List[ActionType]:
        """Get all registered action types."""
        return list(self._actions.values())
```

#### src/server/tools/action_builder_tools.py - MCP Tool Implementation
```python
async def km_action_builder(
    operation: Annotated[str, Field(
        description="Operation type: add_action|build_sequence|validate|preview|catalog",
        pattern=r"^(add_action|build_sequence|validate|preview|catalog)$"
    )],
    macro_id: Annotated[str, Field(
        description="Target macro UUID or name",
        min_length=1,
        max_length=255
    )],
    action_type: Annotated[str, Field(
        description="Action type identifier",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\s]+$"
    )],
    position: Annotated[Optional[int], Field(
        default=None,
        description="Position in action list (0-based, None for append)",
        ge=0
    )] = None,
    action_config: Annotated[Dict[str, Any], Field(
        description="Action-specific configuration parameters"
    )] = {},
    timeout: Annotated[Optional[int], Field(
        default=None,
        description="Action timeout in seconds",
        ge=1,
        le=3600
    )] = None,
    enabled: Annotated[bool, Field(
        default=True,
        description="Whether action is enabled"
    )] = True,
    abort_on_failure: Annotated[bool, Field(
        default=False,
        description="Abort macro if action fails"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Add action to existing macro with comprehensive validation and XML generation.
    
    Supports 300+ Keyboard Maestro action types across categories:
    - Text manipulation: Type text, search/replace, format text
    - Application control: Launch apps, select menus, control windows
    - File operations: Copy, move, delete files and folders
    - System control: Execute scripts, control volume, display dialogs
    - Variable management: Set, get, calculate variables
    - Control flow: If/then/else, loops, switch statements
    - Interface automation: Click, drag, type keystrokes
    - Web requests: HTTP requests, form submission
    
    Returns action addition results with XML preview and validation status.
    """
    if ctx:
        await ctx.info(f"Adding action '{action_type}' to macro '{macro_id}'")
    
    try:
        # Get action type from registry
        action_registry = ActionRegistry()
        action_def = action_registry.get_action_type(action_type)
        
        if not action_def:
            raise ValidationError(f"Unknown action type: {action_type}")
        
        # Create action configuration
        action_timeout = Duration.from_seconds(timeout) if timeout else None
        action_config_obj = ActionConfiguration(
            action_type=action_def,
            parameters=action_config,
            position=position,
            enabled=enabled,
            timeout=action_timeout
        )
        
        # Validate action configuration
        if not action_config_obj.validate_parameters():
            raise ValidationError(f"Invalid parameters for action {action_type}")
        
        if ctx:
            await ctx.report_progress(30, 100, "Validating action configuration")
        
        # Build action XML
        builder = ActionBuilder()
        builder.actions = [action_config_obj]
        xml_result = builder.build_xml()
        
        if xml_result.is_left():
            error = xml_result.get_left()
            raise ValidationError(f"XML generation failed: {error.message}")
        
        action_xml = xml_result.get_right()
        
        if ctx:
            await ctx.report_progress(60, 100, "Integrating with Keyboard Maestro")
        
        # Add action to macro via KM client
        km_client = get_km_client()
        add_result = await asyncio.get_event_loop().run_in_executor(
            None,
            km_client.add_action_to_macro,
            macro_id,
            action_xml,
            position
        )
        
        if add_result.is_left():
            error = add_result.get_left()
            raise ExecutionError(f"Failed to add action to macro: {error.message}")
        
        if ctx:
            await ctx.report_progress(100, 100, "Action added successfully")
        
        return {
            "success": True,
            "action_added": {
                "type": action_type,
                "macro_id": macro_id,
                "position": position,
                "xml_preview": action_xml[:200] + "..." if len(action_xml) > 200 else action_xml,
                "parameter_count": len(action_config),
                "enabled": enabled
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "action_id": str(uuid.uuid4()),
                "xml_validated": True
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "ACTION_BUILDER_ERROR",
                "message": str(e),
                "operation": operation,
                "details": {
                    "action_type": action_type,
                    "target_identifier": target_identifier,
                    "sequence_name": sequence_name
                }
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
```

## 🏗️ Modularity Strategy
- **src/actions/**: New directory for action building functionality (<250 lines each)
- **action_builder.py**: Core builder with XML generation (240 lines)
- **action_registry.py**: Action type registry and validation (180 lines)
- **src/server/tools/action_tools.py**: MCP tool implementation (220 lines)
- **Enhance existing files**: Add action types to types.py

## 🔒 Security Implementation
1. **XML Injection Prevention**: Comprehensive XML sanitization and validation
2. **Parameter Validation**: Type-specific parameter validation for all action types
3. **Action Registry Security**: Whitelist of approved action types
4. **Input Sanitization**: Escape all user-provided parameters
5. **XML Structure Validation**: Ensure generated XML matches KM schema
6. **Execution Safety**: Validate actions before adding to macros

## 📊 Performance Targets
- **Action Validation**: <50ms for parameter validation
- **XML Generation**: <200ms for complex action sequences
- **Action Addition**: <2 seconds for adding action to macro
- **Registry Lookup**: <10ms for action type resolution
- **Complex Builders**: <5 seconds for 20+ action sequences

## ✅ Success Criteria
- [ ] All advanced techniques implemented (builder pattern, functional programming, XML security, contract validation)
- [ ] Individual action addition to existing macros with XML generation and validation
- [ ] Intuitive action sequence composition with fluent API and drag-and-drop style ordering
- [ ] Advanced composition features (grouping, dependencies, parallel execution, error handling)
- [ ] Complete security validation with XML injection prevention for actions and sequences
- [ ] Support for 50+ core Keyboard Maestro action types across all categories
- [ ] Comprehensive action catalog with metadata and templates
- [ ] Real Keyboard Maestro integration with action addition and sequence generation
- [ ] Comprehensive error handling with validation and XML security
- [ ] Property-based testing covers all action types, sequences, and XML generation scenarios
- [ ] Performance meets sub-5-second targets for action building and sequence composition
- [ ] Integration with existing MCP framework and macro creation/editing systems
- [ ] TESTING.md updated with action building, sequence composition, and XML security tests
- [ ] Full documentation with action reference, sequence examples, and security guidelines

## 🎨 Usage Examples

### Individual Action Addition
```python
# Add text typing action to existing macro
result = await client.call_tool("km_action_builder", {
    "operation": "add_action",
    "target_identifier": "550e8400-e29b-41d4-a716-446655440000",
    "action_type": "Type a String",
    "action_config": {
        "text": "Hello, World!",
        "by_typing": True
    },
    "position": 0
})

# Add conditional action with complex parameters
result = await client.call_tool("km_action_builder", {
    "operation": "add_action",
    "target_identifier": "Complex Workflow",
    "action_type": "If Then Else",
    "action_config": {
        "condition": {
            "type": "variable_condition",
            "variable": "ProcessingMode",
            "comparison": "equals",
            "value": "batch"
        }
    },
    "timeout": 30,
    "abort_on_failure": True
})
```

### Action Sequence Composition
```python
# Build text processing sequence
result = await client.call_tool("km_action_builder", {
    "operation": "build_sequence",
    "sequence_name": "Text Processing Workflow",
    "execution_mode": "sequential",
    "error_handling": "continue",
    "actions": [
        {
            "action_type": "Get Clipboard",
            "text": ""
        },
        {
            "action_type": "Search and Replace",
            "text": "oldtext",
            "config": {"replacement": "newtext", "use_regex": False}
        },
        {
            "action_type": "Change Case",
            "text": "title_case"
        },
        {
            "action_type": "Set Variable",
            "variable_name": "ProcessedText",
            "value": "%LastResult%"
        },
        {
            "action_type": "Set Clipboard",
            "text": "%Variable%ProcessedText%"
        }
    ]
})

# Build file processing pipeline
result = await client.call_tool("km_action_builder", {
    "operation": "build_sequence",
    "sequence_name": "Document Processing Pipeline",
    "execution_mode": "sequential",
    "actions": [
        {
            "action_type": "Select Files",
            "file_path": "~/Documents/*.pdf"
        },
        {
            "action_type": "Set Variable", 
            "variable_name": "FileCount",
            "value": "%FileCount%"
        },
        {
            "condition_type": "for_each",
            "condition": {"collection": "SelectedFiles", "variable": "CurrentFile"}
        },
        {
            "action_type": "Open File",
            "file_path": "%Variable%CurrentFile%"
        },
        {
            "action_type": "Extract Text",
            "text": ""
        },
        {
            "action_type": "Save Text",
            "file_path": "~/ProcessedText/%FileName%.txt"
        }
    ]
})

# Application automation sequence
result = await client.call_tool("km_action_builder", {
    "operation": "build_sequence",
    "sequence_name": "Email Processing Automation",
    "execution_mode": "sequential",
    "error_handling": "retry",
    "actions": [
        {
            "action_type": "Activate Application",
            "application": "Mail"
        },
        {
            "action_type": "Select Menu",
            "application": "Mail",
            "config": {"menu": "Mailbox", "item": "Get All New Mail"}
        },
        {
            "action_type": "Set Variable",
            "variable_name": "NewEmailCount",
            "value": "%EmailCount%"
        },
        {
            "condition_type": "if_then",
            "condition": {"condition": "%Variable%NewEmailCount% > 0"}
        },
        {
            "action_type": "Select Menu",
            "application": "Mail",
            "config": {"menu": "File", "item": "Export", "subitem": "As PDF"}
        }
    ]
})
```

### Action Catalog and Validation
```python
# Get available action types
result = await client.call_tool("km_action_builder", {
    "operation": "catalog"
})

# Validate action sequence
result = await client.call_tool("km_action_builder", {
    "operation": "validate",
    "sequence_name": "My Sequence",
    "actions": [...]
})
```

## 🧪 Testing Strategy
- **Property-Based Testing**: Random action configurations with various parameter combinations
- **Security Testing**: XML injection attempts, malformed parameters
- **Integration Testing**: Real action addition to Keyboard Maestro macros
- **Performance Testing**: Large action sequences and complex XML generation
- **Validation Testing**: All action types with required/optional parameters
- **XML Security Testing**: Malicious XML patterns and injection vectors