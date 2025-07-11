# TASK_30: km_macro_template_system - Reusable Macro Templates and Library

**Created By**: Agent_1 (Advanced Macro Creation Enhancement) | **Priority**: HIGH | **Duration**: 4 hours
**Technique Focus**: Template Pattern + Functional Programming + Type Safety + Property-Based Testing
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_28 (km_macro_editor), TASK_29 (km_action_sequence_builder)
**Blocking**: Standardized macro creation workflows requiring reusable components

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Template system specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - Macro library and sharing capabilities
- [ ] **Foundation Architecture**: src/creation/templates.py - Existing template patterns
- [ ] **Action Builder**: development/tasks/TASK_29.md - Action sequence composition integration
- [ ] **Macro Editor**: development/tasks/TASK_28.md - Template modification and customization
- [ ] **Type System**: src/core/types.py - Branded types for templates and parameters
- [ ] **Testing Framework**: tests/TESTING.md - Property-based testing requirements

## 🎯 Problem Analysis
**Classification**: Missing Efficiency Enhancement
**Gap Identified**: No reusable template system for common automation patterns
**Impact**: AI must recreate similar macros from scratch - cannot leverage proven automation patterns

<thinking>
Root Cause Analysis:
1. Current tools require building every macro from basic components
2. No library of proven automation patterns and templates
3. Missing parameterization system for customizable macro templates
4. Cannot share and reuse successful automation workflows
5. No standardization of common automation tasks
6. Essential for efficient macro creation and best practice sharing
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Template type system**: Define branded types for templates, parameters, and libraries
- [ ] **Parameterization framework**: Template parameter substitution and validation
- [ ] **Library management**: Template storage, versioning, and organization

### Phase 2: Core Template System
- [ ] **Template creation**: Convert existing macros into reusable templates
- [ ] **Parameter system**: Define template parameters with types and validation
- [ ] **Template instantiation**: Generate macros from templates with parameter substitution
- [ ] **Template validation**: Comprehensive template structure and parameter validation

### Phase 3: Template Library
- [ ] **Built-in templates**: Common automation patterns (file processing, email, backups, etc.)
- [ ] **Template categories**: Organized library with search and filtering
- [ ] **Template metadata**: Documentation, usage examples, compatibility information
- [ ] **Template sharing**: Export/import templates for sharing between systems

### Phase 4: Advanced Features
- [ ] **Template inheritance**: Base templates with specialized variations
- [ ] **Composite templates**: Templates that combine multiple sub-templates
- [ ] **Dynamic parameters**: Parameters that depend on system state or other parameters
- [ ] **Template analytics**: Usage tracking and optimization suggestions

### Phase 5: Integration & Testing
- [ ] **TESTING.md update**: Real-time test status and coverage tracking
- [ ] **Security validation**: Prevent malicious templates and parameter injection
- [ ] **Property-based tests**: Hypothesis validation for all template operations
- [ ] **Integration tests**: Verify compatibility with macro editor and sequence builder

## 🔧 Implementation Files & Specifications
```
src/server/tools/macro_template_tools.py        # Main macro template tool implementation
src/core/macro_templates.py                     # Template type definitions and operations
src/templates/template_library.py               # Built-in template library and catalog
src/templates/parameter_system.py               # Template parameterization and validation
tests/tools/test_macro_template_tools.py        # Unit and integration tests
tests/property_tests/test_macro_templates.py    # Property-based template validation
```

### km_macro_template_system Tool Specification
```python
@mcp.tool()
async def km_macro_template_system(
    operation: str,                             # create|instantiate|library|search|import|export
    template_name: Optional[str] = None,        # Template identifier
    parameters: Optional[Dict] = None,          # Template parameters for instantiation
    template_spec: Optional[Dict] = None,       # Template specification for creation
    search_query: Optional[str] = None,         # Search terms for library
    category: Optional[str] = None,             # Template category filter
    export_format: str = "json",                # Export format (json|xml|kmmacros)
    validation_level: str = "standard",         # Template validation level
    ctx = None
) -> Dict[str, Any]:
```

### Macro Template Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Set
from enum import Enum

class ParameterType(Enum):
    """Template parameter types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    FILE_PATH = "file_path"
    FOLDER_PATH = "folder_path"
    APPLICATION_NAME = "application_name"
    EMAIL_ADDRESS = "email_address"
    URL = "url"
    COLOR = "color"
    CHOICE = "choice"

class TemplateCategory(Enum):
    """Template organization categories."""
    FILE_MANAGEMENT = "file_management"
    TEXT_PROCESSING = "text_processing"
    EMAIL_AUTOMATION = "email_automation"
    APPLICATION_CONTROL = "application_control"
    SYSTEM_ADMINISTRATION = "system_administration"
    MEDIA_PROCESSING = "media_processing"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT_TOOLS = "development_tools"
    COMMUNICATION = "communication"
    CUSTOM = "custom"

@dataclass(frozen=True)
class TemplateParameter:
    """Type-safe template parameter definition."""
    name: str
    parameter_type: ParameterType
    description: str
    default_value: Optional[Any] = None
    required: bool = True
    validation_pattern: Optional[str] = None
    choices: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.description) > 0)
    @require(lambda self: self.parameter_type in ParameterType)
    def __post_init__(self):
        pass
    
    def validate_value(self, value: Any) -> Either[ValidationError, Any]:
        """Validate parameter value against constraints."""
        # Type validation
        if self.parameter_type == ParameterType.STRING and not isinstance(value, str):
            return Either.left(ValidationError(f"Parameter {self.name} must be string"))
        
        # Pattern validation
        if self.validation_pattern and isinstance(value, str):
            if not re.match(self.validation_pattern, value):
                return Either.left(ValidationError(f"Parameter {self.name} doesn't match pattern"))
        
        # Choice validation
        if self.choices and value not in self.choices:
            return Either.left(ValidationError(f"Parameter {self.name} must be one of {self.choices}"))
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return Either.left(ValidationError(f"Parameter {self.name} below minimum {self.min_value}"))
        
        return Either.right(value)

@dataclass(frozen=True)
class MacroTemplate:
    """Complete macro template specification."""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    version: str
    author: str
    parameters: List[TemplateParameter]
    actions: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    
    @require(lambda self: len(self.template_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.actions) > 0)
    @require(lambda self: len(self.parameters) <= 50)  # Reasonable limit
    def __post_init__(self):
        pass
    
    def get_parameter_names(self) -> Set[str]:
        return {param.name for param in self.parameters}
    
    def get_required_parameters(self) -> Set[str]:
        return {param.name for param in self.parameters if param.required}
    
    def validate_parameters(self, param_values: Dict[str, Any]) -> Either[ValidationError, Dict[str, Any]]:
        """Validate all parameter values."""
        validated = {}
        
        # Check required parameters
        required = self.get_required_parameters()
        missing = required - set(param_values.keys())
        if missing:
            return Either.left(ValidationError(f"Missing required parameters: {missing}"))
        
        # Validate each parameter
        for param in self.parameters:
            if param.name in param_values:
                result = param.validate_value(param_values[param.name])
                if result.is_left():
                    return result
                validated[param.name] = result.get_right()
            elif param.default_value is not None:
                validated[param.name] = param.default_value
        
        return Either.right(validated)

@dataclass(frozen=True)
class TemplateInstance:
    """Instantiated template with parameters."""
    template_id: str
    instance_name: str
    parameters: Dict[str, Any]
    generated_macro: Dict[str, Any]
    creation_timestamp: str
    
    @require(lambda self: len(self.template_id) > 0)
    @require(lambda self: len(self.instance_name) > 0)
    def __post_init__(self):
        pass

class MacroTemplateLibrary:
    """Template library management."""
    
    def __init__(self):
        self._templates: Dict[str, MacroTemplate] = {}
        self._load_builtin_templates()
    
    def add_template(self, template: MacroTemplate) -> Either[LibraryError, None]:
        """Add template to library."""
        if template.template_id in self._templates:
            return Either.left(LibraryError("Template already exists"))
        
        self._templates[template.template_id] = template
        return Either.right(None)
    
    def get_template(self, template_id: str) -> Either[LibraryError, MacroTemplate]:
        """Get template by ID."""
        if template_id not in self._templates:
            return Either.left(LibraryError("Template not found"))
        
        return Either.right(self._templates[template_id])
    
    def search_templates(self, query: str, category: Optional[TemplateCategory] = None) -> List[MacroTemplate]:
        """Search templates by query and category."""
        results = []
        query_lower = query.lower()
        
        for template in self._templates.values():
            # Category filter
            if category and template.category != category:
                continue
            
            # Text search
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return sorted(results, key=lambda t: t.name)
    
    def get_categories(self) -> Dict[TemplateCategory, int]:
        """Get template count by category."""
        counts = {}
        for template in self._templates.values():
            counts[template.category] = counts.get(template.category, 0) + 1
        return counts
    
    def _load_builtin_templates(self):
        """Load built-in template library."""
        # File processing template
        file_backup_template = MacroTemplate(
            template_id="builtin_file_backup",
            name="File Backup",
            description="Create timestamped backup copies of selected files",
            category=TemplateCategory.FILE_MANAGEMENT,
            version="1.0",
            author="System",
            parameters=[
                TemplateParameter("source_folder", ParameterType.FOLDER_PATH, "Source folder to backup", required=True),
                TemplateParameter("backup_folder", ParameterType.FOLDER_PATH, "Destination backup folder", required=True),
                TemplateParameter("include_timestamp", ParameterType.BOOLEAN, "Include timestamp in backup names", default_value=True),
                TemplateParameter("file_pattern", ParameterType.STRING, "File pattern to backup", default_value="*")
            ],
            actions=[
                {"type": "select_files", "config": {"folder": "{{source_folder}}", "pattern": "{{file_pattern}}"}},
                {"type": "copy_files", "config": {"destination": "{{backup_folder}}", "timestamp": "{{include_timestamp}}"}}
            ],
            tags={"backup", "files", "utility"}
        )
        
        self._templates[file_backup_template.template_id] = file_backup_template

class TemplateProcessor:
    """Template instantiation and parameter substitution."""
    
    @staticmethod
    def instantiate_template(
        template: MacroTemplate, 
        parameters: Dict[str, Any], 
        instance_name: str
    ) -> Either[ProcessingError, TemplateInstance]:
        """Instantiate template with parameters."""
        # Validate parameters
        param_result = template.validate_parameters(parameters)
        if param_result.is_left():
            return Either.left(ProcessingError(f"Parameter validation failed: {param_result.get_left()}"))
        
        validated_params = param_result.get_right()
        
        # Process template actions
        processed_actions = []
        for action in template.actions:
            processed_action = TemplateProcessor._substitute_parameters(action, validated_params)
            processed_actions.append(processed_action)
        
        # Process template triggers
        processed_triggers = []
        for trigger in template.triggers:
            processed_trigger = TemplateProcessor._substitute_parameters(trigger, validated_params)
            processed_triggers.append(processed_trigger)
        
        # Generate macro specification
        generated_macro = {
            "name": instance_name,
            "description": f"Generated from template: {template.name}",
            "actions": processed_actions,
            "triggers": processed_triggers,
            "enabled": True
        }
        
        return Either.right(TemplateInstance(
            template_id=template.template_id,
            instance_name=instance_name,
            parameters=validated_params,
            generated_macro=generated_macro,
            creation_timestamp=datetime.now().isoformat()
        ))
    
    @staticmethod
    def _substitute_parameters(obj: Any, parameters: Dict[str, Any]) -> Any:
        """Recursively substitute template parameters."""
        if isinstance(obj, str):
            # Replace {{parameter_name}} with actual values
            for param_name, param_value in parameters.items():
                placeholder = f"{{{{{param_name}}}}}"
                obj = obj.replace(placeholder, str(param_value))
            return obj
        elif isinstance(obj, dict):
            return {k: TemplateProcessor._substitute_parameters(v, parameters) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TemplateProcessor._substitute_parameters(item, parameters) for item in obj]
        else:
            return obj
```

## 🔒 Security Implementation
```python
class TemplateSecurityValidator:
    """Security-first template validation."""
    
    @staticmethod
    def validate_template_actions(actions: List[Dict]) -> Either[SecurityError, None]:
        """Validate template actions for security risks."""
        for action in actions:
            # Check for dangerous action types
            action_type = action.get("type", "")
            if action_type in ["execute_shell", "run_applescript", "execute_javascript"]:
                # Validate script content if present
                config = action.get("config", {})
                if "script" in config:
                    script_content = config["script"]
                    if TemplateSecurityValidator._contains_dangerous_patterns(script_content):
                        return Either.left(SecurityError("Dangerous script patterns in template"))
            
            # Validate file operations
            if "file_path" in action.get("config", {}):
                file_path = action["config"]["file_path"]
                if TemplateSecurityValidator._is_dangerous_path(file_path):
                    return Either.left(SecurityError("Dangerous file path in template"))
        
        return Either.right(None)
    
    @staticmethod
    def validate_parameter_injection(template: MacroTemplate, parameters: Dict[str, Any]) -> Either[SecurityError, None]:
        """Prevent parameter injection attacks."""
        for param_name, param_value in parameters.items():
            if not isinstance(param_value, (str, int, float, bool)):
                return Either.left(SecurityError("Invalid parameter type"))
            
            # Check for injection patterns in string parameters
            if isinstance(param_value, str):
                if TemplateSecurityValidator._contains_injection_patterns(param_value):
                    return Either.left(SecurityError(f"Injection pattern detected in parameter {param_name}"))
        
        return Either.right(None)
    
    @staticmethod
    def _contains_dangerous_patterns(script: str) -> bool:
        """Check for dangerous script patterns."""
        dangerous_patterns = [
            "rm -rf", "sudo", "curl", "wget", "nc ", "netcat",
            "eval", "exec", "system", "shell_exec", "passthru"
        ]
        
        script_lower = script.lower()
        return any(pattern in script_lower for pattern in dangerous_patterns)
    
    @staticmethod
    def _is_dangerous_path(path: str) -> bool:
        """Check for dangerous file paths."""
        dangerous_paths = [
            "/etc/", "/usr/bin/", "/System/", "~/Library/Keychains/",
            "C:\\Windows\\", "C:\\Program Files\\"
        ]
        
        return any(path.startswith(dangerous) for dangerous in dangerous_paths)
    
    @staticmethod
    def _contains_injection_patterns(value: str) -> bool:
        """Check for injection patterns in parameter values."""
        injection_patterns = [
            "$(", "`", "{{", "}}", "<script", "javascript:",
            "file://", "data:", "vbscript:"
        ]
        
        value_lower = value.lower()
        return any(pattern in value_lower for pattern in injection_patterns)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=50))
def test_template_parameter_properties(param_name):
    """Property: Template parameters should handle various names."""
    if is_valid_parameter_name(param_name):
        param = TemplateParameter(
            name=param_name,
            parameter_type=ParameterType.STRING,
            description="Test parameter"
        )
        assert param.name == param_name
    else:
        with pytest.raises(ValueError):
            TemplateParameter(param_name, ParameterType.STRING, "Test")

@given(st.dictionaries(st.text(), st.text(), min_size=1, max_size=10))
def test_parameter_substitution_properties(parameters):
    """Property: Parameter substitution should be safe and complete."""
    template_text = "Hello {{name}}, your value is {{value}}"
    
    # Test safe substitution
    result = TemplateProcessor._substitute_parameters(template_text, parameters)
    
    # Should not contain injection patterns
    assert not TemplateSecurityValidator._contains_injection_patterns(result)
    
    # Should substitute known parameters
    for param_name in parameters.keys():
        placeholder = f"{{{{{param_name}}}}}"
        if placeholder in template_text:
            assert placeholder not in result

@given(st.lists(st.dictionaries(st.text(), st.text()), min_size=1, max_size=5))
def test_template_action_validation_properties(actions):
    """Property: Template actions should pass security validation."""
    # Filter out dangerous action types for this test
    safe_actions = [{"type": "type_text", "config": action} for action in actions]
    
    validation_result = TemplateSecurityValidator.validate_template_actions(safe_actions)
    assert validation_result.is_right()
```

## 🏗️ Modularity Strategy
- **macro_template_tools.py**: Main MCP tool interface (<250 lines)
- **macro_templates.py**: Template type definitions and core logic (<350 lines)
- **template_library.py**: Built-in templates and library management (<300 lines)
- **parameter_system.py**: Parameter validation and substitution (<200 lines)

## 📋 Template Examples

### File Processing Template
```python
# Example: Create file backup template
file_backup_template = MacroTemplate(
    template_id="custom_file_backup",
    name="Automated File Backup",
    description="Backup files with customizable filters and destinations",
    category=TemplateCategory.FILE_MANAGEMENT,
    version="1.0",
    author="User",
    parameters=[
        TemplateParameter("source_path", ParameterType.FOLDER_PATH, "Source folder"),
        TemplateParameter("backup_path", ParameterType.FOLDER_PATH, "Backup destination"),
        TemplateParameter("file_extension", ParameterType.STRING, "File extension filter", default_value="*"),
        TemplateParameter("include_subdirs", ParameterType.BOOLEAN, "Include subdirectories", default_value=True)
    ],
    actions=[
        {"type": "find_files", "config": {"path": "{{source_path}}", "extension": "{{file_extension}}", "recursive": "{{include_subdirs}}"}},
        {"type": "copy_files", "config": {"destination": "{{backup_path}}", "preserve_structure": True}}
    ]
)

# Instantiate template
result = await km_macro_template_system(
    operation="instantiate",
    template_name="custom_file_backup",
    parameters={
        "source_path": "~/Documents",
        "backup_path": "~/Backups/Documents",
        "file_extension": "*.pdf",
        "include_subdirs": False
    }
)
```

### Email Processing Template
```python
# Example: Email automation template
email_template = MacroTemplate(
    template_id="email_processor",
    name="Email Processing Automation",
    description="Process emails with customizable actions",
    category=TemplateCategory.EMAIL_AUTOMATION,
    version="1.0",
    author="User",
    parameters=[
        TemplateParameter("sender_filter", ParameterType.EMAIL_ADDRESS, "Filter by sender"),
        TemplateParameter("subject_contains", ParameterType.STRING, "Subject must contain"),
        TemplateParameter("action_type", ParameterType.CHOICE, "Action to perform", 
                         choices=["archive", "forward", "reply", "delete"]),
        TemplateParameter("forward_to", ParameterType.EMAIL_ADDRESS, "Forward destination", required=False)
    ],
    actions=[
        {"type": "filter_emails", "config": {"sender": "{{sender_filter}}", "subject": "{{subject_contains}}"}},
        {"type": "email_action", "config": {"action": "{{action_type}}", "forward_to": "{{forward_to}}"}}
    ]
)
```

## ✅ Success Criteria
- Complete template system with creation, instantiation, and library management
- Comprehensive parameter system with type validation and security checks
- Built-in template library covering common automation patterns
- Advanced features (inheritance, composition, dynamic parameters)
- Comprehensive security validation prevents malicious templates and injection attacks
- Property-based tests validate behavior across all template operations
- Integration with macro editor (TASK_28) and sequence builder (TASK_29)
- Performance: <100ms template instantiation, <200ms library search, <1s template validation
- Documentation: Complete API documentation with template creation guide
- TESTING.md shows 95%+ test coverage with all security and functionality tests passing
- Tool enables AI to leverage proven automation patterns through reusable templates

## 🔄 Integration Points
- **TASK_28 (km_macro_editor)**: Edit templates and template-generated macros
- **TASK_29 (km_action_sequence_builder)**: Create templates from action sequences
- **TASK_31 (km_macro_testing_framework)**: Test templates and template instances
- **TASK_10 (km_create_macro)**: Generate macros from templates
- **All Existing Tools**: Create templates from any macro tool output
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This enables rapid macro creation through proven automation patterns
- Essential for standardizing common automation tasks and best practices
- Template library provides immediate value with built-in automation patterns
- Security is critical - templates can contain complex automation logic
- Must maintain functional programming patterns for testability and composability
- Success here enables efficient macro creation through reusable, parameterized templates