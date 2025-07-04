# TASK_21: km_add_condition - Complex Conditional Logic for Control Flow

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: HIGH | **Duration**: 4 hours
**Technique Focus**: Design by Contract + Type Safety + Functional Programming
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: IN_PROGRESS
**Assigned**: Agent_ADDER+
**Dependencies**: TASK_10 (km_create_macro foundation)
**Blocking**: TASK_22, TASK_23 (Advanced control flow operations)

## 📖 Required Reading (Complete before starting)
- [x] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Condition specification
- [x] **KM Documentation**: development/protocols/KM_MCP.md - Conditional action types
- [x] **Foundation Architecture**: src/server/tools/creation_tools.py - Macro building patterns
- [x] **Type System**: src/core/types.py - Branded types for conditions and logic
- [x] **Testing Framework**: tests/TESTING.md - Property-based testing requirements

## 🎯 Problem Analysis
**Classification**: Missing Critical Functionality
**Gap Identified**: No conditional logic capabilities in current 13-tool implementation
**Impact**: AI limited to linear workflows only - cannot create intelligent automation

<thinking>
Root Cause Analysis:
1. Current implementation focused on basic CRUD operations for macros
2. Missing fundamental control flow that makes automation intelligent
3. Keyboard Maestro has powerful conditional system but no MCP access
4. Without conditions, AI cannot create adaptive, responsive workflows
5. This is THE critical gap preventing sophisticated automation
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Condition type system**: Define branded types for all condition categories
- [ ] **Validation framework**: Input sanitization and security boundaries
- [ ] **Builder pattern**: Fluent API for condition construction

### Phase 2: Core Condition Types
- [ ] **Text conditions**: Contains, matches, regex patterns with injection protection
- [ ] **Application conditions**: App state, window properties, process validation
- [ ] **System conditions**: Time, date, file existence, network connectivity
- [ ] **Variable conditions**: KM variable comparisons with type safety

### Phase 3: Advanced Logic Operations
- [ ] **Boolean logic**: AND, OR, NOT operations with precedence
- [ ] **Comparison operators**: Equals, greater than, less than, ranges
- [ ] **Pattern matching**: Regex with security validation and timeout protection

### Phase 4: Integration & Testing
- [ ] **AppleScript generation**: Safe condition XML with escaping
- [ ] **Property-based tests**: Hypothesis validation for all condition types
- [ ] **Security testing**: Injection prevention and input boundary validation
- [ ] **TESTING.md update**: Real-time test status and coverage tracking

## 🔧 Implementation Files & Specifications
```
src/server/tools/condition_tools.py          # Main condition tool implementation
src/core/conditions.py                       # Condition type definitions and builders
src/integration/km_conditions.py            # KM-specific condition integration
tests/tools/test_condition_tools.py         # Unit and integration tests
tests/property_tests/test_conditions.py     # Property-based condition validation
```

### km_add_condition Tool Specification
```python
@mcp.tool()
async def km_add_condition(
    macro_identifier: str,                    # Target macro (name or UUID)
    condition_type: str,                      # text|app|system|variable|logic
    operator: str,                           # contains|equals|greater|less|regex|exists
    operand: str,                            # Comparison value with validation
    case_sensitive: bool = True,             # Text comparison sensitivity
    negate: bool = False,                    # Invert condition result
    action_on_true: Optional[str] = None,    # Action for true condition
    action_on_false: Optional[str] = None,   # Action for false condition
    timeout_seconds: int = 10,               # Condition evaluation timeout
    ctx = None
) -> Dict[str, Any]:
```

### Security & Validation Requirements
```python
# Input validation with branded types
MacroIdentifier = NewType('MacroIdentifier', str)
ConditionType = Literal['text', 'app', 'system', 'variable', 'logic']
OperatorType = Literal['contains', 'equals', 'greater', 'less', 'regex', 'exists']

@require(lambda macro_id: len(macro_id.strip()) > 0)
@require(lambda operand: len(operand) <= 1000)  # Prevent DoS
@ensure(lambda result: result.get('condition_id') is not None)
```

## 🏗️ Modularity Strategy
- **condition_tools.py**: Main MCP tool interface (<200 lines)
- **conditions.py**: Type definitions and builders (<250 lines)
- **km_conditions.py**: KM integration layer (<200 lines)
- **Maintain separation**: Tool interface, business logic, KM integration

## 🔒 Security Implementation
```python
class ConditionValidator:
    """Security-first condition validation."""
    
    @staticmethod
    def validate_regex_pattern(pattern: str) -> Either[SecurityError, str]:
        """Prevent ReDoS attacks and malicious patterns."""
        if len(pattern) > 500:
            return Either.left(SecurityError("Regex pattern too long"))
        
        # Validate against dangerous patterns
        dangerous_patterns = [r'\(\?\#', r'\(\?\>', r'\(\?\<']
        for danger in dangerous_patterns:
            if danger in pattern:
                return Either.left(SecurityError("Dangerous regex pattern"))
        
        return Either.right(pattern)
    
    @staticmethod  
    def sanitize_file_path(path: str) -> Either[SecurityError, str]:
        """Prevent path traversal in file conditions."""
        if '..' in path or path.startswith('/'):
            return Either.left(SecurityError("Invalid file path"))
        return Either.right(path)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_text_condition_properties(text_value):
    """Property: Text conditions should never execute malicious code."""
    condition = create_text_condition("contains", text_value)
    assert not contains_script_injection(condition.to_applescript())
    assert condition.validate().is_right()

@given(st.integers(min_value=0, max_value=2147483647))
def test_numeric_comparison_properties(number):
    """Property: Numeric conditions should handle all valid integers."""
    condition = create_variable_condition("greater_than", str(number))
    result = condition.evaluate()
    assert result.is_right() or result.get_left().is_validation_error()
```

## ✅ Success Criteria
- Condition tool supports all major KM condition types (text, app, system, variable, logic)
- Comprehensive security validation prevents injection attacks
- Property-based tests validate behavior across input ranges
- Integration with macro creation workflow (TASK_10 compatibility)
- Performance: <100ms condition evaluation, <50ms validation
- Documentation: Complete API documentation with security considerations
- TESTING.md shows 95%+ test coverage with all security tests passing
- Tool enables AI to create intelligent, adaptive automation workflows

## 🔄 Integration Points
- **TASK_10 (km_create_macro)**: Add conditions to newly created macros
- **TASK_22 (km_control_flow)**: Conditions enable if/then/else logic
- **TASK_23 (km_create_trigger_advanced)**: Conditional triggers based on system state
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This is THE critical missing piece for intelligent automation
- Keyboard Maestro's condition system is extremely powerful but unexposed
- Success here unlocks sophisticated AI workflow creation
- Security is paramount - conditions can access sensitive system state
- Must maintain functional programming patterns for testability