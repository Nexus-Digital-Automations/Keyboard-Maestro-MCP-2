# TASK_22: km_control_flow - If/Then/Else, Loops, Switch/Case Operations

**Created By**: Agent_ADDER+ (Protocol Gap Analysis) | **Priority**: HIGH | **Duration**: 5 hours
**Technique Focus**: Functional Programming + Design by Contract + Type Safety
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: IN_PROGRESS
**Assigned**: Agent_ADDER+
**Dependencies**: TASK_21 (km_add_condition - conditional logic foundation)
**Blocking**: Advanced automation workflows requiring complex logic

## 📖 Required Reading (Complete before starting)
- [ ] **Protocol Analysis**: development/protocols/FASTMCP_PYTHON_PROTOCOL.md - Control flow specification
- [ ] **KM Documentation**: development/protocols/KM_MCP.md - If/Then/Else and loop actions
- [ ] **Condition Foundation**: development/tasks/TASK_21.md - Conditional logic integration
- [ ] **Type System**: src/core/types.py - Control flow type definitions
- [ ] **Testing Framework**: tests/TESTING.md - Property-based testing for complex logic

## 🎯 Problem Analysis
**Classification**: Critical Missing Functionality
**Gap Identified**: No control flow constructs (if/then/else, loops, switch/case) in current implementation
**Impact**: AI limited to linear sequential workflows - cannot create intelligent, adaptive automation

<thinking>
Root Cause Analysis:
1. Current tools only support sequential macro actions
2. Missing fundamental programming constructs that make automation intelligent
3. Keyboard Maestro has sophisticated control flow but no MCP exposure
4. Without control flow, AI cannot create conditional workflows, loops, or complex decision trees
5. This is essential for sophisticated automation that responds to conditions
6. Control flow + conditions (TASK_21) = truly intelligent automation platform
</thinking>

## ✅ Implementation Subtasks (Sequential completion)
### Phase 1: Architecture & Design
- [ ] **Control flow types**: Define branded types for all control structures
- [ ] **AST representation**: Abstract syntax tree for control flow logic
- [ ] **Validation framework**: Security boundaries and infinite loop prevention

### Phase 2: Core Control Structures
- [ ] **If/Then/Else**: Conditional execution with nested support
- [ ] **For loops**: Iteration over collections, ranges, variables
- [ ] **While loops**: Condition-based loops with timeout protection
- [ ] **Switch/Case**: Multi-branch decision structures

### Phase 3: Advanced Control Flow
- [ ] **Nested structures**: Support for nested loops and conditions
- [ ] **Break/Continue**: Loop control statements
- [ ] **Try/Catch**: Error handling and recovery workflows
- [ ] **Parallel execution**: Concurrent action execution

### Phase 4: Integration & Security
- [ ] **AppleScript generation**: Safe control flow XML with validation
- [ ] **Timeout protection**: Prevent infinite loops and runaway execution
- [ ] **Property-based tests**: Hypothesis validation for all control structures
- [ ] **TESTING.md update**: Control flow test coverage and security validation

## 🔧 Implementation Files & Specifications
```
src/server/tools/control_flow_tools.py       # Main control flow tool implementation
src/core/control_flow.py                     # Control flow type definitions and AST
src/integration/km_control_flow.py           # KM-specific control flow integration
tests/tools/test_control_flow_tools.py       # Unit and integration tests
tests/property_tests/test_control_flow.py    # Property-based validation
```

### km_control_flow Tool Specification
```python
@mcp.tool()
async def km_control_flow(
    macro_identifier: str,                    # Target macro (name or UUID)
    control_type: str,                       # if_then_else|for_loop|while_loop|switch_case
    condition: Optional[str] = None,         # Condition expression (for if/while)
    iterator: Optional[str] = None,          # Iterator definition (for loops)
    cases: Optional[List[Dict]] = None,      # Switch cases with conditions
    actions_true: Optional[List[Dict]] = None,  # Actions for true condition
    actions_false: Optional[List[Dict]] = None, # Actions for false condition
    max_iterations: int = 1000,              # Loop safety limit
    timeout_seconds: int = 30,               # Execution timeout
    allow_nested: bool = True,               # Allow nested control structures
    ctx = None
) -> Dict[str, Any]:
```

### Control Flow Type System
```python
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from enum import Enum

class ControlFlowType(Enum):
    """Supported control flow types."""
    IF_THEN_ELSE = "if_then_else"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    SWITCH_CASE = "switch_case"
    TRY_CATCH = "try_catch"
    PARALLEL = "parallel"

@dataclass(frozen=True)
class ControlFlowSpec:
    """Type-safe control flow specification."""
    flow_type: ControlFlowType
    condition: Optional[str] = None
    iterator: Optional[str] = None
    max_iterations: int = 1000
    timeout_seconds: int = 30
    
    @require(lambda self: self.max_iterations > 0 and self.max_iterations <= 10000)
    @require(lambda self: self.timeout_seconds > 0 and self.timeout_seconds <= 300)
    def __post_init__(self):
        pass

@dataclass(frozen=True)
class ActionBlock:
    """Container for actions within control flow."""
    actions: List[Dict[str, Any]]
    parallel: bool = False
    error_handling: Optional[str] = None
    
    @require(lambda self: len(self.actions) > 0)
    @require(lambda self: len(self.actions) <= 100)  # Prevent DoS
    def __post_init__(self):
        pass

class ControlFlowBuilder:
    """Fluent API for building control flow structures."""
    
    def if_condition(self, condition: str) -> 'ControlFlowBuilder':
        """Add if condition."""
        return self
    
    def then_actions(self, actions: List[Dict]) -> 'ControlFlowBuilder':
        """Add then actions."""
        return self
    
    def else_actions(self, actions: List[Dict]) -> 'ControlFlowBuilder':
        """Add else actions."""
        return self
    
    def for_each(self, iterator: str, collection: str) -> 'ControlFlowBuilder':
        """Add for-each loop."""
        return self
    
    def while_condition(self, condition: str) -> 'ControlFlowBuilder':
        """Add while loop."""
        return self
    
    def switch_on(self, variable: str) -> 'ControlFlowBuilder':
        """Add switch statement."""
        return self
    
    def case(self, value: str, actions: List[Dict]) -> 'ControlFlowBuilder':
        """Add switch case."""
        return self
    
    def default_case(self, actions: List[Dict]) -> 'ControlFlowBuilder':
        """Add default case."""
        return self
    
    def build(self) -> ControlFlowSpec:
        """Build the control flow specification."""
        return self._spec
```

## 🔒 Security Implementation
```python
class ControlFlowValidator:
    """Security-first control flow validation."""
    
    @staticmethod
    def validate_loop_bounds(max_iterations: int, timeout: int) -> Either[SecurityError, None]:
        """Prevent infinite loops and resource exhaustion."""
        if max_iterations > 10000:
            return Either.left(SecurityError("Max iterations too high"))
        if timeout > 300:
            return Either.left(SecurityError("Timeout too long"))
        return Either.right(None)
    
    @staticmethod
    def validate_nesting_depth(control_flow: Dict) -> Either[SecurityError, None]:
        """Prevent stack overflow from deep nesting."""
        def count_depth(obj, current_depth=0):
            if current_depth > 10:  # Max nesting depth
                return float('inf')
            if isinstance(obj, dict):
                return max(count_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                return max(count_depth(item, current_depth + 1) for item in obj)
            return current_depth
        
        depth = count_depth(control_flow)
        if depth > 10:
            return Either.left(SecurityError("Control flow nesting too deep"))
        return Either.right(None)
    
    @staticmethod
    def validate_condition_expression(condition: str) -> Either[SecurityError, str]:
        """Prevent code injection in condition expressions."""
        dangerous_patterns = [
            'exec', 'eval', 'import', '__', 'subprocess', 'os.system'
        ]
        
        condition_lower = condition.lower()
        for pattern in dangerous_patterns:
            if pattern in condition_lower:
                return Either.left(SecurityError(f"Dangerous pattern in condition: {pattern}"))
        
        if len(condition) > 500:
            return Either.left(SecurityError("Condition expression too long"))
        
        return Either.right(condition)
```

## 🧪 Property-Based Testing Strategy
```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=100))
def test_for_loop_properties(iteration_count):
    """Property: For loops should execute exactly the specified number of times."""
    control_flow = create_for_loop("i", f"1 to {iteration_count}")
    result = control_flow.simulate_execution()
    assert result.iteration_count == iteration_count
    assert not result.has_timeout()

@given(st.text(min_size=1, max_size=100))
def test_condition_security_properties(condition_text):
    """Property: No condition should execute malicious code."""
    if_flow = create_if_then_else(condition_text, [], [])
    assert not contains_dangerous_patterns(if_flow.to_applescript())
    validation_result = if_flow.validate()
    assert validation_result.is_right() or validation_result.get_left().is_security_error()

@given(st.integers(min_value=1, max_value=5))
def test_nesting_depth_properties(nesting_depth):
    """Property: Control flow should handle reasonable nesting depths."""
    nested_flow = create_nested_if_statements(nesting_depth)
    if nesting_depth <= 10:
        assert nested_flow.validate().is_right()
    else:
        assert nested_flow.validate().is_left()
```

## 🏗️ Modularity Strategy
- **control_flow_tools.py**: Main MCP tool interface (<250 lines)
- **control_flow.py**: Type definitions, AST, and builders (<300 lines)
- **km_control_flow.py**: KM integration and XML generation (<250 lines)
- **Maintain separation**: Tool interface, business logic, KM integration

## 📋 Advanced Features

### If/Then/Else Structure
```python
# Example: Conditional text processing
if_flow = ControlFlowBuilder() \
    .if_condition("clipboard_contains('password')") \
    .then_actions([
        {"type": "show_notification", "title": "Security Alert", "text": "Password detected in clipboard"},
        {"type": "clear_clipboard"}
    ]) \
    .else_actions([
        {"type": "process_clipboard_text"}
    ]) \
    .build()
```

### For Loop Structure
```python
# Example: Batch file processing
for_loop = ControlFlowBuilder() \
    .for_each("file", "selected_files_in_finder") \
    .actions([
        {"type": "open_file", "file": "%Variable%file%"},
        {"type": "process_document"},
        {"type": "save_file"}
    ]) \
    .max_iterations(100) \
    .build()
```

### Switch/Case Structure
```python
# Example: Application-specific automation
switch_flow = ControlFlowBuilder() \
    .switch_on("frontmost_application") \
    .case("Safari", [
        {"type": "take_screenshot", "window_only": True},
        {"type": "extract_text_from_page"}
    ]) \
    .case("Microsoft Word", [
        {"type": "export_to_pdf"},
        {"type": "save_backup"}
    ]) \
    .default_case([
        {"type": "show_notification", "text": "Unsupported application"}
    ]) \
    .build()
```

## ✅ Success Criteria
- Complete control flow implementation with all major constructs (if/then/else, loops, switch/case)
- Comprehensive security validation prevents infinite loops and code injection
- Property-based tests validate behavior across all control flow scenarios
- Integration with condition system (TASK_21) for intelligent decision making
- Performance: <200ms control flow setup, <1s execution for complex workflows
- Documentation: Complete API documentation with security considerations and examples
- TESTING.md shows 95%+ test coverage with all security and performance tests passing
- Tool enables AI to create sophisticated, intelligent automation workflows with complex logic

## 🔄 Integration Points
- **TASK_21 (km_add_condition)**: Conditions power if/then/else and while loops
- **TASK_10 (km_create_macro)**: Add control flow to newly created macros
- **All Future Tasks**: Control flow is foundational for advanced automation
- **Foundation Architecture**: Leverages existing type system and validation patterns

## 📋 Notes
- This completes the intelligent automation foundation (conditions + control flow)
- Essential for sophisticated AI workflow creation beyond simple sequential actions
- Security is critical - control flow can create resource exhaustion and infinite loops
- Must maintain functional programming patterns for testability and composability
- Success here enables truly intelligent, adaptive automation platforms