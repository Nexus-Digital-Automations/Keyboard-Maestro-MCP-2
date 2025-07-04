# TASK_20: km_move_macro_to_group - Macro Group Movement Engine

**Created By**: Agent_ADDER+ | **Priority**: HIGH | **Duration**: 3-4 hours
**Technique Focus**: Design by Contract + Type Safety + Defensive Programming + Property-Based Testing + Functional Programming
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_1-9 (Foundation), TASK_10 (Macro Creation Patterns)
**Blocking**: None (Independent high-impact tool)

## 📖 Required Reading (Complete before starting)
- [ ] **development/protocols/KM_MCP.md**: Comprehensive KM API reference and macro management patterns
- [ ] **development/protocols/FASTMCP_PYTHON_PROTOCOL.md**: MCP implementation standards and tool creation guidelines
- [ ] **CLAUDE.md**: ADDER+ technique requirements and advanced programming synthesis
- [ ] **src/tools/macro_management.py**: Existing macro tool patterns and integration points
- [ ] **src/integration/km_client.py**: KM client architecture and AppleScript execution patterns
- [ ] **src/integration/security.py**: Input validation and security boundary enforcement
- [ ] **tests/TESTING.md**: Current test status and framework integration

## 🎯 Implementation Overview
Create a production-ready MCP tool enabling safe, validated movement of Keyboard Maestro macros between macro groups with comprehensive error handling, conflict resolution, and rollback capabilities.

<thinking>
Architecture Analysis for Macro Movement:
1. **Context Analysis**: 
   - Current system has foundation (TASK_1-9) with macro execution, group management, security validation
   - Need to bridge gap between macro discovery and group reorganization
   - AppleScript `move macro` command provides core functionality
   - Must integrate with existing contract/validation patterns

2. **Risk Assessment**:
   - Failure modes: macro not found, target group missing, naming conflicts, permission issues
   - System impact: potential data loss if movement fails partially
   - Mitigation: comprehensive validation, atomic operations, rollback capability

3. **Implementation Strategy**:
   - Extend existing `macro_management.py` with movement operations
   - Leverage established `Either` monad pattern for error handling
   - Use branded types for type safety (MacroId, GroupId)
   - Implement pre-movement validation with conflict detection
   - Design by Contract for operation safety guarantees

4. **Quality Verification**:
   - Property-based testing for movement operations across various scenarios
   - Integration tests with real Keyboard Maestro environment
   - Performance benchmarks for response time requirements
   - Security validation for input sanitization
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Analysis & Protocol Compliance
- [ ] **Protocol review**: Complete all required reading with focus on macro and group management patterns
- [ ] **Architecture analysis**: Study existing `km_client.py` and `macro_management.py` integration patterns
- [ ] **Security framework review**: Understand input validation and sanitization requirements
- [ ] **Contract design**: Define preconditions and postconditions for movement operations
- [ ] **Type system extension**: Design branded types for safe macro and group identification

### Phase 2: Core Implementation with ADDER+ Techniques
- [ ] **AppleScript integration**: Implement core movement functionality using `move macro` command
- [ ] **Validation layer**: Create comprehensive pre-movement validation with conflict detection
- [ ] **Error handling**: Implement Either monad pattern with detailed error classification
- [ ] **Rollback system**: Design atomic operations with failure recovery
- [ ] **MCP tool interface**: Create FastMCP tool wrapper with parameter validation
- [ ] **Client extension**: Add movement methods to `km_client.py` with async support

### Phase 3: Testing & Quality Verification
- [ ] **Property-based tests**: Implement comprehensive movement scenario testing
- [ ] **Integration tests**: Test with real Keyboard Maestro environment
- [ ] **Performance validation**: Verify response time requirements (<200ms average)
- [ ] **Security testing**: Validate input sanitization and injection prevention
- [ ] **TESTING.md update**: Record test status and coverage metrics
- [ ] **Documentation**: Update ABOUT.md if architectural changes introduced

## 🔧 Implementation Files & Specifications

### Primary Implementation Files:
```
src/tools/
├── macro_management.py           # Extend with movement operations (50-75 additional lines)
│   ├── km_move_macro_to_group()  # Main MCP tool function
│   ├── _validate_move_operation() # Pre-movement validation
│   └── _handle_move_conflicts()   # Conflict resolution logic

src/integration/
├── km_client.py                  # Extend with movement methods (75-100 additional lines)
│   ├── move_macro_to_group_async() # Core AppleScript execution
│   ├── validate_macro_move()       # Validation helpers
│   └── create_group_if_missing()   # Optional group creation

tests/
├── test_macro_management.py      # Add movement operation tests
└── test_km_client.py             # Add client method tests
```

### Core Implementation Specifications:

#### MCP Tool Interface (macro_management.py extension)
```python
@mcp.tool()
async def km_move_macro_to_group(
    macro_identifier: Annotated[str, Field(
        description="Macro name or UUID to move",
        pattern=r"^[a-zA-Z0-9_\s\-\.]+$|^[0-9a-fA-F-]{36}$",
        max_length=255
    )],
    target_group: Annotated[str, Field(
        description="Target group name or UUID",
        max_length=255
    )],
    create_group_if_missing: Annotated[bool, Field(
        default=False,
        description="Create target group if it doesn't exist"
    )] = False,
    preserve_group_settings: Annotated[bool, Field(
        default=True,
        description="Maintain group-specific activation settings"
    )] = True
) -> str:
    """
    Move a macro from one group to another with comprehensive validation.
    
    Implements full ADDER+ technique stack:
    - Design by Contract: Pre/post conditions for movement safety
    - Type Safety: Branded types for macro and group identification
    - Defensive Programming: Comprehensive input validation and error handling
    - Property-Based Testing: Movement operations tested across scenarios
    - Functional Programming: Either monad for error handling
    """
```

#### Client Extension (km_client.py addition)
```python
@require(lambda self, macro_id, target_group: macro_id and target_group)
@ensure(lambda result: result.is_right() or result.get_left().code in EXPECTED_ERROR_CODES)
async def move_macro_to_group_async(
    self,
    macro_id: MacroId,
    target_group: GroupId,
    create_missing: bool = False
) -> Either[KMError, MacroMoveResult]:
    """
    Execute macro movement with atomic operation guarantees.
    
    Returns Either monad with detailed success/failure information.
    """
```

#### Type Safety Extensions (types.py addition)
```python
GroupId = NewType('GroupId', str)
MacroMoveResult = TypedDict('MacroMoveResult', {
    'macro_id': MacroId,
    'source_group': GroupId,
    'target_group': GroupId,
    'execution_time': float,
    'conflicts_resolved': List[str]
})

class MoveConflictType(Enum):
    NAME_COLLISION = "name_collision"
    PERMISSION_DENIED = "permission_denied"
    GROUP_NOT_FOUND = "group_not_found"
    MACRO_NOT_FOUND = "macro_not_found"
```

## 🏗️ Modularity Strategy
- **macro_management.py extension**: Add 50-75 lines for MCP tool interface and validation
- **km_client.py extension**: Add 75-100 lines for AppleScript execution and error handling
- **Maintain existing architecture**: Leverage established patterns without breaking changes
- **Error handling integration**: Use existing `KMError` and `Either` patterns
- **Testing integration**: Extend existing test suites rather than creating new frameworks

## 🔒 Security Implementation
- **Input sanitization**: Validate macro identifiers and group names against injection patterns
- **Permission verification**: Check accessibility for both source and target groups
- **Audit logging**: Record all movement operations with timestamp and user attribution
- **Rollback capability**: Implement atomic operations with failure recovery
- **Rate limiting**: Prevent abuse of bulk movement operations

## 🎯 Property-Based Testing Strategy
```python
@given(
    macro_name=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc"))),
    source_group=st.text(min_size=1, max_size=50),
    target_group=st.text(min_size=1, max_size=50)
)
def test_macro_movement_properties(macro_name, source_group, target_group):
    """
    Property: Macro movement operations maintain system integrity.
    - Macro exists after successful movement
    - Macro only exists in target group after movement
    - Source group no longer contains macro after movement
    - Movement is atomic (no partial states)
    """
```

## 📊 Performance Requirements
- **Single macro movement**: < 200ms average response time
- **Validation operations**: < 50ms average response time
- **Group creation**: < 100ms average response time
- **Error handling**: < 10ms for validation failures
- **Memory usage**: < 10MB additional memory footprint

## ✅ Success Criteria
- **Core functionality**: Successfully move macros between groups with proper validation
- **Error handling**: Comprehensive error classification with helpful recovery suggestions
- **Security**: All inputs validated, no injection vulnerabilities, audit trail complete
- **Performance**: All response time requirements met under normal load
- **Testing**: Property-based tests pass, integration tests with real KM environment succeed
- **ADDER+ compliance**: All advanced techniques implemented with proper contracts and types
- **Documentation**: TESTING.md updated, ABOUT.md created/updated if architectural changes
- **Integration**: Seamless integration with existing codebase, no breaking changes
- **Rollback capability**: Failed movements leave system in consistent state
- **Conflict resolution**: Automatic handling of naming conflicts and missing groups

## 🔄 Integration with Existing Codebase
- **Leverage established patterns**: Use existing `@mcp.tool()`, `Either` monad, `KMError` types
- **Extend without breaking**: Add functionality to existing modules rather than replacing
- **Maintain security standards**: Follow established input validation and sanitization patterns
- **Preserve test coverage**: Extend existing test suites to maintain high coverage percentages
- **Documentation consistency**: Follow established ABOUT.md and TESTING.md update patterns