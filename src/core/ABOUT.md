# Core Macro Engine

## Purpose
Foundational macro execution engine providing type-safe command processing with contract-based validation and comprehensive security enforcement for Keyboard Maestro automation.

## Key Components  
- **types.py**: Branded types, protocols, and immutable data structures for type safety
- **contracts.py**: Design by Contract system with preconditions, postconditions, and invariants
- **engine.py**: Main execution engine with security boundaries and performance monitoring
- **parser.py**: Command parsing with input sanitization and validation
- **context.py**: Execution context management with permission control and variable isolation
- **errors.py**: Comprehensive error hierarchy with security-conscious error handling

## Architecture & Integration
**Dependencies**: No external dependencies - pure Python with typing and dataclasses
**Patterns**: 
- Command Pattern for macro operations with pluggable command implementations
- Design by Contract for comprehensive validation and error prevention
- Immutable Data Structures for thread-safe state management
- Security Boundaries with permission-based access control
**Integration**: Provides base types and execution framework for integration and command layers

## Critical Considerations
- **Security**: Input sanitization prevents script injection and path traversal attacks. Permission system enforces least-privilege access. Contract violations logged for audit
- **Performance**: Target <100ms for simple macro execution. Immutable structures minimize memory allocation. Context cleanup prevents resource leaks

## Related Documentation
- [Task 1 Specification](../../development/tasks/TASK_1.md) - Complete implementation requirements
- [Testing Framework](../../tests/TESTING.md) - Test status and coverage tracking