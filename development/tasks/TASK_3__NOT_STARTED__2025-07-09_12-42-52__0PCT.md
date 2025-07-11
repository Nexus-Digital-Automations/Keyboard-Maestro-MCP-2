# TASK_3: Macro Command Library

**Created By**: Agent_1 | **Priority**: MEDIUM | **Duration**: 2 hours
**Technique Focus**: Design by Contract + Security Boundaries + Type Safety
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_1 (Core engine), TASK_2 (KM integration)
**Blocking**: TASK_4

## 📖 Required Reading (Complete before starting)
- [ ] **src/core/**: Review core engine types and contracts from TASK_1
- [ ] **src/integration/**: Review KM integration patterns from TASK_2
- [ ] **development/protocols/KM_MCP.md**: Command specifications and capabilities
- [ ] **tests/TESTING.md**: Current test framework and status

## 🎯 Implementation Overview
Implement a comprehensive library of macro commands with strong type safety, contract-based validation, and security boundaries to prevent malicious operations.

<thinking>
Command library architecture:
1. Command Pattern: Each macro operation as executable command object
2. Security First: All commands validated and sandboxed
3. Type Safety: Strong typing prevents runtime errors
4. Composability: Commands can be combined and chained
5. Extensibility: Easy addition of new command types
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Command Architecture
- [ ] **Base command interface**: Define contract-based command protocol
- [ ] **Command registry**: Type-safe registration and discovery system
- [ ] **Validation framework**: Security validation for all command types
- [ ] **Error handling**: Comprehensive error types for command failures

### Phase 2: Core Command Implementations
- [ ] **Text commands**: Text manipulation with security validation
- [ ] **System commands**: Safe system operations with sandboxing
- [ ] **Application commands**: Application control with permission checking
- [ ] **Flow control**: Conditional and loop commands with safety limits

### Phase 3: Security & Testing
- [ ] **Security audit**: Validate all commands against security requirements
- [ ] **Command tests**: Comprehensive testing for each command type
- [ ] **TESTING.md update**: Document command library test coverage
- [ ] **Documentation**: Create ABOUT.md for command architecture

## 🔧 Implementation Files & Specifications

### Command Library Files to Create:
```
src/
├── commands/
│   ├── __init__.py           # Command registry and public API
│   ├── base.py               # Base command contracts (75-100 lines)
│   ├── text.py               # Text manipulation commands (150-200 lines)
│   ├── system.py             # System operation commands (150-200 lines)
│   ├── application.py        # Application control commands (150-200 lines)
│   ├── flow.py               # Flow control commands (100-150 lines)
│   ├── validation.py         # Security validation utilities (100-150 lines)
│   └── registry.py           # Command registration system (75-100 lines)
└── tests/
    └── test_commands/
        ├── test_base.py          # Base command tests
        ├── test_text.py          # Text command tests
        ├── test_system.py        # System command tests
        ├── test_application.py   # Application command tests
        ├── test_flow.py          # Flow control tests
        └── test_validation.py    # Security validation tests
```

### Key Implementation Requirements:

#### base.py - Command Contracts
```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass

class CommandContract(Protocol):
    """Contract for all macro commands."""
    
    @require(lambda self, context: context.is_valid())
    @ensure(lambda result: result.is_successful() or result.has_error_info())
    def execute(self, context: ExecutionContext) -> CommandResult: ...
    
    @require(lambda self: True)
    @ensure(lambda result: isinstance(result, bool))
    def validate(self) -> bool: ...
    
    def get_required_permissions(self) -> frozenset[Permission]: ...

@dataclass(frozen=True)
class BaseCommand(ABC):
    """Base implementation for all commands."""
    command_id: CommandId
    parameters: CommandParameters
    
    @abstractmethod
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Implementation-specific execution logic."""
    
    @require(lambda self, context: self.validate() and context.has_permissions(self.get_required_permissions()))
    @ensure(lambda result: result.execution_time < MAX_COMMAND_DURATION)
    def execute(self, context: ExecutionContext) -> CommandResult:
        """Execute command with contract enforcement."""
        with security_context(context, self.get_required_permissions()):
            return self._execute_impl(context)
```

#### text.py - Text Commands
```python
@dataclass(frozen=True)
class TypeTextCommand(BaseCommand):
    """Safely type text with input validation."""
    text: str
    typing_speed: TypingSpeed = TypingSpeed.NORMAL
    
    @require(lambda self: len(self.text) <= MAX_TEXT_LENGTH)
    @require(lambda self: is_safe_text_content(self.text))
    @ensure(lambda result: result.characters_typed == len(self.text))
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Type text with security validation."""
        
@dataclass(frozen=True)  
class FindTextCommand(BaseCommand):
    """Find text in active application."""
    search_pattern: str
    case_sensitive: bool = False
    
    @require(lambda self: is_valid_search_pattern(self.search_pattern))
    @ensure(lambda result: result.match_count >= 0)
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Find text with pattern validation."""
```

#### system.py - System Commands
```python
@dataclass(frozen=True)
class PauseCommand(BaseCommand):
    """Pause execution for specified duration."""
    duration: Duration
    
    @require(lambda self: self.duration > Duration.ZERO and self.duration <= MAX_PAUSE_DURATION)
    @ensure(lambda result: result.actual_duration >= self.duration)
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Pause with duration validation."""

@dataclass(frozen=True)
class PlaySoundCommand(BaseCommand):
    """Play system sound safely."""
    sound_name: SoundName
    volume: Volume = Volume.DEFAULT
    
    @require(lambda self: self.sound_name in ALLOWED_SOUNDS)
    @require(lambda self: self.volume.is_valid())
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Play sound with security restrictions."""
```

#### validation.py - Security Validation
```python
def validate_text_input(text: str) -> bool:
    """Validate text input for security threats."""
    if len(text) > MAX_TEXT_LENGTH:
        return False
    if contains_script_injection(text):
        return False
    if contains_system_commands(text):
        return False
    return True

def validate_file_path(path: str) -> bool:
    """Validate file path for directory traversal attacks."""
    normalized = os.path.normpath(path)
    if '..' in normalized:
        return False
    if not normalized.startswith(ALLOWED_BASE_PATHS):
        return False
    return True

@require(lambda command_type: command_type in REGISTERED_COMMANDS)
def validate_command_parameters(command_type: CommandType, parameters: dict) -> bool:
    """Validate command parameters against security policies."""
```

#### registry.py - Command Registration
```python
from typing import Dict, Type, FrozenSet

class CommandRegistry:
    """Type-safe command registration and discovery."""
    
    def __init__(self):
        self._commands: Dict[CommandType, Type[BaseCommand]] = {}
        self._permissions: Dict[CommandType, FrozenSet[Permission]] = {}
    
    @require(lambda self, command_type, command_class: issubclass(command_class, BaseCommand))
    def register_command(self, command_type: CommandType, command_class: Type[BaseCommand]) -> None:
        """Register new command type with validation."""
        
    @require(lambda self, command_type: command_type in self._commands)
    def create_command(self, command_type: CommandType, parameters: dict) -> BaseCommand:
        """Create command instance with parameter validation."""
```

## 🏗️ Modularity Strategy
- **base.py**: Command contracts and base implementation (target: 90 lines)
- **text.py**: Text manipulation commands (target: 175 lines)  
- **system.py**: System operation commands (target: 175 lines)
- **application.py**: Application control commands (target: 175 lines)
- **flow.py**: Flow control commands (target: 125 lines)
- **validation.py**: Security validation utilities (target: 125 lines)
- **registry.py**: Command registration system (target: 90 lines)

## ✅ Success Criteria
- All commands implement contract-based validation
- Security boundaries prevent malicious operations
- Type safety prevents runtime command errors
- Comprehensive test coverage for all command types
- TESTING.md updated with command library test results
- Performance: Command validation < 5ms, execution varies by command type
- Documentation: ABOUT.md explains command architecture and security model
- Modularity: All files within size constraints
- Extensibility: Easy registration of new command types
- Zero security vulnerabilities in command implementations
- Zero regressions: All existing tests continue passing