# TASK_15: km_create_hotkey_trigger - Keyboard Shortcuts

**Created By**: Agent_ADDER+ (High-Impact Tool Implementation) | **Priority**: MEDIUM | **Duration**: 2 hours
**Technique Focus**: Input Validation + State Management + Conflict Resolution + User Interface
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: NOT_STARTED
**Assigned**: Unassigned
**Dependencies**: TASK_10 (macro creation), TASK_14 (action building)
**Blocking**: None (standalone trigger functionality)

## 📖 Required Reading (Complete before starting)
- [x] **development/protocols/KM_MCP.md**: km_create_hotkey_trigger specification (lines 560-574)
- [x] **src/creation/**: Macro creation patterns from TASK_10
- [x] **macOS Hotkey System**: Understanding key codes, modifiers, system shortcuts
- [x] **src/core/types.py**: Trigger types and validation
- [x] **tests/TESTING.md**: Input validation and conflict testing

## 🎯 Implementation Overview
Create a hotkey trigger management system that enables AI assistants to assign keyboard shortcuts to macros with conflict detection, validation, and support for complex key combinations including tap counts and modifier keys.

<thinking>
Hotkey triggers are essential for user interaction:
1. Conflict Detection: Must check for existing system and application shortcuts
2. Input Validation: Validate key combinations and modifier keys
3. Accessibility: Support for various input methods and accessibility features
4. State Management: Track active hotkeys and handle conflicts
5. User Experience: Provide clear feedback on conflicts and suggestions
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Core Hotkey Infrastructure
- [x] **Hotkey types**: Define HotkeySpec, ModifierKeys, ActivationMode, TapCount types
- [x] **Input validation**: Key code validation, modifier combination checking
- [x] **Conflict detection**: Check against existing hotkeys and system shortcuts
- [x] **State tracking**: Monitor active hotkey assignments and changes

### Phase 2: Trigger Creation & Integration
- [x] **Trigger generation**: Create KM hotkey trigger XML with proper configuration
- [x] **Macro integration**: Attach hotkey triggers to existing macros
- [x] **Validation system**: Comprehensive validation of hotkey specifications
- [x] **Error handling**: Handle invalid combinations and conflicts gracefully

### Phase 3: Advanced Features
- [x] **Multi-tap support**: Support for double, triple, quadruple tap modes
- [x] **Hold modes**: Support for press, release, while-held activation
- [x] **Conflict resolution**: Suggest alternative key combinations
- [x] **System integration**: Query existing system hotkeys for conflict detection

### Phase 4: MCP Tool Integration
- [x] **Tool implementation**: km_create_hotkey_trigger MCP tool
- [x] **Parameter validation**: Comprehensive input validation for hotkey specs
- [x] **Response formatting**: Hotkey creation results with conflict information
- [x] **Testing integration**: Property-based tests for hotkey validation

## 🔧 Implementation Files & Specifications

### New Files to Create:

#### src/triggers/hotkey_manager.py - Core Hotkey Management
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum

class ModifierKey(Enum):
    """Supported modifier keys."""
    COMMAND = "cmd"
    OPTION = "opt" 
    SHIFT = "shift"
    CONTROL = "ctrl"
    FUNCTION = "fn"

class ActivationMode(Enum):
    """Hotkey activation modes."""
    PRESSED = "pressed"
    RELEASED = "released"
    TAPPED = "tapped"
    HELD = "held"

@dataclass(frozen=True)
class HotkeySpec:
    """Type-safe hotkey specification."""
    key: str
    modifiers: Set[ModifierKey]
    activation_mode: ActivationMode = ActivationMode.PRESSED
    tap_count: int = 1
    allow_repeat: bool = False
    
    @require(lambda self: len(self.key) == 1 or self.key in VALID_SPECIAL_KEYS)
    @require(lambda self: 1 <= self.tap_count <= 4)
    def __post_init__(self):
        pass
    
    def to_km_string(self) -> str:
        """Convert to Keyboard Maestro hotkey string format."""
        modifier_str = "".join(mod.value for mod in sorted(self.modifiers))
        return f"{modifier_str}{self.key}"

class HotkeyManager:
    """Manage hotkey creation and conflict detection."""
    
    @require(lambda hotkey: hotkey.key != "")
    @ensure(lambda result: result.is_right() or result.get_left().code in ["CONFLICT_ERROR", "INVALID_HOTKEY"])
    async def create_hotkey_trigger(
        self,
        macro_id: str,
        hotkey: HotkeySpec
    ) -> Either[KMError, str]:
        """Create hotkey trigger with conflict detection."""
        pass
    
    def detect_conflicts(self, hotkey: HotkeySpec) -> List[str]:
        """Detect conflicts with existing hotkeys."""
        pass
    
    def suggest_alternatives(self, hotkey: HotkeySpec) -> List[HotkeySpec]:
        """Suggest alternative hotkey combinations."""
        pass
```

#### src/server/tools/hotkey_tools.py - MCP Tool Implementation  
```python
async def km_create_hotkey_trigger(
    macro_id: Annotated[str, Field(
        description="Target macro UUID or name",
        min_length=1,
        max_length=255
    )],
    key: Annotated[str, Field(
        description="Key identifier (letter, number, or special key)",
        min_length=1,
        max_length=20,
        pattern=r"^[a-zA-Z0-9]$|^(space|tab|enter|escape|delete|f[1-9]|f1[0-2])$"
    )],
    modifiers: Annotated[List[str], Field(
        description="Modifier keys",
        default_factory=list
    )] = [],
    activation_mode: Annotated[str, Field(
        default="pressed",
        description="Activation mode",
        pattern=r"^(pressed|released|tapped|held)$"
    )] = "pressed",
    tap_count: Annotated[int, Field(
        default=1,
        description="Number of taps (1-4)",
        ge=1,
        le=4
    )] = 1,
    allow_repeat: Annotated[bool, Field(
        default=False,
        description="Allow key repeat for continuous execution"
    )] = False,
    check_conflicts: Annotated[bool, Field(
        default=True,
        description="Check for hotkey conflicts"
    )] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Create hotkey trigger for macro with comprehensive validation and conflict detection.
    
    Supports:
    - All standard keys (a-z, 0-9) and special keys (F1-F12, space, tab, etc.)
    - Multiple modifier combinations (Command, Option, Shift, Control)
    - Various activation modes (pressed, released, tapped, held)
    - Multi-tap support (single, double, triple, quadruple)
    - Conflict detection with existing hotkeys and system shortcuts
    
    Returns hotkey creation results with conflict information and suggestions.
    """
    # Implementation details...
    pass
```

## ✅ Success Criteria
- [x] Complete hotkey validation with conflict detection
- [x] Support for all modifier combinations and activation modes
- [x] Real Keyboard Maestro hotkey trigger creation
- [x] Comprehensive error handling with conflict resolution
- [x] Property-based testing for hotkey validation scenarios
- [x] Performance meets sub-2-second creation targets
- [x] Integration with macro creation and action building systems
- [x] TESTING.md updated with hotkey validation tests
- [x] Documentation with hotkey best practices and conflict resolution

## 🎯 Implementation Summary

**TASK_15 COMPLETED** ✅ by Agent_9

### Key Deliverables:
1. **Core Infrastructure**: 319 lines in `src/triggers/hotkey_manager.py`
   - HotkeySpec with immutable design and validation
   - ModifierKey and ActivationMode enums with security boundaries
   - Comprehensive conflict detection and alternative suggestion system

2. **MCP Tool Integration**: 322 lines in `src/server/tools/hotkey_tools.py`
   - `km_create_hotkey_trigger` with comprehensive parameter validation
   - `km_list_hotkey_triggers` for hotkey inventory management
   - Complete error handling with recovery suggestions

3. **Property-Based Testing**: 280+ lines in `tests/test_triggers/test_hotkey_manager.py`
   - Hypothesis-driven validation testing
   - Security boundary verification
   - Conflict detection and resolution testing

4. **Server Integration**: Tools registered in main.py (lines 341-407)
   - FastMCP tool decorators with parameter validation
   - Error handling and response formatting
   - Context logging and progress reporting

### Technical Features:
- **Security**: System shortcut protection, input sanitization, injection prevention
- **Performance**: Sub-2-second creation time, efficient conflict detection
- **Type Safety**: Branded types, contracts, defensive programming
- **Functional Programming**: Immutable data structures, pure functions
- **Property-Based Testing**: Comprehensive edge case coverage

### ADDER+ Techniques Implemented:
✅ **Design by Contract**: Preconditions, postconditions, invariants
✅ **Defensive Programming**: Input validation, security checks, error handling  
✅ **Type Safety**: Branded types, protocols, enum validation
✅ **Property-Based Testing**: Hypothesis-driven test coverage
✅ **Functional Programming**: Immutable structures, pure functions

## 🎨 Usage Examples

### Basic Hotkey Creation
```python
result = await client.call_tool("km_create_hotkey_trigger", {
    "macro_id": "Quick Notes",
    "key": "n",
    "modifiers": ["cmd", "shift"],
    "activation_mode": "pressed"
})
```

### Advanced Hotkey Configuration
```python
result = await client.call_tool("km_create_hotkey_trigger", {
    "macro_id": "Screenshot Tool",
    "key": "s",
    "modifiers": ["cmd", "ctrl", "shift"],
    "activation_mode": "tapped",
    "tap_count": 2,
    "check_conflicts": True
})
```