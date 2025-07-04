"""
Hypothesis strategies for property-based testing of the Keyboard Maestro MCP system.

This module provides comprehensive data generators for testing system behavior
across wide input ranges with property-based testing techniques.
"""

from typing import List, Dict, Any, Optional, Union
import string
import re
from hypothesis import strategies as st
from hypothesis.strategies import composite

from src.core import (
    MacroId, CommandId, ExecutionToken, TriggerId, GroupId, VariableName,
    ExecutionContext, MacroDefinition, CommandParameters, Permission,
    Duration, CommandType, ExecutionStatus
)


# Basic type generators
@composite
def macro_ids(draw) -> MacroId:
    """Generate valid macro IDs."""
    # Generate alphanumeric identifiers with underscores
    identifier = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "_-",
        min_size=1,
        max_size=50
    ).filter(lambda x: x and x[0].isalpha()))
    return MacroId(identifier)


@composite
def command_ids(draw) -> CommandId:
    """Generate valid command IDs."""
    identifier = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "_",
        min_size=1,
        max_size=30
    ).filter(lambda x: x and x[0].isalpha()))
    return CommandId(identifier)


@composite
def execution_tokens(draw) -> ExecutionToken:
    """Generate execution tokens (UUID-like strings)."""
    # Generate UUID-like tokens
    parts = [
        draw(st.text(alphabet="0123456789abcdef", min_size=8, max_size=8)),
        draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=4)),
        draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=4)),
        draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=4)),
        draw(st.text(alphabet="0123456789abcdef", min_size=12, max_size=12))
    ]
    return ExecutionToken("-".join(parts))


@composite
def variable_names(draw) -> VariableName:
    """Generate valid variable names."""
    name = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "_",
        min_size=1,
        max_size=100
    ).filter(lambda x: x and x[0].isalpha()))
    return VariableName(name)


# Duration generators
@composite
def durations(draw, min_seconds: float = 0.1, max_seconds: float = 300.0) -> Duration:
    """Generate duration objects within reasonable bounds."""
    seconds = draw(st.floats(
        min_value=min_seconds,
        max_value=max_seconds,
        allow_nan=False,
        allow_infinity=False
    ))
    return Duration.from_seconds(seconds)


# Permission generators
@composite
def permission_sets(draw, min_size: int = 0, max_size: int = 8) -> frozenset[Permission]:
    """Generate sets of permissions."""
    permissions = draw(st.frozensets(
        st.sampled_from(list(Permission)),
        min_size=min_size,
        max_size=max_size
    ))
    return permissions


# Command parameter generators
@composite
def command_parameters(draw, command_type: Optional[CommandType] = None) -> CommandParameters:
    """Generate command parameters based on command type."""
    if command_type == CommandType.TEXT_INPUT:
        text = draw(safe_text_content())
        speed = draw(st.sampled_from(["slow", "normal", "fast"]))
        return CommandParameters({"text": text, "speed": speed})
    
    elif command_type == CommandType.PAUSE:
        duration = draw(st.floats(min_value=0.1, max_value=10.0))
        return CommandParameters({"duration": duration})
    
    elif command_type == CommandType.PLAY_SOUND:
        sound_name = draw(st.sampled_from([
            "beep", "basso", "blow", "bottle", "frog", "funk",
            "glass", "hero", "morse", "ping", "pop", "purr",
            "sosumi", "submarine", "tink"
        ]))
        volume = draw(st.integers(min_value=0, max_value=100))
        return CommandParameters({"sound_name": sound_name, "volume": volume})
    
    elif command_type == CommandType.VARIABLE_SET:
        name = draw(variable_names())
        value = draw(safe_text_content(max_length=1000))
        return CommandParameters({"name": name, "value": value})
    
    elif command_type == CommandType.VARIABLE_GET:
        name = draw(variable_names())
        default = draw(st.one_of(st.none(), safe_text_content(max_length=100)))
        params = {"name": name}
        if default is not None:
            params["default"] = default
        return CommandParameters(params)
    
    else:
        # Generic parameters for other command types
        num_params = draw(st.integers(min_value=0, max_value=5))
        params = {}
        for i in range(num_params):
            key = f"param_{i}"
            value = draw(st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ))
            params[key] = value
        return CommandParameters(params)


# Text content generators
@composite
def safe_text_content(draw, min_length: int = 0, max_length: int = 1000) -> str:
    """Generate safe text content without injection patterns."""
    # Generate text that doesn't contain dangerous patterns
    text = draw(st.text(
        alphabet=string.ascii_letters + string.digits + " .,!?-_()[]{}",
        min_size=min_length,
        max_size=max_length
    ))
    
    # Filter out potentially dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'\.\./',
        r'[<>"\']'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Generate simpler safe text
            return draw(st.text(
                alphabet=string.ascii_letters + string.digits + " ",
                min_size=min_length,
                max_size=min(max_length, 100)
            ))
    
    return text


@composite
def malicious_text_content(draw) -> str:
    """Generate text content with potential security threats."""
    malicious_patterns = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "eval(malicious_code)",
        "exec(dangerous_command)",
        "../../../etc/passwd",
        "'; DROP TABLE users; --",
        "$(rm -rf /)",
        "`cat /etc/shadow`",
        "%SYSTEMROOT%\\system32\\"
    ]
    
    base_pattern = draw(st.sampled_from(malicious_patterns))
    
    # Sometimes combine with legitimate text
    if draw(st.booleans()):
        legitimate_text = draw(st.text(
            alphabet=string.ascii_letters + " ",
            min_size=5,
            max_size=50
        ))
        return f"{legitimate_text} {base_pattern}"
    
    return base_pattern


# Execution context generators
@composite
def execution_contexts(draw, require_permissions: Optional[List[Permission]] = None) -> ExecutionContext:
    """Generate valid execution contexts."""
    # Generate base permissions
    base_permissions = draw(permission_sets(min_size=1, max_size=6))
    
    # Add required permissions if specified
    if require_permissions:
        base_permissions = base_permissions | frozenset(require_permissions)
    
    timeout = draw(durations(min_seconds=1.0, max_seconds=300.0))
    
    # Generate some context variables
    num_vars = draw(st.integers(min_value=0, max_value=5))
    variables = {}
    for i in range(num_vars):
        var_name = draw(variable_names())
        var_value = draw(safe_text_content(max_length=200))
        variables[var_name] = var_value
    
    return ExecutionContext(
        permissions=base_permissions,
        timeout=timeout,
        variables=variables
    )


# Macro definition generators
@composite
def simple_macro_definitions(draw) -> MacroDefinition:
    """Generate simple macro definitions with 1-3 commands."""
    macro_id = draw(macro_ids())
    name = draw(st.text(
        alphabet=string.ascii_letters + string.digits + " _-",
        min_size=1,
        max_size=100
    ))
    
    # Generate 1-3 commands
    num_commands = draw(st.integers(min_value=1, max_value=3))
    command_types = draw(st.lists(
        st.sampled_from([
            CommandType.TEXT_INPUT,
            CommandType.PAUSE,
            CommandType.PLAY_SOUND
        ]),
        min_size=num_commands,
        max_size=num_commands
    ))
    
    # For now, we'll use placeholder commands from the engine
    from src.core.engine import PlaceholderCommand
    commands = []
    for i, cmd_type in enumerate(command_types):
        command_id = CommandId(f"cmd_{i}")
        parameters = draw(command_parameters(cmd_type))
        command = PlaceholderCommand(
            command_id=command_id,
            command_type=cmd_type,
            parameters=parameters
        )
        commands.append(command)
    
    enabled = draw(st.booleans())
    description = draw(st.one_of(
        st.none(),
        st.text(max_size=500)
    ))
    
    return MacroDefinition(
        macro_id=macro_id,
        name=name,
        commands=commands,
        enabled=enabled,
        description=description
    )


@composite
def complex_macro_definitions(draw) -> MacroDefinition:
    """Generate complex macro definitions with many commands."""
    macro_id = draw(macro_ids())
    name = draw(st.text(
        alphabet=string.ascii_letters + string.digits + " _-",
        min_size=1,
        max_size=100
    ))
    
    # Generate 3-10 commands of various types
    num_commands = draw(st.integers(min_value=3, max_value=10))
    command_types = draw(st.lists(
        st.sampled_from(list(CommandType)),
        min_size=num_commands,
        max_size=num_commands
    ))
    
    from src.core.engine import PlaceholderCommand
    commands = []
    for i, cmd_type in enumerate(command_types):
        command_id = CommandId(f"cmd_{i}")
        parameters = draw(command_parameters(cmd_type))
        command = PlaceholderCommand(
            command_id=command_id,
            command_type=cmd_type,
            parameters=parameters
        )
        commands.append(command)
    
    enabled = draw(st.booleans())
    description = draw(st.one_of(
        st.none(),
        st.text(max_size=1000)
    ))
    
    return MacroDefinition(
        macro_id=macro_id,
        name=name,
        commands=commands,
        enabled=enabled,
        description=description
    )


# Input validation test generators
@composite
def invalid_identifiers(draw) -> str:
    """Generate invalid identifiers for validation testing."""
    invalid_types = [
        # Empty strings
        "",
        # Too long
        "a" * 1000,
        # Special characters
        "macro<script>",
        "test;DROP TABLE",
        "name with\nnewlines",
        "name\twith\ttabs",
        # Starting with numbers or special chars
        "123macro",
        "_underscore_start",
        "-dash-start",
        # Injection patterns
        "../traversal",
        "eval()",
        "system()",
    ]
    
    return draw(st.sampled_from(invalid_types))


@composite
def edge_case_durations(draw) -> Duration:
    """Generate edge case durations for testing."""
    edge_cases = [
        0.0,      # Zero duration
        -1.0,     # Negative (should be invalid)
        0.001,    # Very small
        86400.0,  # Very large (24 hours)
        float('inf'),  # Infinity
    ]
    
    duration_value = draw(st.sampled_from(edge_cases))
    try:
        return Duration.from_seconds(duration_value)
    except ValueError:
        # For invalid durations, return a valid one
        return Duration.from_seconds(1.0)


# Generators for specific test scenarios
@composite
def concurrent_execution_scenarios(draw) -> List[Dict[str, Any]]:
    """Generate scenarios for concurrent execution testing."""
    num_scenarios = draw(st.integers(min_value=2, max_value=10))
    scenarios = []
    
    for i in range(num_scenarios):
        scenario = {
            'macro': draw(simple_macro_definitions()),
            'context': draw(execution_contexts()),
            'delay': draw(st.floats(min_value=0.0, max_value=0.1))
        }
        scenarios.append(scenario)
    
    return scenarios


# Utility functions
def is_safe_for_validation(text: str) -> bool:
    """Check if text is safe for validation testing."""
    dangerous_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'eval\s*\(',
        r'\.\./',
        r'system\s*\(',
        r'exec\s*\(',
    ]
    
    return not any(re.search(pattern, text, re.IGNORECASE) for pattern in dangerous_patterns)


def contains_injection_pattern(text: str) -> bool:
    """Check if text contains potential injection patterns."""
    return not is_safe_for_validation(text)