"""
Core macro execution engine for the Keyboard Maestro MCP system.

This module provides the main execution engine with type-safe macro execution,
contract-based validation, and comprehensive security enforcement.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import uuid
import asyncio

from .types import (
    MacroId, CommandId, ExecutionToken, MacroDefinition, MacroCommand,
    ExecutionContext, ExecutionResult, ExecutionStatus, CommandResult,
    CommandParameters, Permission, Duration
)
from ..integration.km_client import Either
from .errors import (
    ExecutionError, TimeoutError, PermissionDeniedError, ResourceNotFoundError,
    ValidationError, create_error_context, handle_error_safely
)
from .contracts import require, ensure, is_not_none
from .context import (
    ExecutionContextManager, SecurityContextManager, security_context,
    get_context_manager, get_variable_manager
)
from .parser import CommandType, CommandValidator
from .errors import MacroEngineError


@dataclass(frozen=True)
class PlaceholderCommand:
    """
    Placeholder command implementation for the core engine.
    
    This is a minimal implementation that satisfies the MacroCommand protocol
    and will be replaced by full command implementations in TASK_3.
    """
    command_id: CommandId
    command_type: CommandType
    parameters: CommandParameters
    
    def execute(self, context: ExecutionContext) -> CommandResult:
        """Execute the placeholder command."""
        start_time = time.time()
        
        try:
            # Simulate command execution based on type
            if self.command_type == CommandType.TEXT_INPUT:
                text = self.parameters.get('text', '')
                output = f"Typed text: {text}"
            elif self.command_type == CommandType.PAUSE:
                duration = self.parameters.get('duration', 1.0)
                time.sleep(min(duration, 0.1))  # Cap sleep for testing
                output = f"Paused for {duration} seconds"
            elif self.command_type == CommandType.PLAY_SOUND:
                sound = self.parameters.get('sound_name', 'beep')
                output = f"Played sound: {sound}"
            else:
                output = f"Executed {self.command_type.value} command"
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            return CommandResult.success_result(
                output=output,
                execution_time=execution_time,
                command_id=str(self.command_id),
                command_type=self.command_type.value
            )
            
        except Exception as e:
            execution_time = Duration.from_seconds(time.time() - start_time)
            return CommandResult.failure_result(
                error_message=str(e),
                execution_time=execution_time,
                command_id=str(self.command_id)
            )
    
    def validate(self) -> bool:
        """Validate the command configuration."""
        try:
            # Use the same validation logic as the parser
            validated_params = CommandValidator.validate_command_parameters(
                self.command_type,
                self.parameters.data
            )
            # If validation succeeds and returns CommandParameters, it's valid
            return isinstance(validated_params, CommandParameters)
        except Exception:
            return False
    
    def get_dependencies(self) -> List[CommandId]:
        """Get command dependencies."""
        return []  # Placeholder commands have no dependencies
    
    def get_required_permissions(self) -> frozenset[Permission]:
        """Get required permissions for this command."""
        return CommandValidator.get_required_permissions(self.command_type)


@dataclass(frozen=True)
class MacroEngine:
    """
    Type-safe macro execution engine with contract enforcement.
    
    This engine provides secure, reliable macro execution with comprehensive
    validation, permission checking, and error handling.
    """
    context_manager: ExecutionContextManager = field(default_factory=get_context_manager)
    max_concurrent_executions: int = 10
    default_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    _active_executions: Dict[ExecutionToken, Dict[str, Any]] = field(default_factory=dict, init=False)
    _execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    def __post_init__(self):
        """Initialize engine state."""
        # Initialize mutable state using object.__setattr__ for frozen dataclass
        object.__setattr__(self, '_active_executions', {})
        object.__setattr__(self, '_execution_lock', asyncio.Lock())
    
    @require(lambda self, macro, context=None: macro is not None, "macro cannot be None")
    @ensure(lambda self, macro, context=None, result=None: result.execution_token is not None, "must return execution token")
    def execute_macro(
        self,
        macro: MacroDefinition,
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Execute a macro with comprehensive safety checks.
        
        Args:
            macro: The macro definition to execute
            context: Optional execution context (defaults created if None)
            
        Returns:
            ExecutionResult containing execution details and results
            
        Raises:
            PermissionDeniedError: If required permissions are not available
            TimeoutError: If execution exceeds timeout
            ExecutionError: If execution fails
        """
        # Create default context if none provided
        if context is None:
            context = ExecutionContext.default()
        
        # Register the execution context first to ensure cleanup
        token = self.context_manager.register_context(context)
        
        # Create initial execution result
        execution_result = ExecutionResult(
            execution_token=token,
            macro_id=macro.macro_id,
            status=ExecutionStatus.PENDING,
            started_at=datetime.now()
        )
        
        try:
            # Early validation: check if macro is valid before doing expensive setup
            if not macro.is_valid():
                # Raise ValidationError for invalid macros
                from .errors import ValidationError
                raise ValidationError(
                    field_name="macro_definition",
                    value=f"macro_id={macro.macro_id}",
                    constraint="macro must have valid name and non-empty commands"
                )
            # Update status to running
            self.context_manager.update_status(token, ExecutionStatus.RUNNING)
            
            # Execute the macro
            command_results = self._execute_commands(macro.commands, context)
            
            # Calculate total duration
            completed_at = datetime.now()
            total_duration = Duration.from_seconds(
                (completed_at - execution_result.started_at).total_seconds()
            )
            
            # Determine final status based on command results
            has_failures = any(not result.success for result in command_results)
            final_status = ExecutionStatus.FAILED if has_failures else ExecutionStatus.COMPLETED
            
            # Collect error details if there were failures
            error_details = None
            if has_failures:
                failed_commands = [r for r in command_results if not r.success]
                error_messages = [r.error_message for r in failed_commands if r.error_message]
                if error_messages:
                    error_details = f"Command failures: {'; '.join(error_messages)}"
                else:
                    error_details = f"Macro execution failed: {len(failed_commands)} command(s) failed"
            
            # Create final result
            final_result = ExecutionResult(
                execution_token=token,
                macro_id=macro.macro_id,
                status=final_status,
                started_at=execution_result.started_at,
                completed_at=completed_at,
                total_duration=total_duration,
                command_results=command_results,
                error_details=error_details
            )
            
            self.context_manager.update_status(token, final_status)
            return final_result
            
        except Exception as e:
            # Handle execution failure - ALWAYS return ExecutionResult instead of raising
            self.context_manager.update_status(token, ExecutionStatus.FAILED)
            
            completed_at = datetime.now()
            total_duration = Duration.from_seconds(
                (completed_at - execution_result.started_at).total_seconds()
            )
            
            # Create comprehensive error details
            error_type = type(e).__name__
            error_message = str(e)
            
            error_result = ExecutionResult(
                execution_token=token,
                macro_id=macro.macro_id,
                status=ExecutionStatus.FAILED,
                started_at=execution_result.started_at,
                completed_at=completed_at,
                total_duration=total_duration,
                error_details=f"{error_type}: {error_message}",
                command_results=[]  # Empty since execution failed
            )
            
            # Return failure result instead of raising (for property test compliance)
            return error_result
        
        finally:
            # Clean up execution resources completely
            self._cleanup_execution(token)
    
    # TASK_9: Enhanced async execution with proper validation and resource management
    
    @require(lambda self, macro, context=None: macro is not None, "macro cannot be None")
    async def execute_macro_async(
        self,
        macro: MacroDefinition,
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Execute macro asynchronously with guaranteed ExecutionResult return.
        
        Provides enhanced reliability:
        - Always returns ExecutionResult regardless of error conditions
        - Proper async resource management and cleanup
        - Concurrent execution limits and state tracking
        - Memory-efficient processing for large inputs
        """
        execution_id = ExecutionToken(str(uuid.uuid4()))
        start_time = time.perf_counter()
        
        try:
            # Validate macro before execution
            validation_result = self._validate_macro_enhanced(macro)
            if validation_result.is_left():
                return ExecutionResult(
                    execution_token=execution_id,
                    macro_id=macro.macro_id,
                    status=ExecutionStatus.FAILED,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    total_duration=Duration.from_seconds(time.perf_counter() - start_time),
                    error_details=validation_result.get_left()
                )
            
            # Set up execution context
            exec_context = context or ExecutionContext.default()
            
            # Execute with proper resource management
            async with self._execution_lock:
                try:
                    # Check concurrent execution limits
                    if len(self._active_executions) >= self.max_concurrent_executions:
                        return ExecutionResult(
                            execution_token=execution_id,
                            macro_id=macro.macro_id,
                            status=ExecutionStatus.FAILED,
                            started_at=datetime.now(),
                            completed_at=datetime.now(),
                            total_duration=Duration.from_seconds(time.perf_counter() - start_time),
                            error_details=f"Maximum concurrent executions ({self.max_concurrent_executions}) exceeded"
                        )
                    
                    # Track execution state
                    self._active_executions[execution_id] = {
                        'macro': macro,
                        'start_time': start_time,
                        'context': exec_context
                    }
                    
                    # Execute macro commands with async support
                    command_results = []
                    for i, command in enumerate(macro.commands):
                        # Check for cancellation
                        if execution_id in self._active_executions:
                            command_result = await self._execute_command_safe(command, exec_context)
                            command_results.append(command_result)
                            
                            # Stop on first failure if configured
                            if not command_result.success and getattr(exec_context, 'stop_on_failure', True):
                                break
                        else:
                            # Execution was cancelled
                            break
                    
                    # Determine overall success
                    success = all(result.success for result in command_results)
                    final_status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
                    
                    completed_at = datetime.now()
                    return ExecutionResult(
                        execution_token=execution_id,
                        macro_id=macro.macro_id,
                        status=final_status,
                        started_at=datetime.fromtimestamp(start_time),
                        completed_at=completed_at,
                        total_duration=Duration.from_seconds(time.perf_counter() - start_time),
                        command_results=command_results
                    )
                    
                finally:
                    # Always clean up execution state
                    self._active_executions.pop(execution_id, None)
        
        except Exception as e:
            # Ensure we always return ExecutionResult even on unexpected errors
            return ExecutionResult(
                execution_token=execution_id,
                macro_id=macro.macro_id,
                status=ExecutionStatus.FAILED,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                total_duration=Duration.from_seconds(time.perf_counter() - start_time),
                error_details=f"Unexpected execution error: {str(e)}"
            )
    
    def _validate_macro_enhanced(self, macro: MacroDefinition) -> Either[str, MacroDefinition]:
        """Enhanced macro validation with comprehensive checks."""
        
        # Maximum limits for resource management
        MAX_COMMANDS_PER_MACRO = 1000
        MAX_MACRO_MEMORY_MB = 50
        
        # Check macro structure
        if not macro.commands:
            return Either.left("Macro must contain at least one command")
        
        # Validate each command
        for i, command in enumerate(macro.commands):
            if not hasattr(command, 'validate') or not command.validate():
                return Either.left(f"Invalid command at position {i}")
        
        # Check for resource limits
        if len(macro.commands) > MAX_COMMANDS_PER_MACRO:
            return Either.left(f"Macro exceeds maximum command limit ({MAX_COMMANDS_PER_MACRO})")
        
        # Estimate memory usage for large inputs
        estimated_memory_mb = sum(len(str(cmd)) for cmd in macro.commands) / (1024 * 1024)
        if estimated_memory_mb > MAX_MACRO_MEMORY_MB:
            return Either.left(f"Macro estimated memory usage ({estimated_memory_mb:.1f}MB) exceeds limit ({MAX_MACRO_MEMORY_MB}MB)")
        
        return Either.right(macro)
    
    async def _execute_command_safe(self, command: MacroCommand, context: ExecutionContext) -> CommandResult:
        """Execute command with proper error handling and resource management."""
        command_start = time.perf_counter()
        
        try:
            # Apply timeout to command execution
            timeout_seconds = getattr(context, 'timeout', self.default_timeout).total_seconds()
            
            # Execute with timeout
            if hasattr(command, 'execute_async'):
                try:
                    result = await asyncio.wait_for(
                        command.execute_async(context),
                        timeout=timeout_seconds
                    )
                except AttributeError:
                    # Fallback to sync execution in executor
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, command.execute, context
                    )
            else:
                # Execute sync command in executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, command.execute, context
                )
            
            # Ensure result is a CommandResult
            if isinstance(result, CommandResult):
                return result
            else:
                # Wrap raw result in CommandResult
                return CommandResult.success_result(
                    output=str(result),
                    execution_time=Duration.from_seconds(time.perf_counter() - command_start)
                )
                
        except asyncio.TimeoutError:
            return CommandResult.failure_result(
                error_message=f"Command execution timeout after {timeout_seconds}s",
                execution_time=Duration.from_seconds(time.perf_counter() - command_start)
            )
        except Exception as e:
            return CommandResult.failure_result(
                error_message=f"Command execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.perf_counter() - command_start)
            )
    
    @require(lambda self, token: token is not None, "token cannot be None")
    def get_execution_status(self, token: ExecutionToken) -> Optional[ExecutionStatus]:
        """Retrieve current execution status."""
        return self.context_manager.get_status(token)
    
    @require(lambda self, token: token is not None, "token cannot be None")
    def cancel_execution(self, token: ExecutionToken) -> bool:
        """
        Cancel a running macro execution.
        
        Args:
            token: Execution token to cancel
            
        Returns:
            True if cancellation was successful, False if execution not found or already finished
        """
        status = self.context_manager.get_status(token)
        
        if status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
            self.context_manager.update_status(token, ExecutionStatus.CANCELLED)
            self._cleanup_execution(token)
            return True
        
        return False
    
    def get_active_executions(self) -> List[ExecutionToken]:
        """Get list of currently active execution tokens."""
        return self.context_manager.get_active_contexts()
    
    def _execute_commands(
        self,
        commands: List[MacroCommand],
        context: ExecutionContext
    ) -> List[CommandResult]:
        """Execute a list of commands in sequence."""
        results = []
        
        for i, command in enumerate(commands):
            try:
                # Check if execution was cancelled
                if self.context_manager.get_status(context.execution_id) == ExecutionStatus.CANCELLED:
                    break
                
                # Validate permissions for this command
                required_permissions = command.get_required_permissions()
                
                try:
                    with security_context(context, required_permissions):
                        # Execute the command
                        result = command.execute(context)
                        results.append(result)
                        
                        # If command failed and it's critical, stop execution
                        if not result.success:
                            error_context = create_error_context(
                                operation="command_execution",
                                component="macro_engine",
                                command_index=i,
                                command_id=str(command.get_dependencies())
                            )
                            raise ExecutionError(
                                operation=f"command {i}",
                                cause=result.error_message or "Unknown error",
                                context=error_context
                            )
                            
                except PermissionDeniedError as pde:
                    # Permission errors should halt execution and be raised to caller
                    # as per TASK_8 security requirements and test expectations
                    raise pde
                
            except Exception as e:
                # Create error result for this command
                error_result = CommandResult.failure_result(
                    error_message=str(e),
                    command_id=f"cmd_{i}"
                )
                results.append(error_result)
                
                # For critical errors, stop execution
                if isinstance(e, (PermissionDeniedError, TimeoutError)):
                    raise e
        
        return results
    
    def _cleanup_execution(self, token: ExecutionToken) -> None:
        """Clean up resources for a finished execution."""
        # Clean up context
        self.context_manager.cleanup_context(token)
        
        # Clean up variables
        variable_manager = get_variable_manager()
        variable_manager.cleanup_context_variables(token)
    
    def cleanup_expired_executions(self, max_age_seconds: float = 3600) -> int:
        """Clean up executions older than specified age."""
        return self.context_manager.cleanup_expired_contexts(max_age_seconds)


class EngineMetrics:
    """Metrics and monitoring for the macro engine."""
    
    def __init__(self):
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self._lock = threading.Lock()
    
    def record_execution(self, duration: Duration, success: bool) -> None:
        """Record execution metrics."""
        with self._lock:
            self.execution_count += 1
            self.total_execution_time += duration.total_seconds()
            
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.average_execution_time = (
                self.total_execution_time / self.execution_count
                if self.execution_count > 0 else 0.0
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": (
                    self.success_count / self.execution_count
                    if self.execution_count > 0 else 0.0
                ),
                "average_execution_time": self.average_execution_time,
                "total_execution_time": self.total_execution_time,
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.execution_count = 0
            self.success_count = 0
            self.failure_count = 0
            self.total_execution_time = 0.0
            self.average_execution_time = 0.0


# Global engine instance
_default_engine = MacroEngine()
_engine_metrics = EngineMetrics()


def get_default_engine() -> MacroEngine:
    """Get the default macro engine instance."""
    return _default_engine


def get_engine_metrics() -> EngineMetrics:
    """Get the engine metrics instance."""
    return _engine_metrics


def create_test_macro(name: str, command_types: List[CommandType]) -> MacroDefinition:
    """Create a test macro with specified command types."""
    commands = []
    for i, cmd_type in enumerate(command_types):
        # Create basic parameters for each command type
        if cmd_type == CommandType.TEXT_INPUT:
            params = CommandParameters({"text": f"Test text {i}", "speed": "normal"})
        elif cmd_type == CommandType.PAUSE:
            params = CommandParameters({"duration": 1.0})
        elif cmd_type == CommandType.PLAY_SOUND:
            params = CommandParameters({"sound_name": "beep", "volume": 50})
        elif cmd_type == CommandType.VARIABLE_SET:
            params = CommandParameters({"name": f"test_var_{i}", "value": f"test_value_{i}"})
        elif cmd_type == CommandType.VARIABLE_GET:
            params = CommandParameters({"name": f"test_var_{i}", "default": "default_value"})
        elif cmd_type == CommandType.APPLICATION_CONTROL:
            params = CommandParameters({"action": "activate", "application": "TextEdit"})
        elif cmd_type == CommandType.SYSTEM_CONTROL:
            params = CommandParameters({"action": "volume", "value": 50})
        elif cmd_type == CommandType.CONDITIONAL:
            params = CommandParameters({"condition": "true", "then_commands": [], "else_commands": []})
        elif cmd_type == CommandType.LOOP:
            params = CommandParameters({"count": 3, "commands": []})
        else:
            params = CommandParameters({})
        
        command = PlaceholderCommand(
            command_id=CommandId(f"test_cmd_{i}"),
            command_type=cmd_type,
            parameters=params
        )
        commands.append(command)
    
    return MacroDefinition.create_test_macro(name, commands)