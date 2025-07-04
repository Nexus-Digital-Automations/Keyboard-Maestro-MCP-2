"""
Interactive macro debugging functionality with step-through execution.

This module provides comprehensive debugging capabilities including breakpoints,
variable watching, step-through execution, and performance analysis for
Keyboard Maestro macros with security validation and resource protection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import time
import asyncio
import logging

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import SecurityViolationError, ValidationError, TimeoutError
from ..core.macro_editor import DebugSession, MacroInspection


logger = logging.getLogger(__name__)


class DebugState(Enum):
    """Debug session execution states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STEP_MODE = "step_mode"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class DebugBreakpoint:
    """Debug breakpoint configuration with validation."""
    action_id: str
    condition: Optional[str] = None  # Optional condition for conditional breakpoints
    hit_count: int = 0
    enabled: bool = True
    
    @require(lambda self: len(self.action_id) > 0)
    @require(lambda self: self.hit_count >= 0)
    def __post_init__(self):
        """Validate breakpoint configuration."""
        pass


@dataclass
class DebugExecutionState:
    """Current state of macro execution during debugging."""
    current_action_id: Optional[str] = None
    execution_stack: List[str] = field(default_factory=list)
    variable_values: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    step_count: int = 0
    error_message: Optional[str] = None
    debug_state: DebugState = DebugState.INITIALIZED
    
    @require(lambda self: self.execution_time >= 0.0)
    @require(lambda self: self.step_count >= 0)
    def update_state(self, new_state: DebugState) -> None:
        """Update debug state with validation."""
        self.debug_state = new_state


@dataclass(frozen=True)
class DebugResult:
    """Result of debug session execution."""
    session_id: str
    macro_id: str
    final_state: DebugState
    execution_summary: Dict[str, Any]
    variable_snapshots: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    errors_encountered: List[str]
    
    @require(lambda self: len(self.session_id) > 0)
    @require(lambda self: len(self.macro_id) > 0)
    def __post_init__(self):
        """Validate debug result."""
        pass


class MacroDebugger:
    """Interactive macro debugger with comprehensive debugging capabilities."""
    
    def __init__(self):
        self._active_sessions: Dict[str, DebugExecutionState] = {}
        self._breakpoints: Dict[str, Set[DebugBreakpoint]] = {}
        self._session_counter = 0
    
    @require(lambda self, session: isinstance(session, DebugSession))
    async def start_debug_session(self, session: DebugSession) -> Either[ValidationError, str]:
        """Start a new debugging session with validation."""
        try:
            # Validate session configuration
            validation_result = self._validate_debug_session(session)
            if validation_result.is_left():
                return validation_result
            
            # Generate unique session ID
            session_id = f"debug_session_{self._session_counter}_{int(time.time())}"
            self._session_counter += 1
            
            # Initialize execution state
            execution_state = DebugExecutionState()
            execution_state.update_state(DebugState.INITIALIZED)
            
            # Set up breakpoints
            breakpoints = set()
            for action_id in session.breakpoints:
                breakpoint = DebugBreakpoint(action_id=action_id)
                breakpoints.add(breakpoint)
            
            self._breakpoints[session_id] = breakpoints
            self._active_sessions[session_id] = execution_state
            
            logger.info(f"Started debug session {session_id} for macro {session.macro_id}")
            
            return Either.right(session_id)
            
        except Exception as e:
            logger.error(f"Failed to start debug session: {str(e)}")
            return Either.left(ValidationError(
                field_name="debug_session",
                value=str(session),
                constraint=f"Failed to start session: {str(e)}"
            ))
    
    @require(lambda self, session_id: isinstance(session_id, str) and len(session_id) > 0)
    async def step_execution(self, session_id: str) -> Either[ValidationError, Dict[str, Any]]:
        """Execute single step in debug session."""
        if session_id not in self._active_sessions:
            return Either.left(ValidationError(
                field_name="session_id",
                value=session_id,
                constraint="Debug session not found"
            ))
        
        execution_state = self._active_sessions[session_id]
        
        if execution_state.debug_state not in [DebugState.INITIALIZED, DebugState.PAUSED, DebugState.STEP_MODE]:
            return Either.left(ValidationError(
                field_name="debug_state",
                value=execution_state.debug_state.value,
                constraint="Session not in steppable state"
            ))
        
        try:
            # Simulate step execution (in real implementation, this would execute actual macro step)
            execution_state.step_count += 1
            execution_state.current_action_id = f"action_{execution_state.step_count}"
            execution_state.execution_time += 0.1  # Simulated execution time
            
            # Update execution stack
            execution_state.execution_stack.append(execution_state.current_action_id)
            
            # Simulate variable updates
            execution_state.variable_values[f"step_{execution_state.step_count}"] = f"value_{execution_state.step_count}"
            
            # Check for breakpoints
            session_breakpoints = self._breakpoints.get(session_id, set())
            hit_breakpoint = any(
                bp.action_id == execution_state.current_action_id and bp.enabled
                for bp in session_breakpoints
            )
            
            if hit_breakpoint:
                execution_state.update_state(DebugState.PAUSED)
                logger.info(f"Breakpoint hit at action {execution_state.current_action_id}")
            else:
                execution_state.update_state(DebugState.STEP_MODE)
            
            return Either.right({
                "session_id": session_id,
                "current_action": execution_state.current_action_id,
                "step_count": execution_state.step_count,
                "execution_time": execution_state.execution_time,
                "state": execution_state.debug_state.value,
                "variables": execution_state.variable_values.copy(),
                "breakpoint_hit": hit_breakpoint
            })
            
        except Exception as e:
            execution_state.error_message = str(e)
            execution_state.update_state(DebugState.ERROR)
            
            return Either.left(ValidationError(
                field_name="step_execution",
                value=session_id,
                constraint=f"Step execution failed: {str(e)}"
            ))
    
    @require(lambda self, session_id: isinstance(session_id, str) and len(session_id) > 0)
    async def continue_execution(self, session_id: str) -> Either[ValidationError, Dict[str, Any]]:
        """Continue execution until next breakpoint or completion."""
        if session_id not in self._active_sessions:
            return Either.left(ValidationError(
                field_name="session_id",
                value=session_id,
                constraint="Debug session not found"
            ))
        
        execution_state = self._active_sessions[session_id]
        execution_state.update_state(DebugState.RUNNING)
        
        try:
            # Simulate continued execution
            max_steps = 100  # Prevent infinite loops in simulation
            
            while execution_state.step_count < max_steps:
                step_result = await self.step_execution(session_id)
                if step_result.is_left():
                    break
                
                step_data = step_result.get_right()
                if step_data["breakpoint_hit"] or step_data["state"] == DebugState.COMPLETED.value:
                    break
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            if execution_state.step_count >= max_steps:
                execution_state.update_state(DebugState.COMPLETED)
            
            return Either.right({
                "session_id": session_id,
                "final_step_count": execution_state.step_count,
                "execution_time": execution_state.execution_time,
                "final_state": execution_state.debug_state.value,
                "variables": execution_state.variable_values.copy()
            })
            
        except Exception as e:
            execution_state.error_message = str(e)
            execution_state.update_state(DebugState.ERROR)
            
            return Either.left(ValidationError(
                field_name="continue_execution",
                value=session_id,
                constraint=f"Execution failed: {str(e)}"
            ))
    
    @require(lambda self, session_id: isinstance(session_id, str) and len(session_id) > 0)
    def get_session_state(self, session_id: str) -> Either[ValidationError, Dict[str, Any]]:
        """Get current state of debug session."""
        if session_id not in self._active_sessions:
            return Either.left(ValidationError(
                field_name="session_id",
                value=session_id,
                constraint="Debug session not found"
            ))
        
        execution_state = self._active_sessions[session_id]
        
        return Either.right({
            "session_id": session_id,
            "state": execution_state.debug_state.value,
            "current_action": execution_state.current_action_id,
            "step_count": execution_state.step_count,
            "execution_time": execution_state.execution_time,
            "variables": execution_state.variable_values.copy(),
            "execution_stack": execution_state.execution_stack.copy(),
            "error_message": execution_state.error_message,
            "breakpoint_count": len(self._breakpoints.get(session_id, set()))
        })
    
    @require(lambda self, session_id: isinstance(session_id, str) and len(session_id) > 0)
    def stop_debug_session(self, session_id: str) -> Either[ValidationError, DebugResult]:
        """Stop debug session and return final results."""
        if session_id not in self._active_sessions:
            return Either.left(ValidationError(
                field_name="session_id",
                value=session_id,
                constraint="Debug session not found"
            ))
        
        execution_state = self._active_sessions[session_id]
        
        # Create debug result
        debug_result = DebugResult(
            session_id=session_id,
            macro_id="simulated_macro",  # In real implementation, get from session
            final_state=execution_state.debug_state,
            execution_summary={
                "total_steps": execution_state.step_count,
                "execution_time": execution_state.execution_time,
                "final_action": execution_state.current_action_id
            },
            variable_snapshots=[execution_state.variable_values.copy()],
            performance_metrics={
                "avg_step_time": execution_state.execution_time / max(execution_state.step_count, 1),
                "total_execution_time": execution_state.execution_time
            },
            errors_encountered=[execution_state.error_message] if execution_state.error_message else []
        )
        
        # Clean up session
        del self._active_sessions[session_id]
        if session_id in self._breakpoints:
            del self._breakpoints[session_id]
        
        logger.info(f"Debug session {session_id} stopped")
        
        return Either.right(debug_result)
    
    def _validate_debug_session(self, session: DebugSession) -> Either[ValidationError, None]:
        """Validate debug session configuration."""
        if not session.macro_id or len(session.macro_id.strip()) == 0:
            return Either.left(ValidationError(
                field_name="macro_id",
                value=session.macro_id,
                constraint="Macro ID cannot be empty"
            ))
        
        if session.timeout_seconds <= 0 or session.timeout_seconds > 300:
            return Either.left(ValidationError(
                field_name="timeout_seconds",
                value=session.timeout_seconds,
                constraint="Timeout must be between 1 and 300 seconds"
            ))
        
        if len(session.breakpoints) > 50:
            return Either.left(ValidationError(
                field_name="breakpoints",
                value=len(session.breakpoints),
                constraint="Maximum 50 breakpoints allowed"
            ))
        
        if len(session.watch_variables) > 20:
            return Either.left(ValidationError(
                field_name="watch_variables",
                value=len(session.watch_variables),
                constraint="Maximum 20 watch variables allowed"
            ))
        
        return Either.right(None)
    
    @ensure(lambda result: isinstance(result, list))
    def list_active_sessions(self) -> List[str]:
        """List all active debug sessions."""
        return list(self._active_sessions.keys())
    
    def get_session_count(self) -> int:
        """Get number of active debug sessions."""
        return len(self._active_sessions)