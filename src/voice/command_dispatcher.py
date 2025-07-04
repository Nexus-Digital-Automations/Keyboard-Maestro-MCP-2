"""
Voice Command Dispatcher - TASK_66 Phase 2 Core Voice Engine

Voice command execution, automation workflow triggering, and comprehensive
command routing with security validation and performance optimization.

Architecture: Command Router + Execution Engine + Workflow Integration + Security Layer
Performance: <100ms command validation, <500ms execution start, <2s workflow trigger
Security: Command authorization, execution validation, safe parameter handling
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass, field
import asyncio
import logging
import json
import uuid
from enum import Enum

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.voice_architecture import (
    VoiceCommand, VoiceCommandExecution, VoiceCommandType, CommandPriority,
    VoiceCommandId, SpeakerId, VoiceCommandError, VoiceControlError,
    VoiceProfile
)

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Command execution status."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class CommandHandler:
    """Base class for voice command handlers."""
    
    def __init__(self, command_type: VoiceCommandType, handler_name: str):
        self.command_type = command_type
        self.handler_name = handler_name
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.average_execution_time = 0.0
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if handler can process the command."""
        return command.command_type == self.command_type
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute the voice command."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    async def validate_command(self, command: VoiceCommand) -> Either[VoiceCommandError, None]:
        """Validate command before execution."""
        return Either.success(None)
    
    def update_stats(self, execution_time: float, success: bool):
        """Update handler execution statistics."""
        self.execution_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update average execution time
        self.average_execution_time = (
            (self.average_execution_time * (self.execution_count - 1)) + execution_time
        ) / self.execution_count


@dataclass
class PendingCommand:
    """Command pending execution or confirmation."""
    command: VoiceCommand
    created_at: datetime
    expires_at: datetime
    confirmation_required: bool = False
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if command has expired."""
        return datetime.now(UTC) > self.expires_at
    
    def needs_confirmation(self) -> bool:
        """Check if command requires confirmation."""
        return self.confirmation_required or self.command.requires_confirmation


class AutomationHandler(CommandHandler):
    """Handler for automation trigger commands."""
    
    def __init__(self):
        super().__init__(VoiceCommandType.AUTOMATION_TRIGGER, "automation_handler")
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute automation trigger command."""
        try:
            automation_name = command.get_parameter("automation_name")
            if not automation_name:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "Missing automation name"
                ))
            
            # Placeholder for actual automation execution
            # In real implementation, this would integrate with the automation system
            logger.info(f"Triggering automation: {automation_name}")
            
            # Simulate automation execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            result = {
                "automation_name": automation_name,
                "execution_status": "triggered",
                "execution_time": datetime.now(UTC).isoformat(),
                "result": f"Automation '{automation_name}' has been triggered successfully"
            }
            
            return Either.success(result)
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, str(e)
            ))


class ApplicationControlHandler(CommandHandler):
    """Handler for application control commands."""
    
    def __init__(self):
        super().__init__(VoiceCommandType.APPLICATION_CONTROL, "application_handler")
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute application control command."""
        try:
            application = command.get_parameter("application")
            if not application:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "Missing application name"
                ))
            
            intent = command.intent
            
            if intent == "open_application":
                result = await self._open_application(application)
            elif intent == "close_application":
                result = await self._close_application(application)
            else:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, f"Unknown application intent: {intent}"
                ))
            
            return Either.success(result)
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, str(e)
            ))
    
    async def _open_application(self, app_name: str) -> Dict[str, Any]:
        """Open application."""
        # Build AppleScript to open application
        applescript = f'tell application "{app_name}" to activate'
        
        # Execute AppleScript (placeholder)
        logger.info(f"Opening application: {app_name}")
        
        return {
            "application": app_name,
            "action": "open",
            "status": "success",
            "result": f"Application '{app_name}' opened successfully"
        }
    
    async def _close_application(self, app_name: str) -> Dict[str, Any]:
        """Close application."""
        # Build AppleScript to quit application
        applescript = f'tell application "{app_name}" to quit'
        
        # Execute AppleScript (placeholder)
        logger.info(f"Closing application: {app_name}")
        
        return {
            "application": app_name,
            "action": "close",
            "status": "success",
            "result": f"Application '{app_name}' closed successfully"
        }


class SystemControlHandler(CommandHandler):
    """Handler for system control commands."""
    
    def __init__(self):
        super().__init__(VoiceCommandType.SYSTEM_CONTROL, "system_handler")
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute system control command."""
        try:
            intent = command.intent
            
            if intent == "system_volume":
                return await self._handle_volume_control(command)
            elif intent == "system_display":
                return await self._handle_display_control(command)
            elif intent == "file_operation":
                return await self._handle_file_operation(command)
            else:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, f"Unknown system intent: {intent}"
                ))
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, str(e)
            ))
    
    async def _handle_volume_control(self, command: VoiceCommand) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Handle volume control commands."""
        volume = command.get_parameter("volume")
        direction = command.get_parameter("direction")
        action = command.get_parameter("action")
        
        if volume:
            # Set specific volume level
            logger.info(f"Setting volume to {volume}")
            result = f"Volume set to {volume}%"
        elif direction:
            # Adjust volume up/down
            logger.info(f"Turning volume {direction}")
            result = f"Volume turned {direction}"
        elif action:
            # Mute/unmute
            logger.info(f"Volume action: {action}")
            result = f"Audio {action}d"
        else:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, "No volume parameter specified"
            ))
        
        return Either.success({
            "intent": "volume_control",
            "parameters": command.parameters,
            "result": result
        })
    
    async def _handle_display_control(self, command: VoiceCommand) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Handle display control commands."""
        brightness = command.get_parameter("brightness")
        direction = command.get_parameter("direction")
        action = command.get_parameter("action")
        
        if brightness:
            logger.info(f"Setting brightness to {brightness}")
            result = f"Brightness set to {brightness}%"
        elif direction:
            logger.info(f"Turning brightness {direction}")
            result = f"Brightness turned {direction}"
        elif action:
            logger.info(f"Screen action: {action}")
            result = f"Screen {action}ed"
        else:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, "No display parameter specified"
            ))
        
        return Either.success({
            "intent": "display_control",
            "parameters": command.parameters,
            "result": result
        })
    
    async def _handle_file_operation(self, command: VoiceCommand) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Handle file operation commands."""
        action = command.get_parameter("action")
        file_name = command.get_parameter("file_name", "untitled")
        
        if not action:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, "No file action specified"
            ))
        
        logger.info(f"File operation: {action} {file_name}")
        
        return Either.success({
            "intent": "file_operation",
            "action": action,
            "file_name": file_name,
            "result": f"File operation '{action}' executed for '{file_name}'"
        })


class TextInputHandler(CommandHandler):
    """Handler for text input commands."""
    
    def __init__(self):
        super().__init__(VoiceCommandType.TEXT_INPUT, "text_input_handler")
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute text input command."""
        try:
            text = command.get_parameter("text")
            if not text:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "No text content specified"
                ))
            
            # Sanitize text for typing
            sanitized_text = self._sanitize_text_input(text)
            
            logger.info(f"Typing text: {sanitized_text[:50]}...")
            
            # Placeholder for actual text typing
            # In real implementation, this would send keystrokes to the active application
            
            return Either.success({
                "intent": "type_text",
                "text": sanitized_text,
                "length": len(sanitized_text),
                "result": f"Typed {len(sanitized_text)} characters successfully"
            })
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, str(e)
            ))
    
    def _sanitize_text_input(self, text: str) -> str:
        """Sanitize text for safe input."""
        # Remove potentially dangerous characters
        dangerous_chars = ['`', '$', ';', '&', '|', '<', '>']
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()


class NavigationHandler(CommandHandler):
    """Handler for navigation commands."""
    
    def __init__(self):
        super().__init__(VoiceCommandType.NAVIGATION, "navigation_handler")
    
    async def execute(self, command: VoiceCommand, context: Dict[str, Any]) -> Either[VoiceCommandError, Dict[str, Any]]:
        """Execute navigation command."""
        try:
            direction = command.get_parameter("direction")
            if not direction:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "No navigation direction specified"
                ))
            
            logger.info(f"Navigation: {direction}")
            
            # Placeholder for actual navigation
            # In real implementation, this would send appropriate key combinations
            
            return Either.success({
                "intent": "navigate_direction",
                "direction": direction,
                "result": f"Navigated {direction} successfully"
            })
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, str(e)
            ))


class VoiceCommandDispatcher:
    """
    Comprehensive voice command dispatcher and execution system.
    
    Contracts:
        Preconditions:
            - Commands must be validated for security and authorization
            - Handlers must be registered for all supported command types
            - Execution context must contain required permissions and settings
        
        Postconditions:
            - Command execution results are properly tracked and logged
            - Failed commands provide clear error messages and recovery options
            - Performance metrics are maintained for optimization
        
        Invariants:
            - Command execution order respects priority levels
            - Security boundaries are maintained throughout execution
            - Resource usage is monitored and controlled
    """
    
    def __init__(self):
        self.command_handlers: Dict[VoiceCommandType, CommandHandler] = {}
        self.pending_commands: Dict[VoiceCommandId, PendingCommand] = {}
        self.execution_history: List[VoiceCommandExecution] = []
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.execution_stats = {
            "total_commands": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "commands_by_type": {},
            "commands_by_priority": {}
        }
        
        # Authorization settings
        self.require_confirmation_for_high_priority = True
        self.require_speaker_auth_for_system_commands = True
        self.max_pending_commands = 10
        self.command_timeout = timedelta(minutes=5)
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        # Start background processing if event loop is running
        self.processing_task = None
        try:
            self.processing_task = asyncio.create_task(self._process_command_queue())
        except RuntimeError:
            # No event loop running, will start task later if needed
            pass
    
    def _initialize_default_handlers(self):
        """Initialize default command handlers."""
        handlers = [
            AutomationHandler(),
            ApplicationControlHandler(),
            SystemControlHandler(),
            TextInputHandler(),
            NavigationHandler()
        ]
        
        for handler in handlers:
            self.command_handlers[handler.command_type] = handler
        
        logger.info(f"Initialized {len(handlers)} command handlers")
    
    @require(lambda self, command: command.command_id and command.intent)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def dispatch_command(
        self,
        command: VoiceCommand,
        speaker_profile: Optional[VoiceProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Either[VoiceCommandError, VoiceCommandExecution]:
        """
        Dispatch voice command for execution with comprehensive validation.
        
        Performance:
            - <100ms command validation and routing
            - <500ms execution start for simple commands
            - <2s execution start for complex automation workflows
        """
        try:
            start_time = datetime.now(UTC)
            
            # Validate command
            validation_result = await self._validate_command(command, speaker_profile)
            if validation_result.is_error():
                return validation_result
            
            # Check for duplicate commands
            if command.command_id in self.pending_commands:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "Command already pending"
                ))
            
            # Check pending command limit
            if len(self.pending_commands) >= self.max_pending_commands:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, "Too many pending commands"
                ))
            
            # Get appropriate handler
            handler = self.command_handlers.get(command.command_type)
            if not handler:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command.command_id, f"No handler for command type: {command.command_type.value}"
                ))
            
            # Create execution context
            execution_context = context or {}
            execution_context.update({
                "speaker_profile": speaker_profile,
                "dispatch_time": start_time,
                "handler": handler
            })
            
            # Check if confirmation is required
            if self._requires_confirmation(command, speaker_profile):
                # Add to pending commands for confirmation
                pending_command = PendingCommand(
                    command=command,
                    created_at=start_time,
                    expires_at=start_time + self.command_timeout,
                    confirmation_required=True,
                    execution_context=execution_context
                )
                
                self.pending_commands[command.command_id] = pending_command
                
                # Create execution result indicating confirmation needed
                execution = VoiceCommandExecution(
                    command_id=command.command_id,
                    execution_status="requires_confirmation",
                    result_data={"confirmation_required": True},
                    voice_feedback="Please confirm this command by saying 'yes' or cancel by saying 'no'."
                )
                
                logger.info(f"Command requires confirmation: {command.intent}")
                return Either.success(execution)
            
            # Execute command immediately
            execution_result = await self._execute_command_with_handler(command, handler, execution_context)
            
            # Update statistics
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            self._update_execution_stats(command, execution_time, execution_result.is_success())
            
            return execution_result
            
        except Exception as e:
            error_msg = f"Command dispatch failed: {str(e)}"
            logger.error(error_msg)
            return Either.error(VoiceCommandError.command_execution_failed(command.command_id, str(e)))
    
    async def _validate_command(
        self,
        command: VoiceCommand,
        speaker_profile: Optional[VoiceProfile]
    ) -> Either[VoiceCommandError, None]:
        """Validate command for execution."""
        try:
            # Check speaker authorization for sensitive commands
            if self.require_speaker_auth_for_system_commands and command.command_type == VoiceCommandType.SYSTEM_CONTROL:
                if not speaker_profile or not speaker_profile.requires_authentication():
                    return Either.error(VoiceCommandError.speaker_not_authorized(
                        speaker_profile.speaker_id if speaker_profile else "unknown",
                        command.intent
                    ))
            
            # Get handler and validate command with it
            handler = self.command_handlers.get(command.command_type)
            if handler:
                handler_validation = await handler.validate_command(command)
                if handler_validation.is_error():
                    return handler_validation
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command.command_id, f"Validation failed: {str(e)}"
            ))
    
    def _requires_confirmation(self, command: VoiceCommand, speaker_profile: Optional[VoiceProfile]) -> bool:
        """Check if command requires confirmation."""
        # Command explicitly requires confirmation
        if command.requires_confirmation:
            return True
        
        # High priority commands require confirmation if configured
        if self.require_confirmation_for_high_priority and command.is_high_priority():
            return True
        
        # System control commands require confirmation
        if command.command_type == VoiceCommandType.SYSTEM_CONTROL:
            return True
        
        return False
    
    async def _execute_command_with_handler(
        self,
        command: VoiceCommand,
        handler: CommandHandler,
        context: Dict[str, Any]
    ) -> Either[VoiceCommandError, VoiceCommandExecution]:
        """Execute command with specified handler."""
        try:
            execution_start = datetime.now(UTC)
            
            # Execute command
            result = await handler.execute(command, context)
            
            execution_time = (datetime.now(UTC) - execution_start).total_seconds() * 1000
            
            if result.is_success():
                # Create successful execution
                execution = VoiceCommandExecution(
                    command_id=command.command_id,
                    execution_status="completed",
                    result_data=result.value,
                    execution_time_ms=execution_time,
                    voice_feedback=result.value.get("result", "Command completed successfully")
                )
                
                # Update handler stats
                handler.update_stats(execution_time, True)
                
                logger.info(f"Command executed successfully: {command.intent} ({execution_time:.0f}ms)")
            else:
                # Create failed execution
                execution = VoiceCommandExecution(
                    command_id=command.command_id,
                    execution_status="failed",
                    error_message=str(result.error_value),
                    execution_time_ms=execution_time,
                    voice_feedback=f"Command failed: {str(result.error_value)}"
                )
                
                # Update handler stats
                handler.update_stats(execution_time, False)
                
                logger.error(f"Command execution failed: {command.intent} - {str(result.error_value)}")
            
            # Add to execution history
            self.execution_history.append(execution)
            
            # Limit history size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            return Either.success(execution)
            
        except Exception as e:
            execution_time = (datetime.now(UTC) - execution_start).total_seconds() * 1000
            
            execution = VoiceCommandExecution(
                command_id=command.command_id,
                execution_status="failed",
                error_message=str(e),
                execution_time_ms=execution_time,
                voice_feedback=f"Command execution error: {str(e)}"
            )
            
            handler.update_stats(execution_time, False)
            self.execution_history.append(execution)
            
            return Either.success(execution)
    
    async def confirm_command(self, command_id: VoiceCommandId, confirmed: bool) -> Either[VoiceCommandError, VoiceCommandExecution]:
        """Confirm or cancel a pending command."""
        try:
            if command_id not in self.pending_commands:
                return Either.error(VoiceCommandError.command_execution_failed(
                    command_id, "No pending command found"
                ))
            
            pending_command = self.pending_commands[command_id]
            
            # Check if command has expired
            if pending_command.is_expired():
                del self.pending_commands[command_id]
                return Either.error(VoiceCommandError.command_execution_failed(
                    command_id, "Command has expired"
                ))
            
            if confirmed:
                # Execute the confirmed command
                command = pending_command.command
                context = pending_command.execution_context
                handler = context["handler"]
                
                # Remove from pending
                del self.pending_commands[command_id]
                
                # Execute command
                return await self._execute_command_with_handler(command, handler, context)
            else:
                # Cancel the command
                del self.pending_commands[command_id]
                
                execution = VoiceCommandExecution(
                    command_id=command_id,
                    execution_status="cancelled",
                    voice_feedback="Command cancelled by user"
                )
                
                logger.info(f"Command cancelled: {command_id}")
                return Either.success(execution)
            
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                command_id, f"Confirmation failed: {str(e)}"
            ))
    
    async def _process_command_queue(self):
        """Background task to process command queue."""
        while True:
            try:
                # Clean up expired pending commands
                await self._cleanup_expired_commands()
                
                # Process any queued commands
                # (This would be used for batch processing or delayed execution)
                
                # Sleep to prevent busy waiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Command queue processing error: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_expired_commands(self):
        """Clean up expired pending commands."""
        expired_commands = [
            command_id for command_id, pending in self.pending_commands.items()
            if pending.is_expired()
        ]
        
        for command_id in expired_commands:
            del self.pending_commands[command_id]
            logger.info(f"Expired pending command removed: {command_id}")
    
    def _update_execution_stats(self, command: VoiceCommand, execution_time: float, success: bool):
        """Update command execution statistics."""
        self.execution_stats["total_commands"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total_commands = self.execution_stats["total_commands"]
        current_avg = self.execution_stats["average_execution_time"]
        new_avg = ((current_avg * (total_commands - 1)) + execution_time) / total_commands
        self.execution_stats["average_execution_time"] = new_avg
        
        # Track by command type
        cmd_type = command.command_type.value
        if cmd_type not in self.execution_stats["commands_by_type"]:
            self.execution_stats["commands_by_type"][cmd_type] = 0
        self.execution_stats["commands_by_type"][cmd_type] += 1
        
        # Track by priority
        priority = command.priority.value
        if priority not in self.execution_stats["commands_by_priority"]:
            self.execution_stats["commands_by_priority"][priority] = 0
        self.execution_stats["commands_by_priority"][priority] += 1
    
    async def register_handler(self, handler: CommandHandler) -> Either[VoiceCommandError, None]:
        """Register custom command handler."""
        try:
            self.command_handlers[handler.command_type] = handler
            logger.info(f"Command handler registered: {handler.handler_name}")
            return Either.success(None)
        except Exception as e:
            return Either.error(VoiceCommandError.command_execution_failed(
                "handler_registration", f"Handler registration failed: {str(e)}"
            ))
    
    async def get_pending_commands(self) -> List[Dict[str, Any]]:
        """Get list of pending commands."""
        pending = []
        
        for command_id, pending_command in self.pending_commands.items():
            pending.append({
                "command_id": command_id,
                "intent": pending_command.command.intent,
                "command_type": pending_command.command.command_type.value,
                "created_at": pending_command.created_at.isoformat(),
                "expires_at": pending_command.expires_at.isoformat(),
                "confirmation_required": pending_command.confirmation_required,
                "original_text": pending_command.command.original_text
            })
        
        return pending
    
    async def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent command execution history."""
        history = []
        
        for execution in self.execution_history[-limit:]:
            history.append({
                "command_id": execution.command_id,
                "execution_status": execution.execution_status,
                "execution_time_ms": execution.execution_time_ms,
                "result_data": execution.result_data,
                "error_message": execution.error_message,
                "automation_triggered": execution.automation_triggered
            })
        
        return history
    
    async def get_dispatcher_stats(self) -> Dict[str, Any]:
        """Get command dispatcher statistics."""
        stats = self.execution_stats.copy()
        stats.update({
            "pending_commands": len(self.pending_commands),
            "registered_handlers": len(self.command_handlers),
            "execution_history_size": len(self.execution_history),
            "handler_stats": {
                handler.handler_name: {
                    "execution_count": handler.execution_count,
                    "success_count": handler.success_count,
                    "failure_count": handler.failure_count,
                    "average_execution_time": handler.average_execution_time
                }
                for handler in self.command_handlers.values()
            }
        })
        
        return stats


# Helper functions for command dispatch
def create_emergency_command(intent: str, parameters: Dict[str, Any], original_text: str) -> VoiceCommand:
    """Create emergency priority voice command."""
    from ..core.voice_architecture import create_voice_command_id
    
    return VoiceCommand(
        command_id=create_voice_command_id(),
        command_type=VoiceCommandType.SYSTEM_CONTROL,
        intent=intent,
        parameters=parameters,
        original_text=original_text,
        confidence=1.0,
        priority=CommandPriority.EMERGENCY,
        requires_confirmation=False
    )


def create_automation_command(automation_name: str, original_text: str, confidence: float = 0.8) -> VoiceCommand:
    """Create automation trigger command."""
    from ..core.voice_architecture import create_voice_command_id
    
    return VoiceCommand(
        command_id=create_voice_command_id(),
        command_type=VoiceCommandType.AUTOMATION_TRIGGER,
        intent="trigger_automation",
        parameters={"automation_name": automation_name},
        original_text=original_text,
        confidence=confidence,
        priority=CommandPriority.MEDIUM
    )