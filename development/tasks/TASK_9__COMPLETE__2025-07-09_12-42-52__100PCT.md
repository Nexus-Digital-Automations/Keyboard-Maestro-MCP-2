# TASK_9: Engine Properties and Integration Failures Resolution

**Created By**: Agent_1 (Dynamic Detection) | **Priority**: HIGH | **Duration**: 2 hours
**Technique Focus**: Property-Based Testing + Engine Architecture + Integration Patterns
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_1
**Dependencies**: None (Core engine reliability)
**Blocking**: Test suite validation, engine stability

## 📖 Required Reading (Complete before starting)
- [x] **Error Context**: 10+ engine and integration test failures
- [x] **System Impact**: Core engine reliability and KM integration issues
- [x] **Related Documentation**: src/core/engine.py and integration modules
- [x] **Protocol Compliance**: Property-based testing requirements

## 🎯 Problem Analysis
**Classification**: Integration/Logic
**Location**: src/core/engine.py, src/integration/km_client.py, src/integration/triggers.py
**Impact**: Engine execution reliability and KM integration functionality compromised

<thinking>
Root Cause Analysis:
1. Engine execution not always returning proper ExecutionResult objects
2. Invalid macro rejection logic not working correctly
3. Concurrent execution handling has race conditions
4. Large input handling causing memory or performance issues
5. KM client trigger operations failing due to API integration gaps
6. Resource management integration not properly cleaning up
7. Mock integration not matching real KM client behavior patterns
</thinking>

## ✅ Resolution Subtasks (Sequential completion)

### Phase 1: Engine Core Reliability
- [x] **Execution result consistency**: Ensure all execution paths return ExecutionResult
- [x] **Invalid macro rejection**: Fix macro validation and rejection logic
- [x] **Concurrent execution safety**: Implement proper thread safety and state management
- [x] **Large input handling**: Add memory-efficient processing for large inputs

### Phase 2: Integration Layer Fixes
- [x] **KM client trigger operations**: Fix register/unregister trigger functionality
- [x] **Parameter handling**: Ensure proper parameter passing and escaping
- [x] **Async operation consistency**: Fix async trigger operations and event handling
- [x] **Resource management**: Implement proper cleanup and resource tracking

### Phase 3: Property-Based Test Compliance
- [x] **Property validation**: Ensure all engine properties pass consistently
- [x] **Integration properties**: Fix end-to-end property validation
- [x] **Performance properties**: Address memory and timing property failures
- [x] **TESTING.md update**: Update engine and integration test status

## 🔧 Implementation Files & Specifications

### Core Engine Files to Fix:

#### src/core/engine.py - Engine Execution Reliability
```python
@require(lambda macro: macro is not None and isinstance(macro, Macro))
@ensure(lambda result: isinstance(result, ExecutionResult))
async def execute_macro_async(self, macro: Macro, context: Optional[ExecutionContext] = None) -> ExecutionResult:
    """Execute macro with guaranteed ExecutionResult return."""
    execution_id = MacroExecutionId(str(uuid.uuid4()))
    start_time = time.perf_counter()
    
    try:
        # Validate macro before execution
        validation_result = self._validate_macro(macro)
        if validation_result.is_left():
            return ExecutionResult(
                success=False,
                execution_id=execution_id,
                error=validation_result.get_left(),
                duration=Duration.from_seconds(time.perf_counter() - start_time)
            )
        
        # Set up execution context
        exec_context = context or ExecutionContext.create_default()
        
        # Execute with proper resource management
        async with self._execution_lock:
            try:
                # Track execution state
                self._active_executions[execution_id] = {
                    'macro': macro,
                    'start_time': start_time,
                    'context': exec_context
                }
                
                # Execute macro commands
                command_results = []
                for command in macro.commands:
                    command_result = await self._execute_command_safe(command, exec_context)
                    command_results.append(command_result)
                    
                    # Stop on first failure if configured
                    if not command_result.success and exec_context.stop_on_failure:
                        break
                
                # Determine overall success
                success = all(result.success for result in command_results)
                
                return ExecutionResult(
                    success=success,
                    execution_id=execution_id,
                    results=command_results,
                    duration=Duration.from_seconds(time.perf_counter() - start_time),
                    metadata={'command_count': len(command_results)}
                )
                
            finally:
                # Always clean up execution state
                self._active_executions.pop(execution_id, None)
    
    except Exception as e:
        # Ensure we always return ExecutionResult even on unexpected errors
        logger.exception(f"Unexpected error in macro execution {execution_id}")
        return ExecutionResult(
            success=False,
            execution_id=execution_id,
            error=ExecutionError(f"Unexpected execution error: {str(e)}"),
            duration=Duration.from_seconds(time.perf_counter() - start_time)
        )

def _validate_macro(self, macro: Macro) -> Either[ValidationError, Macro]:
    """Enhanced macro validation with comprehensive checks."""
    # Check macro structure
    if not macro.commands:
        return Either.left(ValidationError("Macro must contain at least one command"))
    
    # Validate each command
    for i, command in enumerate(macro.commands):
        if not isinstance(command, Command):
            return Either.left(ValidationError(f"Invalid command at position {i}"))
        
        # Validate command parameters
        validation_result = command.validate()
        if validation_result.is_left():
            return Either.left(ValidationError(f"Command {i} validation failed: {validation_result.get_left()}"))
    
    # Check for resource limits
    if len(macro.commands) > MAX_COMMANDS_PER_MACRO:
        return Either.left(ValidationError(f"Macro exceeds maximum command limit ({MAX_COMMANDS_PER_MACRO})"))
    
    # Estimate memory usage for large inputs
    estimated_memory = sum(len(str(cmd)) for cmd in macro.commands)
    if estimated_memory > MAX_MACRO_MEMORY:
        return Either.left(ValidationError(f"Macro estimated memory usage exceeds limit"))
    
    return Either.right(macro)

async def _execute_command_safe(self, command: Command, context: ExecutionContext) -> CommandResult:
    """Execute command with proper error handling and resource management."""
    command_start = time.perf_counter()
    
    try:
        # Apply timeout to command execution
        timeout = context.command_timeout or DEFAULT_COMMAND_TIMEOUT
        
        async with asyncio.timeout(timeout.total_seconds()):
            result = await command.execute_async(context)
            
        return CommandResult(
            success=True,
            result=result,
            duration=Duration.from_seconds(time.perf_counter() - command_start)
        )
        
    except asyncio.TimeoutError:
        return CommandResult(
            success=False,
            error=TimeoutError(f"Command execution timeout after {timeout}"),
            duration=Duration.from_seconds(time.perf_counter() - command_start)
        )
    except Exception as e:
        return CommandResult(
            success=False,
            error=ExecutionError(f"Command execution failed: {str(e)}"),
            duration=Duration.from_seconds(time.perf_counter() - command_start)
        )
```

#### src/integration/km_client.py - Integration Reliability
```python
@require(lambda trigger_data: validate_trigger_data(trigger_data).is_right())
@ensure(lambda result: isinstance(result, Either))
async def register_trigger_async(self, trigger_data: Dict[str, Any]) -> Either[KMError, TriggerRegistration]:
    """Register trigger with comprehensive error handling."""
    
    try:
        # Sanitize trigger data
        sanitized_data = sanitize_trigger_data(trigger_data)
        if sanitized_data.is_left():
            return sanitized_data
        
        trigger_safe = sanitized_data.get_right()
        
        # Generate unique trigger ID
        trigger_id = TriggerId(str(uuid.uuid4()))
        
        # Build AppleScript for trigger registration
        script_result = self._build_trigger_script(trigger_safe, trigger_id)
        if script_result.is_left():
            return script_result
        
        script = script_result.get_right()
        
        # Execute with proper timeout and error handling
        execution_result = await self._execute_applescript_safe(script)
        if execution_result.is_left():
            return execution_result
        
        # Parse result and create registration
        registration = TriggerRegistration(
            trigger_id=trigger_id,
            trigger_data=trigger_safe,
            registration_time=datetime.utcnow(),
            status=TriggerStatus.ACTIVE
        )
        
        # Store in local registry
        self._trigger_registry[trigger_id] = registration
        
        return Either.right(registration)
        
    except Exception as e:
        logger.exception(f"Unexpected error in trigger registration: {e}")
        return Either.left(KMError.execution_error(f"Trigger registration failed: {str(e)}"))

def _build_trigger_script(self, trigger_data: Dict[str, Any], trigger_id: TriggerId) -> Either[KMError, str]:
    """Build AppleScript for trigger registration with validation."""
    
    trigger_type = trigger_data.get('trigger_type', 'hotkey')
    
    if trigger_type == 'hotkey':
        key = trigger_data.get('key', '')
        modifiers = trigger_data.get('modifiers', [])
        
        # Validate key and modifiers
        if not key or not isinstance(key, str):
            return Either.left(KMError.validation_error("Invalid or missing hotkey"))
        
        # Escape values for AppleScript
        escaped_key = key.replace('"', '\\"')
        escaped_modifiers = [mod.replace('"', '\\"') for mod in modifiers if isinstance(mod, str)]
        
        script = f'''
        tell application "Keyboard Maestro"
            set newTrigger to make new hotkey trigger with properties {{
                key: "{escaped_key}",
                modifiers: {{{", ".join(f'"{mod}"' for mod in escaped_modifiers)}}},
                unique_id: "{trigger_id}"
            }}
            return unique_id of newTrigger
        end tell
        '''
        
        return Either.right(script)
    
    else:
        return Either.left(KMError.validation_error(f"Unsupported trigger type: {trigger_type}"))

async def _execute_applescript_safe(self, script: str) -> Either[KMError, str]:
    """Execute AppleScript with proper error handling and security."""
    
    try:
        # Validate script safety
        if self._contains_dangerous_commands(script):
            return Either.left(KMError.security_error("Dangerous AppleScript commands detected"))
        
        # Execute with timeout
        process = await asyncio.create_subprocess_exec(
            'osascript', '-e', script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.timeout.total_seconds()
            )
        except asyncio.TimeoutError:
            process.terminate()
            return Either.left(KMError.timeout_error("AppleScript execution timeout"))
        
        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown AppleScript error"
            return Either.left(KMError.execution_error(f"AppleScript failed: {error_msg}"))
        
        result = stdout.decode().strip()
        return Either.right(result)
        
    except Exception as e:
        return Either.left(KMError.execution_error(f"AppleScript execution error: {str(e)}"))
```

## 🏗️ Modularity Strategy
- **engine.py fixes**: Enhanced execution reliability and validation (target: +150 lines)
- **km_client.py enhancements**: Improved integration and error handling (target: +100 lines)
- **New execution manager**: Separate concurrent execution handling (target: 125 lines)
- **Resource tracking**: Memory and resource management utilities (target: 75 lines)

## ✅ Success Criteria
- All engine property tests pass with 100% reliability
- Macro execution always returns valid ExecutionResult objects
- Invalid macros properly rejected with clear error messages
- Concurrent execution handles race conditions and state properly
- Large input processing works within memory constraints
- KM client trigger operations work reliably with proper error handling
- Parameter handling correctly escapes and validates all inputs
- Async operations maintain consistency and proper resource cleanup
- Property-based tests validate engine behavior across input ranges
- Integration tests demonstrate reliable KM connectivity and operations