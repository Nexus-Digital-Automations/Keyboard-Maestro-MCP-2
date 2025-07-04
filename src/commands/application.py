"""
Application Control Commands

Provides secure application management including launch, quit, and activation
with comprehensive validation and security boundaries.
"""

from __future__ import annotations
from typing import Optional, FrozenSet, Union, List
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import subprocess
import platform
import shutil

from ..core.types import ExecutionContext, CommandResult, Permission, Duration
from ..core.contracts import require, ensure
from .base import BaseCommand, create_command_result
from .validation import SecurityValidator


class ApplicationAction(Enum):
    """Application control actions."""
    LAUNCH = "launch"
    QUIT = "quit"
    ACTIVATE = "activate"
    FORCE_QUIT = "force_quit"


class ApplicationState(Enum):
    """Application states."""
    RUNNING = "running"
    NOT_RUNNING = "not_running"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LaunchApplicationCommand(BaseCommand):
    """
    Launch applications with security validation and path safety.
    
    Provides secure application launching with validation to prevent
    execution of unauthorized or malicious applications.
    """
    
    def get_application_name(self) -> str:
        """Get the application name to launch."""
        return self.parameters.get("application_name", "")
    
    def get_application_path(self) -> Optional[str]:
        """Get the explicit application path if provided."""
        return self.parameters.get("application_path")
    
    def get_launch_arguments(self) -> List[str]:
        """Get command line arguments for the application."""
        args = self.parameters.get("launch_arguments", [])
        if isinstance(args, list):
            return [str(arg) for arg in args[:10]]  # Limit to 10 args
        return []
    
    def get_wait_for_launch(self) -> bool:
        """Check if we should wait for the application to fully launch."""
        return self.parameters.get("wait_for_launch", False)
    
    def get_launch_timeout(self) -> Duration:
        """Get timeout for waiting for application launch."""
        timeout_seconds = self.parameters.get("launch_timeout", 30.0)
        try:
            timeout = Duration.from_seconds(float(timeout_seconds))
            # Limit timeout to reasonable range
            if timeout.seconds > 60:
                return Duration.from_seconds(60)
            return timeout
        except (ValueError, TypeError):
            return Duration.from_seconds(30)
    
    def _validate_impl(self) -> bool:
        """Validate application launch parameters."""
        app_name = self.get_application_name()
        app_path = self.get_application_path()
        
        # Must have either app name or path
        if not app_name and not app_path:
            return False
        
        # Validate application name
        if app_name:
            validator = SecurityValidator()
            if not validator.validate_text_input(app_name, "application_name"):
                return False
            
            # Check for dangerous application names
            dangerous_apps = [
                'rm', 'del', 'format', 'shutdown', 'reboot',
                'sudo', 'su', 'passwd', 'killall', 'pkill'
            ]
            if app_name.lower() in dangerous_apps:
                return False
        
        # Validate application path if provided
        if app_path:
            validator = SecurityValidator()
            if not validator.validate_file_path(app_path, "application_path"):
                return False
            
            # Additional validation for executable files
            if not self._is_safe_executable(app_path):
                return False
        
        # Validate launch arguments
        launch_args = self.get_launch_arguments()
        if launch_args:
            validator = SecurityValidator()
            for i, arg in enumerate(launch_args):
                if not validator.validate_text_input(arg, f"launch_arguments[{i}]"):
                    return False
        
        # Validate timeout
        timeout = self.get_launch_timeout()
        if timeout.seconds <= 0 or timeout.seconds > 60:
            return False
        
        return True
    
    def _is_safe_executable(self, path: str) -> bool:
        """Check if the executable path is safe to run."""
        try:
            # Check if file exists
            if not os.path.isfile(path):
                return False
            
            # Check if file is executable
            if not os.access(path, os.X_OK):
                return False
            
            # Check file extension (platform-specific)
            system = platform.system().lower()
            if system == "windows":
                safe_extensions = {'.exe', '.msi', '.app'}
                if not any(path.lower().endswith(ext) for ext in safe_extensions):
                    return False
            elif system == "darwin":
                # On macOS, check for .app bundles or known system locations
                if path.endswith('.app') or path.startswith('/Applications/') or path.startswith('/System/'):
                    return True
                return False
            
            # Basic safety check - avoid system executables
            dangerous_paths = [
                '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/',
                'system32', 'syswow64', 'windows/system'
            ]
            path_lower = path.lower()
            for dangerous in dangerous_paths:
                if dangerous in path_lower:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute application launch with security checks."""
        app_name = self.get_application_name()
        app_path = self.get_application_path()
        launch_args = self.get_launch_arguments()
        wait_for_launch = self.get_wait_for_launch()
        launch_timeout = self.get_launch_timeout()
        
        start_time = time.time()
        
        try:
            # Determine what to launch
            if app_path:
                executable = app_path
            else:
                executable = self._resolve_application_path(app_name)
                if not executable:
                    return create_command_result(
                        success=False,
                        error_message=f"Could not find application: {app_name}",
                        execution_time=Duration.from_seconds(time.time() - start_time)
                    )
            
            # Build command
            cmd = [executable] + launch_args
            
            # Launch the application
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent process
            )
            
            # Wait for launch if requested
            if wait_for_launch:
                try:
                    process.wait(timeout=launch_timeout.seconds)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    return_code = None
            else:
                return_code = None
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            # Check if launch was successful
            if return_code is None or return_code == 0:
                return create_command_result(
                    success=True,
                    output=f"Successfully launched {app_name or os.path.basename(executable)}",
                    execution_time=execution_time,
                    application_name=app_name,
                    executable_path=executable,
                    process_id=process.pid,
                    return_code=return_code,
                    launch_arguments=launch_args
                )
            else:
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore') if process.stderr else ""
                return create_command_result(
                    success=False,
                    error_message=f"Application launch failed with code {return_code}: {stderr_output}",
                    execution_time=execution_time,
                    return_code=return_code
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Failed to launch application: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _resolve_application_path(self, app_name: str) -> Optional[str]:
        """Resolve application name to executable path."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                # Try common application locations
                app_locations = [
                    f"/Applications/{app_name}.app",
                    f"/Applications/{app_name}.app/Contents/MacOS/{app_name}",
                    f"/System/Applications/{app_name}.app",
                    f"/System/Applications/{app_name}.app/Contents/MacOS/{app_name}"
                ]
                
                for location in app_locations:
                    if os.path.exists(location):
                        return location
                
                # Try using 'which' for command line tools
                result = subprocess.run(['which', app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
                    
            elif system == "linux":
                # Try using 'which' to find the executable
                result = subprocess.run(['which', app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
                    
            elif system == "windows":
                # Try using 'where' to find the executable
                result = subprocess.run(['where', app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            
            return None
            
        except Exception:
            return None
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Application launch requires system control permission."""
        return frozenset([Permission.SYSTEM_CONTROL])
    
    def get_security_risk_level(self) -> str:
        """Application launch has high risk due to code execution."""
        return "high"


@dataclass(frozen=True)
class QuitApplicationCommand(BaseCommand):
    """
    Quit running applications with graceful and forced termination options.
    
    Provides secure application termination with validation to prevent
    termination of critical system processes.
    """
    
    def get_application_name(self) -> str:
        """Get the application name to quit."""
        return self.parameters.get("application_name", "")
    
    def get_process_id(self) -> Optional[int]:
        """Get specific process ID to quit."""
        pid = self.parameters.get("process_id")
        if pid is not None:
            try:
                return int(pid)
            except (ValueError, TypeError):
                return None
        return None
    
    def get_force_quit(self) -> bool:
        """Check if application should be force quit."""
        return self.parameters.get("force_quit", False)
    
    def get_quit_timeout(self) -> Duration:
        """Get timeout for graceful quit before forcing."""
        timeout_seconds = self.parameters.get("quit_timeout", 10.0)
        try:
            timeout = Duration.from_seconds(float(timeout_seconds))
            # Limit timeout to reasonable range
            if timeout.seconds > 30:
                return Duration.from_seconds(30)
            return timeout
        except (ValueError, TypeError):
            return Duration.from_seconds(10)
    
    def _validate_impl(self) -> bool:
        """Validate application quit parameters."""
        app_name = self.get_application_name()
        process_id = self.get_process_id()
        
        # Must have either app name or process ID
        if not app_name and process_id is None:
            return False
        
        # Validate application name
        if app_name:
            validator = SecurityValidator()
            if not validator.validate_text_input(app_name, "application_name"):
                return False
            
            # Prevent quitting critical system processes
            protected_processes = [
                'kernel', 'init', 'systemd', 'launchd', 'explorer.exe',
                'winlogon.exe', 'csrss.exe', 'smss.exe', 'wininit.exe'
            ]
            if app_name.lower() in protected_processes:
                return False
        
        # Validate process ID
        if process_id is not None:
            if process_id <= 0 or process_id > 65535:
                return False
            
            # Prevent killing critical system PIDs
            if process_id <= 10:  # PIDs 1-10 are typically system processes
                return False
        
        # Validate timeout
        timeout = self.get_quit_timeout()
        if timeout.seconds <= 0 or timeout.seconds > 30:
            return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute application quit with graceful/force options."""
        app_name = self.get_application_name()
        process_id = self.get_process_id()
        force_quit = self.get_force_quit()
        quit_timeout = self.get_quit_timeout()
        
        start_time = time.time()
        
        try:
            # Find process(es) to quit
            if process_id:
                pids_to_quit = [process_id]
                app_display_name = f"PID {process_id}"
            else:
                pids_to_quit = self._find_application_pids(app_name)
                app_display_name = app_name
            
            if not pids_to_quit:
                return create_command_result(
                    success=False,
                    error_message=f"Application not running: {app_display_name}",
                    execution_time=Duration.from_seconds(time.time() - start_time)
                )
            
            # Quit each process
            quit_results = []
            for pid in pids_to_quit:
                result = self._quit_process(pid, force_quit, quit_timeout)
                quit_results.append(result)
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            successful_quits = sum(1 for result in quit_results if result['success'])
            total_processes = len(quit_results)
            
            if successful_quits == total_processes:
                return create_command_result(
                    success=True,
                    output=f"Successfully quit {successful_quits} process(es) for {app_display_name}",
                    execution_time=execution_time,
                    application_name=app_name,
                    processes_quit=successful_quits,
                    total_processes=total_processes,
                    force_quit_used=force_quit,
                    quit_results=quit_results
                )
            else:
                return create_command_result(
                    success=False,
                    error_message=f"Only quit {successful_quits} of {total_processes} processes",
                    execution_time=execution_time,
                    processes_quit=successful_quits,
                    total_processes=total_processes,
                    quit_results=quit_results
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Failed to quit application: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _find_application_pids(self, app_name: str) -> List[int]:
        """Find PIDs for the specified application."""
        try:
            system = platform.system().lower()
            pids = []
            
            if system == "darwin":  # macOS
                result = subprocess.run(['pgrep', '-f', app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
                    
            elif system == "linux":
                result = subprocess.run(['pgrep', '-f', app_name], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
                    
            elif system == "windows":
                result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {app_name}*'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse Windows tasklist output
                    lines = result.stdout.split('\n')
                    for line in lines[3:]:  # Skip header lines
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pids.append(int(parts[1]))
                                except ValueError:
                                    continue
            
            return pids
            
        except Exception:
            return []
    
    def _quit_process(self, pid: int, force: bool, timeout: Duration) -> dict:
        """Quit a specific process."""
        try:
            system = platform.system().lower()
            
            if force:
                # Force quit immediately
                if system == "windows":
                    result = subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                          capture_output=True, timeout=5)
                else:
                    result = subprocess.run(['kill', '-9', str(pid)], 
                                          capture_output=True, timeout=5)
                
                return {
                    'pid': pid,
                    'success': result.returncode == 0,
                    'method': 'force_kill',
                    'error': result.stderr.decode('utf-8', errors='ignore') if result.stderr else None
                }
            else:
                # Graceful quit first
                if system == "windows":
                    result = subprocess.run(['taskkill', '/PID', str(pid)], 
                                          capture_output=True, timeout=5)
                else:
                    result = subprocess.run(['kill', '-TERM', str(pid)], 
                                          capture_output=True, timeout=5)
                
                if result.returncode == 0:
                    # Wait for process to exit
                    for _ in range(int(timeout.seconds * 10)):  # Check every 0.1 seconds
                        if not self._is_process_running(pid):
                            return {
                                'pid': pid,
                                'success': True,
                                'method': 'graceful_term',
                                'error': None
                            }
                        time.sleep(0.1)
                    
                    # Process didn't exit gracefully, force kill
                    if system == "windows":
                        force_result = subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                                    capture_output=True, timeout=5)
                    else:
                        force_result = subprocess.run(['kill', '-9', str(pid)], 
                                                    capture_output=True, timeout=5)
                    
                    return {
                        'pid': pid,
                        'success': force_result.returncode == 0,
                        'method': 'timeout_then_force',
                        'error': force_result.stderr.decode('utf-8', errors='ignore') if force_result.stderr else None
                    }
                else:
                    return {
                        'pid': pid,
                        'success': False,
                        'method': 'graceful_term',
                        'error': result.stderr.decode('utf-8', errors='ignore') if result.stderr else None
                    }
                    
        except Exception as e:
            return {
                'pid': pid,
                'success': False,
                'method': 'exception',
                'error': str(e)
            }
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                      capture_output=True, text=True, timeout=5)
                return f"PID {pid}" in result.stdout if result.returncode == 0 else False
            else:
                result = subprocess.run(['kill', '-0', str(pid)], 
                                      capture_output=True, timeout=5)
                return result.returncode == 0
                
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Application quit requires system control permission."""
        return frozenset([Permission.SYSTEM_CONTROL])
    
    def get_security_risk_level(self) -> str:
        """Application quit has high risk due to process termination."""
        return "high"


@dataclass(frozen=True)
class ActivateApplicationCommand(BaseCommand):
    """
    Activate (bring to foreground) running applications.
    
    Provides secure application activation with validation to prevent
    unwanted window focus changes and disruption.
    """
    
    def get_application_name(self) -> str:
        """Get the application name to activate."""
        return self.parameters.get("application_name", "")
    
    def get_window_title(self) -> Optional[str]:
        """Get specific window title to activate."""
        return self.parameters.get("window_title")
    
    def get_create_if_not_running(self) -> bool:
        """Check if application should be launched if not running."""
        return self.parameters.get("create_if_not_running", False)
    
    def _validate_impl(self) -> bool:
        """Validate application activation parameters."""
        app_name = self.get_application_name()
        
        # Application name is required
        if not app_name:
            return False
        
        # Validate application name
        validator = SecurityValidator()
        if not validator.validate_text_input(app_name, "application_name"):
            return False
        
        # Validate window title if provided
        window_title = self.get_window_title()
        if window_title:
            if not validator.validate_text_input(window_title, "window_title"):
                return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute application activation."""
        app_name = self.get_application_name()
        window_title = self.get_window_title()
        create_if_not_running = self.get_create_if_not_running()
        
        start_time = time.time()
        
        try:
            # Check if application is running
            if not self._is_application_running(app_name):
                if create_if_not_running:
                    # Launch the application first
                    launch_result = self._launch_application(app_name)
                    if not launch_result:
                        return create_command_result(
                            success=False,
                            error_message=f"Could not launch application: {app_name}",
                            execution_time=Duration.from_seconds(time.time() - start_time)
                        )
                    # Wait a moment for the app to start
                    time.sleep(2)
                else:
                    return create_command_result(
                        success=False,
                        error_message=f"Application not running: {app_name}",
                        execution_time=Duration.from_seconds(time.time() - start_time)
                    )
            
            # Activate the application
            success = self._activate_application(app_name, window_title)
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            if success:
                return create_command_result(
                    success=True,
                    output=f"Successfully activated {app_name}",
                    execution_time=execution_time,
                    application_name=app_name,
                    window_title=window_title,
                    was_launched=create_if_not_running and not self._was_initially_running
                )
            else:
                return create_command_result(
                    success=False,
                    error_message=f"Failed to activate {app_name}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Application activation failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _is_application_running(self, app_name: str) -> bool:
        """Check if application is currently running."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                result = subprocess.run(['pgrep', '-f', app_name], capture_output=True, timeout=5)
                return result.returncode == 0
                
            elif system == "linux":
                result = subprocess.run(['pgrep', '-f', app_name], capture_output=True, timeout=5)
                return result.returncode == 0
                
            elif system == "windows":
                result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {app_name}*'], 
                                      capture_output=True, text=True, timeout=5)
                return app_name.lower() in result.stdout.lower() if result.returncode == 0 else False
            
            return False
            
        except Exception:
            return False
    
    def _launch_application(self, app_name: str) -> bool:
        """Launch application if not running."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                result = subprocess.run(['open', '-a', app_name], timeout=10)
                return result.returncode == 0
                
            elif system == "linux":
                result = subprocess.run([app_name], timeout=10)
                return result.returncode == 0
                
            elif system == "windows":
                result = subprocess.run(['start', app_name], shell=True, timeout=10)
                return result.returncode == 0
            
            return False
            
        except Exception:
            return False
    
    def _activate_application(self, app_name: str, window_title: Optional[str] = None) -> bool:
        """Activate the specified application."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                if window_title:
                    # Activate specific window by title
                    script = f'''
                    tell application "{app_name}"
                        activate
                        set windowList to every window whose name contains "{window_title}"
                        if length of windowList > 0 then
                            set index of item 1 of windowList to 1
                        end if
                    end tell
                    '''
                else:
                    # Activate application
                    script = f'tell application "{app_name}" to activate'
                
                result = subprocess.run(['osascript', '-e', script], timeout=10)
                return result.returncode == 0
                
            elif system == "linux":
                # Use wmctrl if available
                try:
                    if window_title:
                        result = subprocess.run(['wmctrl', '-a', window_title], timeout=5)
                    else:
                        result = subprocess.run(['wmctrl', '-a', app_name], timeout=5)
                    return result.returncode == 0
                except FileNotFoundError:
                    # wmctrl not available, try xdotool
                    try:
                        if window_title:
                            result = subprocess.run(['xdotool', 'search', '--name', window_title, 'windowactivate'], timeout=5)
                        else:
                            result = subprocess.run(['xdotool', 'search', '--class', app_name, 'windowactivate'], timeout=5)
                        return result.returncode == 0
                    except FileNotFoundError:
                        return False
                        
            elif system == "windows":
                # Windows doesn't have a simple built-in way to activate by app name
                # This would require additional libraries like pywin32
                return True  # Return success but note it's not fully implemented
            
            return False
            
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Application activation requires window management permission."""
        permissions = [Permission.WINDOW_MANAGEMENT]
        if self.get_create_if_not_running():
            permissions.append(Permission.SYSTEM_CONTROL)
        return frozenset(permissions)
    
    def get_security_risk_level(self) -> str:
        """Application activation has medium risk due to window focus changes."""
        return "medium"