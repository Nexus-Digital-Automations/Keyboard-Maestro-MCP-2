"""
Plugin security sandbox and validation system.

This module provides comprehensive security management for plugin execution
including sandboxing, resource monitoring, and threat detection.
"""

import asyncio
import re
import ast
import logging
import resource
import psutil
import tempfile
from typing import Dict, Any, List, Set, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from ..core.plugin_architecture import (
    PluginMetadata, SecurityProfile, PluginError, CustomAction,
    PluginPermissions, PermissionId
)
from ..core.either import Either
from ..core.errors import create_error_context

logger = logging.getLogger(__name__)


class SecurityLimits:
    """Resource limits and security constraints for plugin execution."""
    
    def __init__(self, security_profile: SecurityProfile):
        self.security_profile = security_profile
        self.limits = self._get_profile_limits()
    
    def _get_profile_limits(self) -> Dict[str, Any]:
        """Get resource limits based on security profile."""
        profile_limits = {
            SecurityProfile.NONE: {
                'memory_mb': 1024,
                'cpu_percent': 50,
                'disk_mb': 100,
                'network_requests': 1000,
                'execution_time_seconds': 300,
                'file_operations': 1000,
                'subprocess_allowed': True,
                'network_allowed': True,
                'file_system_write': True
            },
            SecurityProfile.STANDARD: {
                'memory_mb': 512,
                'cpu_percent': 25,
                'disk_mb': 50,
                'network_requests': 100,
                'execution_time_seconds': 60,
                'file_operations': 100,
                'subprocess_allowed': True,
                'network_allowed': True,
                'file_system_write': True
            },
            SecurityProfile.STRICT: {
                'memory_mb': 256,
                'cpu_percent': 10,
                'disk_mb': 25,
                'network_requests': 50,
                'execution_time_seconds': 30,
                'file_operations': 50,
                'subprocess_allowed': False,
                'network_allowed': False,
                'file_system_write': False
            },
            SecurityProfile.SANDBOX: {
                'memory_mb': 128,
                'cpu_percent': 5,
                'disk_mb': 10,
                'network_requests': 10,
                'execution_time_seconds': 15,
                'file_operations': 10,
                'subprocess_allowed': False,
                'network_allowed': False,
                'file_system_write': False
            }
        }
        
        return profile_limits.get(self.security_profile, profile_limits[SecurityProfile.STANDARD])
    
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes."""
        return self.limits['memory_mb'] * 1024 * 1024
    
    def get_execution_timeout(self) -> float:
        """Get execution timeout in seconds."""
        return float(self.limits['execution_time_seconds'])
    
    def allows_network(self) -> bool:
        """Check if network access is allowed."""
        return self.limits['network_allowed']
    
    def allows_subprocess(self) -> bool:
        """Check if subprocess execution is allowed."""
        return self.limits['subprocess_allowed']
    
    def allows_file_write(self) -> bool:
        """Check if file system write is allowed."""
        return self.limits['file_system_write']


class SecurityScanner:
    """Static analysis and security scanning for plugin code."""
    
    # Dangerous function patterns
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open', 'file',
        'input', 'raw_input', 'execfile', 'reload', 'vars', 'locals', 'globals',
        'getattr', 'setattr', 'delattr', 'hasattr'
    }
    
    # Dangerous module patterns
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'tempfile', 'pickle',
        'marshal', 'shelve', 'dill', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'telnetlib', 'webbrowser'
    }
    
    # Suspicious patterns in code
    SUSPICIOUS_PATTERNS = [
        r'__[a-zA-Z_]+__',              # Dunder methods
        r'\.system\s*\(',               # System calls
        r'\.popen\s*\(',                # Process creation
        r'\.spawn\s*\(',                # Process spawning
        r'subprocess\.',                # Subprocess module
        r'os\.',                        # OS module usage
        r'import\s+os',                 # OS import
        r'from\s+os\s+import',          # OS from import
        r'exec\s*\(',                   # Exec function
        r'eval\s*\(',                   # Eval function
        r'__import__\s*\(',             # Dynamic import
        r'socket\.',                    # Socket usage
        r'urllib\.',                    # URL library
        r'requests\.',                  # Requests library
        r'http\.',                      # HTTP library
        r'\.read\(\)',                  # File reading
        r'\.write\(',                   # File writing
        r'\.delete\(',                  # File deletion
        r'\.remove\(',                  # File removal
        r'\.rmdir\(',                   # Directory removal
        r'\.mkdir\(',                   # Directory creation
    ]
    
    def scan_plugin_code(self, plugin_path: Path) -> Either[PluginError, Dict[str, Any]]:
        """Perform comprehensive security scan of plugin code."""
        try:
            scan_results = {
                'dangerous_functions': [],
                'dangerous_modules': [],
                'suspicious_patterns': [],
                'import_analysis': [],
                'file_operations': [],
                'network_operations': [],
                'security_rating': 'SAFE',  # SAFE, WARNING, DANGEROUS
                'recommendations': []
            }
            
            # Scan all Python files
            python_files = list(plugin_path.rglob("*.py"))
            
            for py_file in python_files:
                file_results = self._scan_file(py_file)
                if file_results.is_left():
                    continue  # Skip files that can't be scanned
                
                file_scan = file_results.get_right()
                
                # Merge results
                for key in ['dangerous_functions', 'dangerous_modules', 'suspicious_patterns',
                           'import_analysis', 'file_operations', 'network_operations']:
                    scan_results[key].extend(file_scan.get(key, []))
            
            # Determine security rating
            scan_results['security_rating'] = self._calculate_security_rating(scan_results)
            scan_results['recommendations'] = self._generate_recommendations(scan_results)
            
            return Either.right(scan_results)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Security scan failed: {str(e)}"))
    
    def _scan_file(self, file_path: Path) -> Either[PluginError, Dict[str, Any]]:
        """Scan individual Python file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_results = {
                'file': str(file_path),
                'dangerous_functions': [],
                'dangerous_modules': [],
                'suspicious_patterns': [],
                'import_analysis': [],
                'file_operations': [],
                'network_operations': []
            }
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                ast_results = self._analyze_ast(tree)
                file_results.update(ast_results)
            except SyntaxError as e:
                file_results['syntax_error'] = str(e)
            
            # Pattern-based scanning
            pattern_results = self._scan_patterns(content)
            for key, values in pattern_results.items():
                file_results[key].extend(values)
            
            return Either.right(file_results)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"File scan failed: {str(e)}"))
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze Abstract Syntax Tree for security issues."""
        results = {
            'dangerous_functions': [],
            'dangerous_modules': [],
            'import_analysis': [],
            'file_operations': [],
            'network_operations': []
        }
        
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.DANGEROUS_FUNCTIONS:
                        results['dangerous_functions'].append(func_name)
                
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    if attr_name in ['system', 'popen', 'spawn', 'read', 'write', 'delete']:
                        if attr_name in ['read', 'write', 'delete']:
                            results['file_operations'].append(attr_name)
                        else:
                            results['dangerous_functions'].append(attr_name)
            
            # Check imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    results['import_analysis'].append(f"import {module_name}")
                    if module_name in self.DANGEROUS_MODULES:
                        results['dangerous_modules'].append(module_name)
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                results['import_analysis'].append(f"from {module_name} import ...")
                if module_name in self.DANGEROUS_MODULES:
                    results['dangerous_modules'].append(module_name)
        
        return results
    
    def _scan_patterns(self, content: str) -> Dict[str, List[str]]:
        """Scan content for suspicious patterns using regex."""
        results = {
            'suspicious_patterns': [],
            'file_operations': [],
            'network_operations': []
        }
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                results['suspicious_patterns'].extend(matches)
                
                # Categorize specific types
                if any(keyword in pattern for keyword in ['read', 'write', 'delete', 'remove', 'mkdir']):
                    results['file_operations'].extend(matches)
                elif any(keyword in pattern for keyword in ['socket', 'urllib', 'requests', 'http']):
                    results['network_operations'].extend(matches)
        
        return results
    
    def _calculate_security_rating(self, scan_results: Dict[str, Any]) -> str:
        """Calculate overall security rating based on scan results."""
        dangerous_count = len(scan_results['dangerous_functions']) + len(scan_results['dangerous_modules'])
        suspicious_count = len(scan_results['suspicious_patterns'])
        
        if dangerous_count > 5 or suspicious_count > 10:
            return 'DANGEROUS'
        elif dangerous_count > 2 or suspicious_count > 5:
            return 'WARNING'
        else:
            return 'SAFE'
    
    def _generate_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if scan_results['dangerous_functions']:
            recommendations.append("Consider removing dangerous function calls or using safer alternatives")
        
        if scan_results['dangerous_modules']:
            recommendations.append("Dangerous modules detected - ensure they are necessary and used safely")
        
        if scan_results['file_operations']:
            recommendations.append("File operations detected - ensure proper path validation and permissions")
        
        if scan_results['network_operations']:
            recommendations.append("Network operations detected - ensure proper input validation and HTTPS usage")
        
        if scan_results['security_rating'] == 'DANGEROUS':
            recommendations.append("Plugin contains high-risk code - manual review strongly recommended")
        
        return recommendations


class ResourceMonitor:
    """Monitor plugin resource usage during execution."""
    
    def __init__(self, limits: SecurityLimits):
        self.limits = limits
        self.start_time: Optional[datetime] = None
        self.initial_memory: Optional[int] = None
        self.peak_memory: int = 0
        self.file_operations: int = 0
        self.network_requests: int = 0
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = datetime.now()
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.peak_memory = self.initial_memory
    
    def check_limits(self) -> Either[PluginError, None]:
        """Check if resource limits are exceeded."""
        try:
            if not self.start_time:
                return Either.right(None)
            
            # Check execution time
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > self.limits.get_execution_timeout():
                return Either.left(PluginError(
                    f"Execution timeout exceeded: {elapsed}s > {self.limits.get_execution_timeout()}s",
                    "EXECUTION_TIMEOUT"
                ))
            
            # Check memory usage
            process = psutil.Process()
            current_memory = process.memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
            
            memory_used = current_memory - (self.initial_memory or 0)
            if memory_used > self.limits.get_memory_limit_bytes():
                return Either.left(PluginError(
                    f"Memory limit exceeded: {memory_used / 1024 / 1024:.1f}MB > {self.limits.limits['memory_mb']}MB",
                    "MEMORY_LIMIT_EXCEEDED"
                ))
            
            # Check file operations
            if self.file_operations > self.limits.limits['file_operations']:
                return Either.left(PluginError(
                    f"File operation limit exceeded: {self.file_operations} > {self.limits.limits['file_operations']}",
                    "FILE_OPERATIONS_EXCEEDED"
                ))
            
            # Check network requests
            if self.network_requests > self.limits.limits['network_requests']:
                return Either.left(PluginError(
                    f"Network request limit exceeded: {self.network_requests} > {self.limits.limits['network_requests']}",
                    "NETWORK_REQUESTS_EXCEEDED"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Resource monitoring failed: {str(e)}"))
    
    def record_file_operation(self):
        """Record a file operation."""
        self.file_operations += 1
    
    def record_network_request(self):
        """Record a network request."""
        self.network_requests += 1
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
        
        memory_used = 0
        if self.initial_memory:
            memory_used = (self.peak_memory - self.initial_memory) / 1024 / 1024  # MB
        
        return {
            'execution_time_seconds': elapsed,
            'peak_memory_mb': memory_used,
            'file_operations': self.file_operations,
            'network_requests': self.network_requests,
            'limits': {
                'max_execution_time': self.limits.get_execution_timeout(),
                'max_memory_mb': self.limits.limits['memory_mb'],
                'max_file_operations': self.limits.limits['file_operations'],
                'max_network_requests': self.limits.limits['network_requests']
            }
        }


class PluginSandbox:
    """Secure plugin execution sandbox with resource monitoring and isolation."""
    
    def __init__(self, security_profile: SecurityProfile):
        self.security_profile = security_profile
        self.limits = SecurityLimits(security_profile)
        self.monitor = ResourceMonitor(self.limits)
    
    @asynccontextmanager
    async def execution_context(self):
        """Context manager for sandboxed execution."""
        self.monitor.start_monitoring()
        
        try:
            # Set resource limits at OS level (Unix/Linux)
            if hasattr(resource, 'RLIMIT_AS'):  # Memory limit
                resource.setrlimit(resource.RLIMIT_AS, (
                    self.limits.get_memory_limit_bytes(),
                    self.limits.get_memory_limit_bytes()
                ))
            
            if hasattr(resource, 'RLIMIT_CPU'):  # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (
                    int(self.limits.get_execution_timeout()),
                    int(self.limits.get_execution_timeout()) + 5
                ))
            
            yield self.monitor
            
        finally:
            # Reset resource limits
            if hasattr(resource, 'RLIMIT_AS'):
                resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            if hasattr(resource, 'RLIMIT_CPU'):
                resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    
    async def execute_action(self, action: CustomAction, parameters: Dict[str, Any]) -> Either[PluginError, Any]:
        """Execute custom action within security sandbox."""
        try:
            async with self.execution_context() as monitor:
                # Create monitoring task
                monitor_task = asyncio.create_task(self._monitor_execution(monitor))
                
                try:
                    # Execute action with timeout
                    execution_task = asyncio.create_task(action.execute(parameters))
                    
                    # Wait for either execution completion or monitoring failure
                    done, pending = await asyncio.wait(
                        [execution_task, monitor_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=self.limits.get_execution_timeout()
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                    
                    # Check if execution completed successfully
                    if execution_task in done:
                        result = await execution_task
                        
                        # Final resource check
                        limit_check = monitor.check_limits()
                        if limit_check.is_left():
                            return limit_check
                        
                        return result
                    
                    # Check if monitor detected violation
                    elif monitor_task in done:
                        monitor_result = await monitor_task
                        return monitor_result
                    
                    else:
                        # Timeout
                        return Either.left(PluginError(
                            f"Action execution timeout: {self.limits.get_execution_timeout()}s",
                            "EXECUTION_TIMEOUT"
                        ))
                
                except asyncio.CancelledError:
                    return Either.left(PluginError("Action execution cancelled", "EXECUTION_CANCELLED"))
            
        except Exception as e:
            context = create_error_context("execute_action", "plugin_sandbox", 
                                         action_id=action.action_id, error=str(e))
            return Either.left(PluginError.execution_error(str(e), context))
    
    async def _monitor_execution(self, monitor: ResourceMonitor) -> Either[PluginError, None]:
        """Continuously monitor execution for resource violations."""
        try:
            while True:
                limit_check = monitor.check_limits()
                if limit_check.is_left():
                    return limit_check
                
                # Check every 100ms
                await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            return Either.right(None)


class PluginSecurityManager:
    """Comprehensive plugin security management."""
    
    def __init__(self):
        self.scanner = SecurityScanner()
        self.approved_plugins: Set[str] = set()
        self.blocked_plugins: Set[str] = set()
    
    async def validate_plugin(self, plugin_path: Path, metadata: PluginMetadata) -> Either[PluginError, None]:
        """Comprehensive plugin security validation."""
        try:
            # Check if plugin is blocked
            if metadata.identifier in self.blocked_plugins:
                return Either.left(PluginError.security_violation(f"Plugin is blocked: {metadata.identifier}"))
            
            # Skip validation for pre-approved plugins
            if metadata.identifier in self.approved_plugins:
                return Either.right(None)
            
            # Validate plugin metadata
            metadata_result = self._validate_metadata_security(metadata)
            if metadata_result.is_left():
                return metadata_result
            
            # Scan plugin code
            scan_result = self.scanner.scan_plugin_code(plugin_path)
            if scan_result.is_left():
                return scan_result
            
            scan_data = scan_result.get_right()
            
            # Check security rating
            if scan_data['security_rating'] == 'DANGEROUS':
                return Either.left(PluginError.security_violation(
                    f"Plugin contains dangerous code: {', '.join(scan_data['recommendations'])}"
                ))
            
            # Validate permissions against capabilities
            permissions_result = self._validate_permissions(metadata.permissions, scan_data)
            if permissions_result.is_left():
                return permissions_result
            
            logger.info(f"Plugin security validation passed: {metadata.identifier} (rating: {scan_data['security_rating']})")
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Security validation failed: {str(e)}"))
    
    def _validate_metadata_security(self, metadata: PluginMetadata) -> Either[PluginError, None]:
        """Validate plugin metadata for security constraints."""
        try:
            # Check for suspicious plugin names
            suspicious_names = ['system', 'admin', 'root', 'hack', 'exploit', 'backdoor']
            if any(sus in metadata.name.lower() for sus in suspicious_names):
                return Either.left(PluginError.security_violation(f"Suspicious plugin name: {metadata.name}"))
            
            # Validate permissions don't exceed security profile
            if metadata.permissions.requires_elevated_access():
                # Additional validation for elevated permissions
                if not metadata.permissions.network_access and not metadata.permissions.file_system_access:
                    # System integration without network/file access is suspicious
                    if metadata.permissions.system_integration:
                        return Either.left(PluginError.security_violation(
                            "System integration without network/file access is not allowed"
                        ))
            
            # Check dependency security
            for dependency in metadata.dependencies:
                if dependency.plugin_id in self.blocked_plugins:
                    return Either.left(PluginError.security_violation(
                        f"Plugin depends on blocked plugin: {dependency.plugin_id}"
                    ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Metadata validation failed: {str(e)}"))
    
    def _validate_permissions(self, permissions: PluginPermissions, scan_data: Dict[str, Any]) -> Either[PluginError, None]:
        """Validate that requested permissions match code capabilities."""
        try:
            # Check if code contains network operations but lacks network permission
            if scan_data['network_operations'] and not permissions.network_access:
                return Either.left(PluginError.security_violation(
                    "Plugin contains network operations but lacks network permission"
                ))
            
            # Check if code contains file operations but lacks file system permission
            if scan_data['file_operations'] and not permissions.file_system_access:
                return Either.left(PluginError.security_violation(
                    "Plugin contains file operations but lacks file system permission"
                ))
            
            # Check for over-permissive requests
            if permissions.network_access and not scan_data['network_operations']:
                logger.warning(f"Plugin requests network access but doesn't use it")
            
            if permissions.file_system_access and not scan_data['file_operations']:
                logger.warning(f"Plugin requests file system access but doesn't use it")
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Permission validation failed: {str(e)}"))
    
    def approve_plugin(self, plugin_id: str):
        """Add plugin to approved list (bypasses security scanning)."""
        self.approved_plugins.add(plugin_id)
        self.blocked_plugins.discard(plugin_id)
    
    def block_plugin(self, plugin_id: str):
        """Add plugin to blocked list."""
        self.blocked_plugins.add(plugin_id)
        self.approved_plugins.discard(plugin_id)
    
    def get_security_report(self, plugin_path: Path) -> Either[PluginError, Dict[str, Any]]:
        """Generate comprehensive security report for plugin."""
        try:
            scan_result = self.scanner.scan_plugin_code(plugin_path)
            if scan_result.is_left():
                return scan_result
            
            scan_data = scan_result.get_right()
            
            # Add additional context
            report = {
                **scan_data,
                'scan_timestamp': datetime.now().isoformat(),
                'scanner_version': '1.0',
                'plugin_path': str(plugin_path)
            }
            
            return Either.right(report)
            
        except Exception as e:
            return Either.left(PluginError.security_violation(f"Security report generation failed: {str(e)}"))