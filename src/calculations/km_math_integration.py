"""
Keyboard Maestro Mathematical Integration

Provides integration with Keyboard Maestro's built-in calculation engine
for processing mathematical expressions with KM token support.
"""

from __future__ import annotations
import subprocess
import asyncio
from typing import Dict, Optional

from ..integration.km_client import Either, KMError


class KMCalculationEngine:
    """Integration with Keyboard Maestro's calculation system."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
    
    async def evaluate_with_km(
        self, 
        expression: str, 
        variables: Optional[Dict[str, str]] = None
    ) -> Either[KMError, str]:
        """Evaluate expression using KM's calculation engine."""
        try:
            # Prepare variables if provided
            variable_setup = ""
            if variables:
                for var_name, var_value in variables.items():
                    # Sanitize variable name and value
                    safe_name = self._sanitize_variable_name(var_name)
                    safe_value = self._escape_for_applescript(str(var_value))
                    variable_setup += f'setvariable "{safe_name}" to "{safe_value}"\n'
            
            # Build AppleScript to use KM's calculate function
            escaped_expression = self._escape_for_applescript(expression)
            
            script = f'''
            tell application "Keyboard Maestro Engine"
                try
                    {variable_setup}
                    set result to calculate "{escaped_expression}"
                    return result as string
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            # Execute AppleScript asynchronously
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown AppleScript error"
                return Either.left(KMError.execution_error(f"KM calculation failed: {error_msg}"))
            
            output = stdout.decode().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error("KM calculation timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"KM calculation error: {str(e)}"))
    
    def _escape_for_applescript(self, text: str) -> str:
        """Escape text for safe AppleScript usage."""
        if not isinstance(text, str):
            text = str(text)
        
        # Replace dangerous characters
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text
    
    def _sanitize_variable_name(self, name: str) -> str:
        """Sanitize variable name for KM usage."""
        # Remove dangerous characters and ensure valid variable name
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
        
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        
        # Limit length and provide fallback
        sanitized = sanitized[:50]
        return sanitized if sanitized else '_var'


class KMVariableResolver:
    """Resolve KM variables for mathematical expressions."""
    
    async def resolve_variables(
        self,
        variable_names: list[str]
    ) -> Either[KMError, Dict[str, str]]:
        """Resolve multiple KM variables."""
        try:
            resolved_vars = {}
            
            for var_name in variable_names:
                var_result = await self._resolve_single_variable(var_name)
                if var_result.is_right():
                    resolved_vars[var_name] = var_result.get_right()
                # Continue with other variables even if one fails
            
            return Either.right(resolved_vars)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Variable resolution failed: {str(e)}"))
    
    async def _resolve_single_variable(self, var_name: str) -> Either[KMError, str]:
        """Resolve a single KM variable."""
        try:
            # Sanitize variable name
            safe_name = self._sanitize_variable_name(var_name)
            
            script = f'''
            tell application "Keyboard Maestro Engine"
                try
                    set varValue to getvariable "{safe_name}"
                    return varValue as string
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            if process.returncode != 0:
                return Either.left(KMError.execution_error(f"Variable resolution failed: {stderr.decode()}"))
            
            output = stdout.decode().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error("Variable resolution timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Variable resolution error: {str(e)}"))
    
    def _sanitize_variable_name(self, name: str) -> str:
        """Sanitize variable name for safe KM access."""
        import re
        # Allow only alphanumeric and underscore characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
        return sanitized[:100] if sanitized else 'unknown_var'


class KMTokenCalculator:
    """Handle calculations involving KM tokens."""
    
    def __init__(self):
        self.km_engine = KMCalculationEngine()
        self.variable_resolver = KMVariableResolver()
    
    async def calculate_with_tokens(
        self,
        expression: str,
        context: str = "calculation"
    ) -> Either[KMError, str]:
        """Calculate expression that may contain KM tokens."""
        try:
            # First, let KM process any tokens in the expression
            token_result = await self._process_tokens(expression, context)
            if token_result.is_left():
                return token_result
            
            processed_expression = token_result.get_right()
            
            # Then evaluate the mathematical expression
            calc_result = await self.km_engine.evaluate_with_km(processed_expression)
            return calc_result
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Token calculation error: {str(e)}"))
    
    async def _process_tokens(self, text: str, context: str) -> Either[KMError, str]:
        """Process KM tokens in text before calculation."""
        try:
            # Escape text for AppleScript
            escaped_text = self.km_engine._escape_for_applescript(text)
            context_param = f" for {context}" if context and context != "calculation" else ""
            
            script = f'''
            tell application "Keyboard Maestro Engine"
                try
                    set result to process tokens "{escaped_text}"{context_param}
                    return result as string
                on error errorMessage
                    return "ERROR: " & errorMessage
                end try
            end tell
            '''
            
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0
            )
            
            if process.returncode != 0:
                return Either.left(KMError.execution_error(f"Token processing failed: {stderr.decode()}"))
            
            output = stdout.decode().strip()
            if output.startswith("ERROR:"):
                return Either.left(KMError.execution_error(output[6:].strip()))
            
            return Either.right(output)
            
        except asyncio.TimeoutError:
            return Either.left(KMError.timeout_error("Token processing timeout"))
        except Exception as e:
            return Either.left(KMError.execution_error(f"Token processing error: {str(e)}"))