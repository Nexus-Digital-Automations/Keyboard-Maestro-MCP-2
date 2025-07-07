#!/usr/bin/env python3
"""Enhanced Function Argument Type Annotation Fix Script.

This script specifically targets ANN001 violations (missing-type-function-argument)
using advanced pattern matching and intelligent type inference.
"""

import ast
import logging
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP")

# Common argument type patterns based on name and usage
ARGUMENT_TYPE_MAPPING = {
    # String patterns
    "name": "str",
    "text": "str", 
    "message": "str",
    "content": "str",
    "data": "str | bytes | Any",
    "value": "Any",
    "key": "str",
    "path": "str | Path",
    "url": "str",
    "uri": "str",
    "email": "str",
    "username": "str",
    "password": "str",
    "token": "str",
    "id": "str",
    "identifier": "str",
    "uuid": "str",
    "filename": "str",
    "filepath": "str | Path",
    "dirname": "str | Path",
    "command": "str",
    "action": "str",
    "operation": "str",
    "method": "str",
    "description": "str",
    "title": "str",
    "label": "str",
    "tag": "str",
    "category": "str",
    "type": "str",
    "kind": "str",
    "status": "str",
    "state": "str",
    "mode": "str",
    "format": "str",
    "encoding": "str",
    "language": "str",
    "locale": "str",
    "pattern": "str",
    "query": "str",
    "expression": "str",
    "script": "str",
    "code": "str",
    "version": "str",
    "branch": "str",
    "commit": "str",
    "hash": "str",
    "signature": "str",
    "error": "str | Exception",
    "exception": "Exception",
    "reason": "str",
    "cause": "str",
    "source": "str",
    "target": "str",
    "destination": "str",
    "output": "str | Any",
    "result": "Any",
    "response": "Any",
    "request": "Any",
    
    # Numeric patterns
    "count": "int",
    "size": "int",
    "length": "int",
    "width": "int",
    "height": "int",
    "depth": "int",
    "radius": "int | float",
    "diameter": "int | float",
    "area": "int | float",
    "volume": "int | float",
    "weight": "int | float",
    "mass": "int | float",
    "temperature": "int | float",
    "time": "int | float",
    "duration": "int | float",
    "timeout": "int | float",
    "delay": "int | float",
    "interval": "int | float",
    "frequency": "int | float",
    "rate": "int | float",
    "speed": "int | float",
    "distance": "int | float",
    "position": "int | float",
    "coordinate": "int | float",
    "index": "int",
    "offset": "int",
    "page": "int",
    "limit": "int",
    "threshold": "int | float",
    "minimum": "int | float",
    "maximum": "int | float",
    "total": "int | float",
    "sum": "int | float",
    "average": "int | float",
    "mean": "int | float",
    "percentage": "int | float",
    "ratio": "float",
    "factor": "int | float",
    "multiplier": "int | float",
    "score": "int | float",
    "rating": "int | float",
    "priority": "int",
    "level": "int",
    "rank": "int",
    "order": "int",
    "sequence": "int",
    "step": "int",
    "stage": "int",
    "phase": "int",
    "iteration": "int",
    "attempt": "int",
    "retry": "int",
    "port": "int",
    "code": "int | str",
    "exit_code": "int",
    "return_code": "int",
    "status_code": "int",
    
    # Boolean patterns
    "enabled": "bool",
    "disabled": "bool",
    "active": "bool",
    "inactive": "bool",
    "visible": "bool",
    "hidden": "bool",
    "open": "bool",
    "closed": "bool",
    "locked": "bool",
    "unlocked": "bool",
    "checked": "bool",
    "selected": "bool",
    "focused": "bool",
    "expanded": "bool",
    "collapsed": "bool",
    "required": "bool",
    "optional": "bool",
    "valid": "bool",
    "invalid": "bool",
    "success": "bool",
    "failed": "bool",
    "complete": "bool",
    "finished": "bool",
    "started": "bool",
    "running": "bool",
    "stopped": "bool",
    "paused": "bool",
    "cancelled": "bool",
    "confirmed": "bool",
    "approved": "bool",
    "authorized": "bool",
    "authenticated": "bool",
    "secure": "bool",
    "encrypted": "bool",
    "compressed": "bool",
    "cached": "bool",
    "async": "bool",
    "sync": "bool",
    "force": "bool",
    "strict": "bool",
    "verbose": "bool",
    "debug": "bool",
    "recursive": "bool",
    "overwrite": "bool",
    "backup": "bool",
    "auto": "bool",
    "manual": "bool",
    
    # Collection patterns
    "items": "list[Any]",
    "elements": "list[Any]",
    "values": "list[Any]",
    "keys": "list[str]",
    "results": "list[Any]",
    "responses": "list[Any]",
    "requests": "list[Any]",
    "files": "list[str | Path]",
    "paths": "list[str | Path]",
    "urls": "list[str]",
    "commands": "list[str]",
    "actions": "list[Any]",
    "operations": "list[Any]",
    "tasks": "list[Any]",
    "jobs": "list[Any]",
    "processes": "list[Any]",
    "threads": "list[Any]",
    "events": "list[Any]",
    "messages": "list[Any]",
    "notifications": "list[Any]",
    "errors": "list[str | Exception]",
    "warnings": "list[str]",
    "logs": "list[str]",
    "records": "list[Any]",
    "entries": "list[Any]",
    "rows": "list[Any]",
    "columns": "list[Any]",
    "fields": "list[str]",
    "attributes": "list[str]",
    "properties": "list[str]",
    "features": "list[str]",
    "options": "list[Any]",
    "parameters": "list[Any]",
    "arguments": "list[Any]",
    "args": "list[Any]",
    "kwargs": "dict[str, Any]",
    "params": "dict[str, Any]",
    "config": "dict[str, Any]",
    "settings": "dict[str, Any]",
    "options": "dict[str, Any]",
    "metadata": "dict[str, Any]",
    "headers": "dict[str, str]",
    "cookies": "dict[str, str]",
    "data": "dict[str, Any] | list[Any] | str | bytes",
    "payload": "dict[str, Any] | str | bytes",
    "context": "dict[str, Any] | Any",
    "environment": "dict[str, str]",
    "variables": "dict[str, Any]",
    "mapping": "dict[str, Any]",
    "lookup": "dict[str, Any]",
    "cache": "dict[str, Any]",
    "registry": "dict[str, Any]",
    "index": "dict[str, Any] | int",
    "schema": "dict[str, Any]",
    "template": "str | dict[str, Any]",
    "specification": "dict[str, Any]",
    "definition": "dict[str, Any]",
    "configuration": "dict[str, Any]",
    
    # Special common patterns
    "callback": "Callable[..., Any]",
    "handler": "Callable[..., Any]",
    "func": "Callable[..., Any]", 
    "function": "Callable[..., Any]",
    "method": "Callable[..., Any] | str",
    "processor": "Callable[..., Any]",
    "validator": "Callable[..., Any]",
    "filter": "Callable[..., Any]",
    "transformer": "Callable[..., Any]",
    "mapper": "Callable[..., Any]",
    "comparator": "Callable[..., Any]",
    "predicate": "Callable[..., bool]",
    "condition": "Callable[..., bool] | bool",
    "criteria": "Callable[..., bool] | Any",
    
    # File and IO patterns
    "file": "str | Path | Any",
    "stream": "Any",
    "buffer": "bytes | bytearray | Any",
    "reader": "Any",
    "writer": "Any",
    "input": "str | bytes | Any",
    "output": "str | bytes | Any",
    
    # Network patterns
    "host": "str",
    "hostname": "str",
    "address": "str",
    "ip": "str",
    "port": "int",
    "endpoint": "str",
    "connection": "Any",
    "socket": "Any",
    "client": "Any",
    "server": "Any",
    "session": "Any",
    "request": "Any",
    "response": "Any",
    
    # Time patterns
    "timestamp": "int | float | str",
    "datetime": "datetime | str",
    "date": "date | str",
    "timedelta": "timedelta | int | float",
    
    # Generic patterns for common suffixes
    "_id": "str",
    "_name": "str",
    "_type": "str",
    "_count": "int",
    "_size": "int",
    "_length": "int",
    "_time": "int | float",
    "_timeout": "int | float",
    "_delay": "int | float",
    "_rate": "int | float",
    "_enabled": "bool",
    "_required": "bool",
    "_optional": "bool",
    "_valid": "bool",
    "_active": "bool",
    "_list": "list[Any]",
    "_dict": "dict[str, Any]",
    "_data": "Any",
    "_config": "dict[str, Any]",
    "_settings": "dict[str, Any]",
    "_options": "dict[str, Any]",
    "_params": "dict[str, Any]",
    "_kwargs": "dict[str, Any]",
}

# Context-based type inference patterns
CONTEXT_PATTERNS = {
    # AsyncMock/Mock patterns
    r"Mock|AsyncMock": "Any",
    r"mock_\w+": "Any",
    
    # FastMCP Context
    r"context|ctx": "Context | Any",
    
    # Either patterns
    r"result|either": "Either[Any, Any] | Any",
    
    # Exception patterns
    r"error|exception|exc": "Exception | str",
    
    # Manager/Service patterns
    r"\w+_manager": "Any",
    r"\w+_service": "Any",
    r"\w+_client": "Any",
    r"\w+_handler": "Any",
    
    # Spec patterns
    r"\w+_spec": "Any",
    
    # Configuration patterns
    r"\w+_config": "dict[str, Any] | Any",
    r"\w+_settings": "dict[str, Any]",
    
    # ID patterns
    r"\w+_id": "str",
    
    # State patterns
    r"\w+_state": "Any",
    
    # Registry patterns
    r"\w+_registry": "dict[str, Any] | Any",
}

def infer_argument_type(arg_name: str, func_content: str, func_name: str) -> str:
    """Infer the type for a function argument based on name, context, and usage."""
    arg_lower = arg_name.lower()
    
    # Check direct mapping first
    if arg_lower in ARGUMENT_TYPE_MAPPING:
        return ARGUMENT_TYPE_MAPPING[arg_lower]
    
    # Check suffix patterns
    for suffix, type_hint in ARGUMENT_TYPE_MAPPING.items():
        if suffix.startswith('_') and arg_lower.endswith(suffix):
            return type_hint
    
    # Check context patterns
    for pattern, type_hint in CONTEXT_PATTERNS.items():
        if re.search(pattern, arg_name, re.IGNORECASE):
            return type_hint
    
    # Special handling for common test patterns
    if 'test_' in func_name or func_name.startswith('test_'):
        if arg_name in ['self']:
            return 'Any'  # Test class self
        if 'mock' in arg_lower:
            return 'Any'
        if arg_name in ['expected', 'actual']:
            return 'Any'
    
    # Special handling for fixture and setup functions
    if func_name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass'] or '_fixture' in func_name:
        if arg_name == 'self':
            return 'Any'
        if arg_name == 'cls':
            return 'type[Any]'
    
    # Check usage patterns in function body
    if f"str({arg_name})" in func_content or f"{arg_name}.strip(" in func_content:
        return "str"
    if f"int({arg_name})" in func_content or f"{arg_name} + 1" in func_content:
        return "int"
    if f"len({arg_name})" in func_content or f"for " in func_content and f" in {arg_name}" in func_content:
        return "list[Any] | str"
    if f"{arg_name}.get(" in func_content or f"{arg_name}[" in func_content:
        return "dict[str, Any] | list[Any]"
    if f"await {arg_name}" in func_content:
        return "Awaitable[Any] | Any"
    if f"{arg_name}(" in func_content:
        return "Callable[..., Any]"
    
    # Default based on common patterns
    if arg_name in ['self']:
        return 'Any'  # Don't annotate self
    if arg_name in ['cls']:
        return 'type[Any]'
    if arg_name in ['args']:
        return '*args: Any'
    if arg_name in ['kwargs']:
        return '**kwargs: Any'
    
    # Default to Any for unknown types
    return "Any"

def fix_function_argument_annotations(file_path: Path) -> tuple[bool, int]:
    """Fix missing argument type annotations in a single file using AST parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Parse the file to find functions with missing argument annotations
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, skipping")
            return False, 0
        
        # Check if typing imports are present
        has_typing_import = any(import_line in content for import_line in [
            "from typing import", "import typing", "from __future__ import annotations"
        ])
        
        # Find functions needing argument annotations
        functions_to_fix = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip functions that already have all annotations
                needs_fix = False
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg not in ['self', 'cls']:
                        needs_fix = True
                        break
                
                # Check for *args and **kwargs
                if node.args.vararg and node.args.vararg.annotation is None:
                    needs_fix = True
                if node.args.kwarg and node.args.kwarg.annotation is None:
                    needs_fix = True
                
                if needs_fix:
                    functions_to_fix.append(node)
        
        # Process each function
        for func_node in functions_to_fix:
            func_start_line = func_node.lineno - 1  # Convert to 0-based indexing
            
            # Get function content for context analysis
            lines = content.split('\n')
            func_lines = []
            indent_level = None
            
            for i in range(func_start_line, len(lines)):
                line = lines[i]
                if i == func_start_line:
                    # First line - get indentation
                    indent_level = len(line) - len(line.lstrip())
                elif line.strip() == "":
                    # Empty line
                    func_lines.append(line)
                elif len(line) - len(line.lstrip()) <= indent_level and line.strip():
                    # End of function (new function or class at same/lower indentation)
                    break
                else:
                    func_lines.append(line)
            
            func_content = '\n'.join(func_lines)
            func_name = func_node.name
            
            # Build new function signature
            signature_lines = []
            current_line = func_start_line
            in_signature = True
            
            while current_line < len(lines) and in_signature:
                line = lines[current_line]
                signature_lines.append(line)
                if ':' in line and not line.strip().endswith('\\'):
                    in_signature = False
                current_line += 1
            
            if not signature_lines:
                continue
                
            signature = '\n'.join(signature_lines)
            original_signature = signature
            
            # Process arguments
            modified_signature = signature
            
            # Handle regular arguments
            for arg in func_node.args.args:
                if arg.annotation is None and arg.arg not in ['self', 'cls']:
                    inferred_type = infer_argument_type(arg.arg, func_content, func_name)
                    
                    # Find the argument in the signature and add type annotation
                    # Use regex to find parameter and add type annotation
                    pattern = rf'\b{re.escape(arg.arg)}\b(?!\s*:)'
                    replacement = f'{arg.arg}: {inferred_type}'
                    
                    if re.search(pattern, modified_signature):
                        modified_signature = re.sub(pattern, replacement, modified_signature, count=1)
                        changes_made += 1
            
            # Handle *args
            if func_node.args.vararg and func_node.args.vararg.annotation is None:
                vararg_name = func_node.args.vararg.arg
                pattern = rf'\*{re.escape(vararg_name)}\b(?!\s*:)'
                replacement = f'*{vararg_name}: Any'
                
                if re.search(pattern, modified_signature):
                    modified_signature = re.sub(pattern, replacement, modified_signature, count=1)
                    changes_made += 1
            
            # Handle **kwargs
            if func_node.args.kwarg and func_node.args.kwarg.annotation is None:
                kwarg_name = func_node.args.kwarg.arg
                pattern = rf'\*\*{re.escape(kwarg_name)}\b(?!\s*:)'
                replacement = f'**{kwarg_name}: Any'
                
                if re.search(pattern, modified_signature):
                    modified_signature = re.sub(pattern, replacement, modified_signature, count=1)
                    changes_made += 1
            
            # Replace the original signature in content
            if modified_signature != original_signature:
                content = content.replace(original_signature, modified_signature)
        
        # Add typing imports if needed and changes were made
        if changes_made > 0 and not has_typing_import:
            # Find the best place to add typing import
            if "from __future__ import annotations" in content:
                # Already has future annotations, add typing import after other imports
                import_pattern = re.search(r'((?:^(?:from __future__|from|import)\s+[^\n]+\n)+)', content, re.MULTILINE)
                if import_pattern:
                    imports_end = import_pattern.end()
                    content = content[:imports_end] + "from typing import Any, Callable, Awaitable\n" + content[imports_end:]
            else:
                # Add future annotations and typing import
                import_pattern = re.search(r'((?:^(?:from|import)\s+[^\n]+\n)+)', content, re.MULTILINE)
                if import_pattern:
                    imports_start = import_pattern.start()
                    content = content[:imports_start] + "from __future__ import annotations\n\nfrom typing import Any, Callable, Awaitable\n" + content[imports_start:]
                else:
                    # Add at the beginning after docstring
                    docstring_end = 0
                    if content.startswith('"""') or content.startswith("'''"):
                        quote_type = '"""' if content.startswith('"""') else "'''"
                        end_pos = content.find(quote_type, 3)
                        if end_pos != -1:
                            docstring_end = end_pos + 3
                    
                    imports = "\nfrom __future__ import annotations\n\nfrom typing import Any, Callable, Awaitable\n"
                    content = content[:docstring_end] + imports + content[docstring_end:]
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        
        return False, 0
    
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False, 0

def process_directory(directory: Path, target_files: list[str] | None = None) -> tuple[int, int]:
    """Process Python files in directory."""
    files_modified = 0
    total_changes = 0
    
    # Get files to process
    if target_files:
        files_to_process = [directory / f for f in target_files if (directory / f).exists()]
    else:
        files_to_process = list(directory.rglob("*.py"))
    
    for py_file in files_to_process:
        if py_file.is_file() and not py_file.name.startswith('.'):
            try:
                modified, changes = fix_function_argument_annotations(py_file)
                if modified:
                    files_modified += 1
                    total_changes += changes
                    logger.info(f"Fixed {changes} argument annotations in {py_file.relative_to(PROJECT_ROOT)}")
            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")
    
    return files_modified, total_changes

def main() -> None:
    """Main execution function."""
    logger.info("Starting function argument type annotation fixing...")
    logger.info(f"Processing directory: {PROJECT_ROOT}")
    
    if not PROJECT_ROOT.exists():
        logger.error(f"Project root does not exist: {PROJECT_ROOT}")
        return
    
    # Focus on source files first (where most violations likely are)
    high_priority_dirs = [
        "src",
        "tests",
    ]
    
    files_modified = 0
    total_changes = 0
    
    for dir_name in high_priority_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            logger.info(f"Processing {dir_name} directory...")
            dir_files, dir_changes = process_directory(dir_path)
            files_modified += dir_files
            total_changes += dir_changes
    
    # Process remaining files
    logger.info("Processing remaining files...")
    remaining_files, remaining_changes = process_directory(PROJECT_ROOT)
    files_modified += remaining_files
    total_changes += remaining_changes
    
    logger.info("Function argument type annotation fix completed!")
    logger.info(f"Files modified: {files_modified}")
    logger.info(f"Total argument annotations added: {total_changes}")
    
    if files_modified > 0:
        logger.info("Remember to:")
        logger.info("1. Run ruff check to verify type annotations")
        logger.info("2. Run tests to ensure functionality is preserved")
        logger.info("3. Review the changes for any context-specific adjustments")

if __name__ == "__main__":
    main()