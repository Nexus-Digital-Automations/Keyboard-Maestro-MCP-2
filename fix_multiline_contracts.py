#!/usr/bin/env python3
"""
Script to find and fix multiline commented contract decorators causing syntax errors.
"""

import os
import re
import glob
from pathlib import Path

def find_and_fix_multiline_contracts(root_dir):
    """Find and fix multiline commented contract decorators."""
    
    # Pattern to match multiline commented contracts
    multiline_pattern = re.compile(
        r'# FIXME: Contract disabled - @require\(\s*\n\s*lambda.*?\n.*?\n\s*\)',
        re.MULTILINE | re.DOTALL
    )
    
    # Pattern to match multiline contracts with messages
    multiline_with_message_pattern = re.compile(
        r'# FIXME: Contract disabled - @require\(\s*\n\s*lambda.*?\n.*?\n.*?\".*?\",\s*\n\s*\)',
        re.MULTILINE | re.DOTALL
    )
    
    files_fixed = []
    
    # Find all Python files
    for py_file in glob.glob(os.path.join(root_dir, "**/*.py"), recursive=True):
        if not os.path.exists(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Fix multiline contracts with messages
            def fix_multiline_with_message(match):
                text = match.group(0)
                # Extract the lambda and message parts
                lines = text.split('\n')
                lambda_parts = []
                message_part = None
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('lambda'):
                        lambda_parts.append(stripped)
                    elif stripped.startswith('and '):
                        lambda_parts.append(stripped)
                    elif stripped.startswith('"') and stripped.endswith('",'):
                        message_part = stripped[:-1]  # Remove trailing comma
                    elif stripped.startswith('"') and stripped.endswith('"'):
                        message_part = stripped
                
                # Join lambda parts
                lambda_expr = ' '.join(lambda_parts)
                
                if message_part:
                    return f'# FIXME: Contract disabled - @require({lambda_expr}, {message_part})'
                else:
                    return f'# FIXME: Contract disabled - @require({lambda_expr})'
            
            # Apply fixes
            content = multiline_with_message_pattern.sub(fix_multiline_with_message, content)
            
            # Fix regular multiline contracts
            def fix_multiline_regular(match):
                text = match.group(0)
                lines = text.split('\n')
                lambda_parts = []
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('lambda'):
                        lambda_parts.append(stripped)
                    elif stripped.startswith('and '):
                        lambda_parts.append(stripped)
                    elif stripped and not stripped.startswith('#') and not stripped.startswith(')'):
                        lambda_parts.append(stripped)
                
                lambda_expr = ' '.join(lambda_parts)
                return f'# FIXME: Contract disabled - @require({lambda_expr})'
            
            content = multiline_pattern.sub(fix_multiline_regular, content)
            
            # Write back if changed
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                files_fixed.append(py_file)
                print(f"Fixed: {py_file}")
                
        except Exception as e:
            print(f"Error processing {py_file}: {e}")
    
    return files_fixed

def main():
    """Main function."""
    root_dir = "/Users/jeremyparker/Desktop/Claude Coding Projects/Keyboard-Maestro-MCP"
    
    print("Finding and fixing multiline commented contract decorators...")
    fixed_files = find_and_fix_multiline_contracts(root_dir)
    
    if fixed_files:
        print(f"\nFixed {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
    else:
        print("\nNo files needed fixing.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()