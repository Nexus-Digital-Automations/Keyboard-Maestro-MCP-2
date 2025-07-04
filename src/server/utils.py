"""
Utility functions for the Keyboard Maestro MCP server.

Contains shared parsing functions, helper utilities, and common operations
used across multiple server modules.
"""

import re
from typing import Any, Dict, List


def parse_group_applescript_records(applescript_output: str) -> List[Dict[str, Any]]:
    """Parse AppleScript group records into Python dictionaries."""
    records = []
    
    # Clean up the output - remove extra whitespace and newlines
    clean_output = re.sub(r'\s+', ' ', applescript_output.strip())
    
    # The actual AppleScript output is in flat comma-separated format
    # Parse format: key:value, key:value, key:value, ...
    # When we see 'groupName' again, it indicates a new record
    
    pairs = []
    current_pair = ""
    in_value = False
    paren_depth = 0
    
    # First, properly split by commas, handling nested content
    for char in clean_output:
        if char == '(' and not in_value:
            paren_depth += 1
        elif char == ')' and not in_value:
            paren_depth -= 1
        elif char == ':' and paren_depth == 0:
            in_value = True
        elif char == ',' and paren_depth == 0 and in_value:
            pairs.append(current_pair.strip())
            current_pair = ""
            in_value = False
            continue
        
        current_pair += char
    
    # Don't forget the last pair
    if current_pair.strip():
        pairs.append(current_pair.strip())
    
    # Now parse the key:value pairs into records
    current_record = {}
    for pair in pairs:
        if ':' in pair:
            # Split only on the first colon to handle values with colons
            key, value = pair.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Clean up the value - remove extra quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # Convert values to appropriate types
            if value == 'true':
                value = True
            elif value == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('-', '').isdigit():  # Handle negative numbers
                value = int(value)
            
            # If we see groupName and we already have a record, start a new one
            if key == 'groupName' and current_record:
                # Clean up the previous record before saving
                if 'groupName' in current_record:
                    records.append(current_record)
                current_record = {}
            
            current_record[key] = value
    
    # Don't forget the last record
    if current_record and 'groupName' in current_record:
        records.append(current_record)
    
    return records


def parse_variable_records(applescript_output: str) -> List[Dict[str, Any]]:
    """Parse AppleScript variable records into Python dictionaries."""
    records = []
    
    # Clean up the output - remove extra whitespace and newlines
    clean_output = re.sub(r'\s+', ' ', applescript_output.strip())
    
    # The actual AppleScript output is in flat comma-separated format
    # Parse format: key:value, key:value, key:value, ...
    # When we see 'varName' again, it indicates a new record
    
    pairs = []
    current_pair = ""
    in_value = False
    paren_depth = 0
    
    # First, properly split by commas, handling nested content
    for char in clean_output:
        if char == '(' and not in_value:
            paren_depth += 1
        elif char == ')' and not in_value:
            paren_depth -= 1
        elif char == ':' and paren_depth == 0:
            in_value = True
        elif char == ',' and paren_depth == 0 and in_value:
            pairs.append(current_pair.strip())
            current_pair = ""
            in_value = False
            continue
        
        current_pair += char
    
    # Don't forget the last pair
    if current_pair.strip():
        pairs.append(current_pair.strip())
    
    # Now parse the key:value pairs into records
    current_record = {}
    for pair in pairs:
        if ':' in pair:
            # Split only on the first colon to handle values with colons
            key, value = pair.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Clean up the value - remove extra quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # If we see varName and we already have a record, start a new one
            if key == 'varName' and current_record:
                # Clean up the previous record before saving
                if 'varName' in current_record:
                    # Transform to standard format
                    records.append({
                        "name": current_record.get("varName", ""),
                        "scope": current_record.get("varScope", "unknown"),
                        "type": current_record.get("varType", "string")
                    })
                current_record = {}
            
            current_record[key] = value
    
    # Don't forget the last record
    if current_record and 'varName' in current_record:
        records.append({
            "name": current_record.get("varName", ""),
            "scope": current_record.get("varScope", "unknown"),
            "type": current_record.get("varType", "string")
        })
    
    return records