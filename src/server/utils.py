"""Utility functions for the Keyboard Maestro MCP server.

Contains shared parsing functions, helper utilities, and common operations
used across multiple server modules.
"""

import re
from typing import Any


def parse_group_applescript_records(applescript_output: str) -> list[dict[str, Any]]:
    """Parse AppleScript group records into Python dictionaries."""
    records = []

    # Clean up the output - remove extra whitespace and newlines
    clean_output = re.sub(r"\s+", " ", applescript_output.strip())

    # The actual AppleScript output is in flat comma-separated format
    # Parse format: key:value, key:value, key:value, ...
    # When we see 'groupName' again, it indicates a new record

    pairs = []
    current_pair = ""
    in_value = False
    paren_depth = 0

    # First, properly split by commas, handling nested content
    for char in clean_output:
        if char == "(" and not in_value:
            paren_depth += 1
        elif char == ")" and not in_value:
            paren_depth -= 1
        elif char == ":" and paren_depth == 0:
            in_value = True
        elif char == "," and paren_depth == 0 and in_value:
            pairs.append(current_pair.strip())
            current_pair = ""
            in_value = False
            continue

        current_pair += char

    # Don't forget the last pair
    if current_pair.strip():
        pairs.append(current_pair.strip())

    # Now parse the key:value pairs into records
    current_record: dict[str, Any] = {}
    for pair in pairs:
        if ":" in pair:
            # Split only on the first colon to handle values with colons
            key, raw_value = pair.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()

            # Clean up the value - remove extra quotes if present
            if raw_value.startswith('"') and raw_value.endswith('"'):
                raw_value = raw_value[1:-1]

            # Convert values to appropriate types
            value: Any
            if raw_value == "true":
                value = True
            elif raw_value == "false":
                value = False
            elif raw_value.isdigit() or raw_value.replace("-", "").isdigit():
                value = int(raw_value)
            else:
                value = raw_value

            # If we see groupName and we already have a record, start a new one
            if key == "groupName" and current_record:
                # Clean up the previous record before saving
                if "groupName" in current_record:
                    records.append(current_record)
                current_record = {}

            current_record[key] = value

    # Don't forget the last record
    if current_record and "groupName" in current_record:
        records.append(current_record)

    return records


def parse_variable_records(applescript_output: str | list[str]) -> list[dict[str, Any]]:
    """Parse AppleScript variable records into Python dictionaries."""
    records: list[dict[str, Any]] = []

    # Handle both string and list inputs
    if isinstance(applescript_output, list):
        if not applescript_output:
            return records
        applescript_str = "\n".join(applescript_output)
    else:
        applescript_str = applescript_output

    # Clean up the output - remove extra whitespace and newlines
    clean_output = re.sub(r"\s+", " ", applescript_str.strip())

    # The actual AppleScript output is in flat comma-separated format
    # Parse format: key:value, key:value, key:value, ...
    # When we see 'varName' again, it indicates a new record

    pairs = []
    current_pair = ""
    in_value = False
    paren_depth = 0

    # First, properly split by commas, handling nested content
    for char in clean_output:
        if char == "(" and not in_value:
            paren_depth += 1
        elif char == ")" and not in_value:
            paren_depth -= 1
        elif char == ":" and paren_depth == 0:
            in_value = True
        elif char == "," and paren_depth == 0 and in_value:
            pairs.append(current_pair.strip())
            current_pair = ""
            in_value = False
            continue

        current_pair += char

    # Don't forget the last pair
    if current_pair.strip():
        pairs.append(current_pair.strip())

    # Now parse the key:value pairs into records
    current_record: dict[str, Any] = {}
    for pair in pairs:
        if ":" in pair:
            # Split only on the first colon to handle values with colons
            key, value = pair.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Clean up the value - remove extra quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            # If we see varName and we already have a record, start a new one
            if key == "varName" and current_record:
                # Clean up the previous record before saving
                if "varName" in current_record:
                    # Transform to standard format
                    records.append(
                        {
                            "name": current_record.get("varName", ""),
                            "scope": current_record.get("varScope", "unknown"),
                            "type": current_record.get("varType", "string"),
                        },
                    )
                current_record = {}

            current_record[key] = value

    # Don't forget the last record
    if current_record and "varName" in current_record:
        records.append(
            {
                "name": current_record.get("varName", ""),
                "scope": current_record.get("varScope", "unknown"),
                "type": current_record.get("varType", "string"),
            },
        )

    return records


def validate_input(value: Any, expected_type: type) -> bool:
    """Validate input against expected type."""
    if value is None:
        return False
    return isinstance(value, expected_type)


def sanitize_output(data: Any) -> Any:
    """Sanitize output data to remove potentially dangerous content."""
    if isinstance(data, str):
        # Remove script tags and dangerous HTML
        import re

        sanitized = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            data,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sanitized = re.sub(
            r"<img[^>]*onerror[^>]*>",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
        return sanitized
    if isinstance(data, dict):
        return {key: sanitize_output(value) for key, value in data.items()}
    if isinstance(data, list):
        return [sanitize_output(item) for item in data]
    return data


def validate_input_schema(data: dict, schema: dict) -> dict:
    """Validate input data against a schema."""
    result: dict[str, Any] = {"valid": True, "errors": [], "data": {}}

    for field_name, field_config in schema.items():
        field_type = field_config.get("type", str)
        required = field_config.get("required", False)
        default = field_config.get("default")
        max_length = field_config.get("max_length")
        min_value = field_config.get("min_value")
        max_value = field_config.get("max_value")

        value = data.get(field_name, default)

        # Check required fields
        if required and value is None:
            result["valid"] = False
            result["errors"].append(f"Field '{field_name}' is required")
            continue

        # Skip validation if value is None and not required
        if value is None and not required:
            result["data"][field_name] = default
            continue

        # Type validation
        if not isinstance(value, field_type):
            result["valid"] = False
            result["errors"].append(
                f"Field '{field_name}' must be of type {field_type.__name__}",
            )
            continue

        # String length validation
        if field_type is str and max_length and len(value) > max_length:
            result["valid"] = False
            result["errors"].append(
                f"Field '{field_name}' exceeds maximum length of {max_length}",
            )
            continue

        # Numeric range validation
        if field_type is int and min_value is not None and value < min_value:
            result["valid"] = False
            result["errors"].append(
                f"Field '{field_name}' is below minimum value of {min_value}",
            )
            continue

        if field_type is int and max_value is not None and value > max_value:
            result["valid"] = False
            result["errors"].append(
                f"Field '{field_name}' exceeds maximum value of {max_value}",
            )
            continue

        result["data"][field_name] = value

    return result
