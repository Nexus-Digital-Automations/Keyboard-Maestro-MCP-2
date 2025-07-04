"""
Message template system for Keyboard Maestro MCP Tools.

This module provides a comprehensive template system for reusable communication
patterns with secure variable substitution and template management.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
import re
import json
import uuid
from datetime import datetime, UTC
from pathlib import Path

from ..core.communication import (
    MessageTemplate, TemplateId, CommunicationType, MessagePriority
)
from ..core.either import Either
from ..core.errors import ValidationError, SecurityError, DataError as FileSystemError
from ..core.contracts import require, ensure


@dataclass(frozen=True)
class TemplateLibrary:
    """Collection of message templates with metadata."""
    library_id: str
    name: str
    description: str
    templates: Dict[TemplateId, MessageTemplate] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0.0"
    
    def __post_init__(self):
        if not self.library_id or len(self.library_id) == 0:
            raise ValidationError("Library ID cannot be empty")
        
        if not self.name or len(self.name) == 0:
            raise ValidationError("Library name cannot be empty")


@dataclass(frozen=True)
class TemplateVariable:
    """Template variable definition with validation rules."""
    name: str
    description: str
    variable_type: str = "string"  # string, number, email, phone, url
    required: bool = True
    default_value: Optional[str] = None
    validation_pattern: Optional[str] = None
    max_length: Optional[int] = None
    
    def __post_init__(self):
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', self.name):
            raise ValidationError(f"Invalid variable name: {self.name}")
        
        if self.max_length and self.max_length <= 0:
            raise ValidationError("Max length must be positive")
        
        if self.validation_pattern:
            try:
                re.compile(self.validation_pattern)
            except re.error as e:
                raise ValidationError(f"Invalid validation pattern: {e}")


class TemplateSecurityValidator:
    """Security validation for template content and variables."""
    
    @staticmethod
    def validate_template_content(subject: Optional[str], body: str) -> Either[SecurityError, None]:
        """Validate template content for security threats."""
        # Check for template injection attempts
        if subject and TemplateSecurityValidator._contains_template_injection(subject):
            return Either.left(SecurityError("TEMPLATE_INJECTION", "Subject contains template injection"))
        
        if TemplateSecurityValidator._contains_template_injection(body):
            return Either.left(SecurityError("TEMPLATE_INJECTION", "Body contains template injection"))
        
        # Check for script injection
        if subject and TemplateSecurityValidator._contains_script_injection(subject):
            return Either.left(SecurityError("SCRIPT_INJECTION", "Subject contains script injection"))
        
        if TemplateSecurityValidator._contains_script_injection(body):
            return Either.left(SecurityError("SCRIPT_INJECTION", "Body contains script injection"))
        
        return Either.right(None)
    
    @staticmethod
    def _contains_template_injection(text: str) -> bool:
        """Check for template injection patterns."""
        dangerous_patterns = [
            r'\{\{.*\}\}',  # Jinja-style templates
            r'\$\{.*\}',    # Shell-style substitution
            r'<%.*%>',      # ASP/ERB-style templates
            r'\{%.*%\}',    # Django-style template tags
            r'<\?.*\?>',    # PHP-style tags
        ]
        
        return any(re.search(pattern, text, re.DOTALL) for pattern in dangerous_patterns)
    
    @staticmethod
    def _contains_script_injection(text: str) -> bool:
        """Check for script injection attempts."""
        script_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in script_patterns)
    
    @staticmethod
    def validate_variable_value(variable: TemplateVariable, value: str) -> Either[ValidationError, str]:
        """Validate variable value against its definition."""
        if not value and variable.required:
            return Either.left(ValidationError(f"Required variable '{variable.name}' is missing"))
        
        if not value and not variable.required:
            return Either.right(variable.default_value or "")
        
        # Check length
        if variable.max_length and len(value) > variable.max_length:
            return Either.left(ValidationError(
                f"Variable '{variable.name}' too long: {len(value)} > {variable.max_length}"
            ))
        
        # Type validation
        validation_result = TemplateSecurityValidator._validate_by_type(
            variable.variable_type, value, variable.name
        )
        if validation_result.is_left():
            return validation_result
        
        # Pattern validation
        if variable.validation_pattern:
            if not re.match(variable.validation_pattern, value):
                return Either.left(ValidationError(
                    f"Variable '{variable.name}' doesn't match required pattern"
                ))
        
        # Security validation
        if TemplateSecurityValidator._contains_injection_attempts(value):
            return Either.left(ValidationError(
                f"Variable '{variable.name}' contains potentially dangerous content"
            ))
        
        return Either.right(value)
    
    @staticmethod
    def _validate_by_type(var_type: str, value: str, var_name: str) -> Either[ValidationError, str]:
        """Validate value by its declared type."""
        if var_type == "email":
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                return Either.left(ValidationError(f"Variable '{var_name}' is not a valid email"))
        
        elif var_type == "phone":
            phone_pattern = r'^[\+]?[1-9][\d]{0,15}$'
            clean_phone = re.sub(r'[^\d+]', '', value)
            if not re.match(phone_pattern, clean_phone):
                return Either.left(ValidationError(f"Variable '{var_name}' is not a valid phone"))
        
        elif var_type == "url":
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, value):
                return Either.left(ValidationError(f"Variable '{var_name}' is not a valid URL"))
        
        elif var_type == "number":
            try:
                float(value)
            except ValueError:
                return Either.left(ValidationError(f"Variable '{var_name}' is not a valid number"))
        
        return Either.right(value)
    
    @staticmethod
    def _contains_injection_attempts(value: str) -> bool:
        """Check for common injection attempts in variable values."""
        injection_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'data:text/html',
            r'eval\s*\(',
            r'\{.*\}',  # Template-like patterns
            r'<%.*%>',
            r'\$\{.*\}',
        ]
        
        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in injection_patterns)


class MessageTemplateManager:
    """Comprehensive template management system."""
    
    def __init__(self, template_directory: Optional[Path] = None):
        self.template_directory = template_directory or Path("./templates")
        self.libraries: Dict[str, TemplateLibrary] = {}
        self.security_validator = TemplateSecurityValidator()
        
        # Built-in templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default template library."""
        default_library = TemplateLibrary(
            library_id="default",
            name="Default Templates",
            description="Built-in message templates for common use cases"
        )
        
        # Common notification template
        notification_template = MessageTemplate(
            template_id=TemplateId("notification_basic"),
            name="Basic Notification",
            subject_template="Notification: {title}",
            body_template="Hello {recipient_name},\n\n{message}\n\nBest regards,\n{sender_name}",
            communication_type=CommunicationType.EMAIL
        )
        
        # Status update template
        status_template = MessageTemplate(
            template_id=TemplateId("status_update"),
            name="Status Update",
            subject_template="Status Update: {project_name}",
            body_template="Hi {recipient_name},\n\nStatus update for {project_name}:\n\nCurrent Status: {status}\nProgress: {progress}%\nNext Steps: {next_steps}\n\nRegards,\n{sender_name}",
            communication_type=CommunicationType.EMAIL
        )
        
        # SMS alert template
        sms_alert_template = MessageTemplate(
            template_id=TemplateId("sms_alert"),
            name="SMS Alert",
            body_template="ALERT: {alert_type} - {message}. Time: {timestamp}",
            communication_type=CommunicationType.SMS
        )
        
        # Add templates to library
        templates = {
            notification_template.template_id: notification_template,
            status_template.template_id: status_template,
            sms_alert_template.template_id: sms_alert_template,
        }
        
        object.__setattr__(default_library, 'templates', templates)
        self.libraries["default"] = default_library
    
    @require(lambda self, template: isinstance(template, MessageTemplate))
    @ensure(lambda result: isinstance(result, Either))
    def add_template(self, template: MessageTemplate, library_id: str = "default") -> Either[ValidationError, None]:
        """Add a new template to the specified library."""
        try:
            # Security validation
            security_result = self.security_validator.validate_template_content(
                template.subject_template, template.body_template
            )
            if security_result.is_left():
                return Either.left(ValidationError(
                    f"Template security validation failed: {security_result.get_left().message}"
                ))
            
            # Get or create library
            if library_id not in self.libraries:
                library = TemplateLibrary(
                    library_id=library_id,
                    name=f"Library {library_id}",
                    description=f"Template library {library_id}"
                )
                self.libraries[library_id] = library
            
            library = self.libraries[library_id]
            
            # Check for duplicate template ID
            if template.template_id in library.templates:
                return Either.left(ValidationError(
                    f"Template ID '{template.template_id}' already exists in library '{library_id}'"
                ))
            
            # Add template
            new_templates = dict(library.templates)
            new_templates[template.template_id] = template
            
            updated_library = TemplateLibrary(
                library_id=library.library_id,
                name=library.name,
                description=library.description,
                templates=new_templates,
                created_at=library.created_at,
                version=library.version
            )
            
            self.libraries[library_id] = updated_library
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError("template", str(e), f"Failed to add template: {str(e)}"))
    
    def get_template(self, template_id: TemplateId, library_id: str = "default") -> Either[ValidationError, MessageTemplate]:
        """Retrieve a template by ID from the specified library."""
        if library_id not in self.libraries:
            return Either.left(ValidationError(f"Library '{library_id}' not found"))
        
        library = self.libraries[library_id]
        if template_id not in library.templates:
            return Either.left(ValidationError("template_id", template_id, f"Template '{template_id}' not found in library '{library_id}'"))
        
        return Either.right(library.templates[template_id])
    
    def list_templates(self, library_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """List all templates, optionally filtered by library."""
        if library_id:
            if library_id not in self.libraries:
                return {}
            
            libraries_to_process = {library_id: self.libraries[library_id]}
        else:
            libraries_to_process = self.libraries
        
        result = {}
        for lib_id, library in libraries_to_process.items():
            template_list = []
            for template_id, template in library.templates.items():
                template_info = {
                    "template_id": template_id,
                    "name": template.name,
                    "communication_type": template.communication_type.value,
                    "variables": list(template.variables),
                    "created_at": template.created_at.isoformat()
                }
                template_list.append(template_info)
            
            result[lib_id] = template_list
        
        return result
    
    @require(lambda self, template, variables: isinstance(template, MessageTemplate))
    @require(lambda self, template, variables: isinstance(variables, dict))
    @ensure(lambda result: isinstance(result, Either))
    def render_template(self, template: MessageTemplate, variables: Dict[str, str]) -> Either[ValidationError, Dict[str, str]]:
        """Render template with provided variables and security validation."""
        try:
            # Validate all variables
            validated_variables = {}
            for var_name in template.variables:
                if var_name not in variables:
                    if var_name == "timestamp":  # Auto-generate timestamp if needed
                        validated_variables[var_name] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                    else:
                        return Either.left(ValidationError(f"Missing required variable: {var_name}"))
                else:
                    value = variables[var_name]
                    
                    # Basic security validation for variable values
                    if self.security_validator._contains_injection_attempts(value):
                        return Either.left(ValidationError(
                            f"Variable '{var_name}' contains potentially dangerous content"
                        ))
                    
                    # Length validation
                    if len(value) > 1000:
                        return Either.left(ValidationError(
                            f"Variable '{var_name}' value too long: {len(value)} chars"
                        ))
                    
                    validated_variables[var_name] = value
            
            # Render template
            rendered = template.render(validated_variables)
            
            # Final security check on rendered content
            final_security_check = self.security_validator.validate_template_content(
                rendered.get("subject"), rendered["body"]
            )
            if final_security_check.is_left():
                return Either.left(ValidationError(
                    f"Rendered template failed security validation: {final_security_check.get_left().message}"
                ))
            
            return Either.right(rendered)
            
        except Exception as e:
            return Either.left(ValidationError(f"Template rendering failed: {str(e)}"))
    
    def save_library(self, library_id: str, file_path: Optional[Path] = None) -> Either[FileSystemError, None]:
        """Save template library to file."""
        try:
            if library_id not in self.libraries:
                return Either.left(FileSystemError(f"Library '{library_id}' not found"))
            
            library = self.libraries[library_id]
            
            if not file_path:
                self.template_directory.mkdir(parents=True, exist_ok=True)
                file_path = self.template_directory / f"{library_id}.json"
            
            # Convert library to serializable format
            library_data = {
                "library_id": library.library_id,
                "name": library.name,
                "description": library.description,
                "version": library.version,
                "created_at": library.created_at.isoformat(),
                "templates": {}
            }
            
            for template_id, template in library.templates.items():
                library_data["templates"][template_id] = {
                    "template_id": template_id,
                    "name": template.name,
                    "subject_template": template.subject_template,
                    "body_template": template.body_template,
                    "communication_type": template.communication_type.value,
                    "created_at": template.created_at.isoformat()
                }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(library_data, f, indent=2, ensure_ascii=False)
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(FileSystemError(f"Failed to save library: {str(e)}"))
    
    def load_library(self, file_path: Path) -> Either[FileSystemError, str]:
        """Load template library from file."""
        try:
            if not file_path.exists():
                return Either.left(FileSystemError(f"Template file not found: {file_path}"))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                library_data = json.load(f)
            
            # Reconstruct templates
            templates = {}
            for template_id, template_data in library_data["templates"].items():
                template = MessageTemplate(
                    template_id=TemplateId(template_data["template_id"]),
                    name=template_data["name"],
                    subject_template=template_data.get("subject_template"),
                    body_template=template_data["body_template"],
                    communication_type=CommunicationType(template_data["communication_type"]),
                    created_at=datetime.fromisoformat(template_data["created_at"])
                )
                templates[template.template_id] = template
            
            # Create library
            library = TemplateLibrary(
                library_id=library_data["library_id"],
                name=library_data["name"],
                description=library_data["description"],
                templates=templates,
                created_at=datetime.fromisoformat(library_data["created_at"]),
                version=library_data["version"]
            )
            
            self.libraries[library.library_id] = library
            return Either.right(library.library_id)
            
        except Exception as e:
            return Either.left(FileSystemError(f"Failed to load library: {str(e)}"))
    
    def delete_template(self, template_id: TemplateId, library_id: str = "default") -> Either[ValidationError, None]:
        """Delete a template from the specified library."""
        try:
            if library_id not in self.libraries:
                return Either.left(ValidationError(f"Library '{library_id}' not found"))
            
            library = self.libraries[library_id]
            if template_id not in library.templates:
                return Either.left(ValidationError(f"Template '{template_id}' not found"))
            
            # Create new library without the template
            new_templates = {tid: template for tid, template in library.templates.items() 
                           if tid != template_id}
            
            updated_library = TemplateLibrary(
                library_id=library.library_id,
                name=library.name,
                description=library.description,
                templates=new_templates,
                created_at=library.created_at,
                version=library.version
            )
            
            self.libraries[library_id] = updated_library
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError(f"Failed to delete template: {str(e)}"))
    
    def get_template_variables(self, template_id: TemplateId, library_id: str = "default") -> Either[ValidationError, Set[str]]:
        """Get the list of variables required by a template."""
        template_result = self.get_template(template_id, library_id)
        if template_result.is_left():
            return template_result
        
        template = template_result.get_right()
        return Either.right(template.variables)