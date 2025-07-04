"""
Template Manager for Knowledge Management System.

This module provides comprehensive template management for standardized documentation
generation, including template creation, validation, and dynamic content population.
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from pathlib import Path
import asyncio
import logging

from src.core.knowledge_architecture import (
    ContentId, DocumentType, ContentFormat, KnowledgeCategory,
    create_content_id, KnowledgeError
)
from src.core.contracts import require, ensure
from ..core.either import Either

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of content templates."""
    DOCUMENTATION = "documentation"
    GUIDE = "guide"
    REFERENCE = "reference"
    REPORT = "report"
    TUTORIAL = "tutorial"
    API_DOC = "api_documentation"
    USER_MANUAL = "user_manual"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class TemplateVariable:
    """Template variable definition."""
    name: str
    description: str
    variable_type: str = "string"  # string, number, boolean, array, object
    required: bool = True
    default_value: Optional[Any] = None
    validation_pattern: Optional[str] = None
    possible_values: Optional[List[str]] = None
    
    def validate_value(self, value: Any) -> bool:
        """Validate variable value."""
        if self.required and value is None:
            return False
        
        if value is None:
            return True
        
        # Type validation
        if self.variable_type == "string" and not isinstance(value, str):
            return False
        elif self.variable_type == "number" and not isinstance(value, (int, float)):
            return False
        elif self.variable_type == "boolean" and not isinstance(value, bool):
            return False
        elif self.variable_type == "array" and not isinstance(value, list):
            return False
        elif self.variable_type == "object" and not isinstance(value, dict):
            return False
        
        # Pattern validation
        if self.validation_pattern and isinstance(value, str):
            if not re.match(self.validation_pattern, value):
                return False
        
        # Possible values validation
        if self.possible_values and value not in self.possible_values:
            return False
        
        return True


@dataclass
class ContentTemplate:
    """Content template definition."""
    template_id: str
    name: str
    description: str
    template_type: TemplateType
    content_structure: str  # Template content with placeholders
    variables: List[TemplateVariable] = field(default_factory=list)
    output_formats: Set[ContentFormat] = field(default_factory=lambda: {ContentFormat.MARKDOWN})
    usage_guidelines: str = ""
    auto_populate: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    author: str = "system"
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    
    def get_required_variables(self) -> List[TemplateVariable]:
        """Get required variables for template."""
        return [var for var in self.variables if var.required]
    
    def get_variable_by_name(self, name: str) -> Optional[TemplateVariable]:
        """Get variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def extract_placeholders(self) -> Set[str]:
        """Extract all placeholders from template content."""
        # Find all {{variable}} placeholders
        placeholders = set(re.findall(r'\{\{([^}]+)\}\}', self.content_structure))
        return placeholders
    
    def validate_structure(self) -> List[str]:
        """Validate template structure and return errors."""
        errors = []
        
        # Check for undefined placeholders
        placeholders = self.extract_placeholders()
        defined_variables = {var.name for var in self.variables}
        
        undefined_placeholders = placeholders - defined_variables
        if undefined_placeholders:
            errors.append(f"Undefined placeholders: {', '.join(undefined_placeholders)}")
        
        # Check for unused variables
        unused_variables = defined_variables - placeholders
        if unused_variables:
            errors.append(f"Unused variables: {', '.join(unused_variables)}")
        
        return errors


@dataclass
class TemplateRenderContext:
    """Context for template rendering."""
    variables: Dict[str, Any]
    output_format: ContentFormat
    auto_populate_enabled: bool = True
    strict_validation: bool = True
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get variable value with default."""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set variable value."""
        self.variables[name] = value


class TemplateManager:
    """Template management system for knowledge content."""
    
    def __init__(self):
        self.templates: Dict[str, ContentTemplate] = {}
        self.template_cache: Dict[str, str] = {}  # Rendered template cache
        self.built_in_templates: Dict[str, ContentTemplate] = {}
        self._initialize_builtin_templates()
        
        logger.info("TemplateManager initialized")
    
    def _initialize_builtin_templates(self) -> None:
        """Initialize built-in templates."""
        # Documentation template
        doc_template = ContentTemplate(
            template_id="documentation_standard",
            name="Standard Documentation",
            description="Standard documentation template with common sections",
            template_type=TemplateType.DOCUMENTATION,
            content_structure="""# {{title}}

## Overview
{{overview}}

## Purpose
{{purpose}}

## Key Features
{{#features}}
- {{.}}
{{/features}}

## Usage
{{usage}}

## Parameters
{{#parameters}}
### {{name}}
- **Type**: {{type}}
- **Required**: {{required}}
- **Description**: {{description}}
{{/parameters}}

## Examples
{{examples}}

## Notes
{{notes}}

## Related Documentation
{{related_docs}}
""",
            variables=[
                TemplateVariable("title", "Document title", required=True),
                TemplateVariable("overview", "Document overview", required=True),
                TemplateVariable("purpose", "Document purpose", required=True),
                TemplateVariable("features", "List of key features", "array", required=False),
                TemplateVariable("usage", "Usage instructions", required=True),
                TemplateVariable("parameters", "Parameter definitions", "array", required=False),
                TemplateVariable("examples", "Usage examples", required=False),
                TemplateVariable("notes", "Additional notes", required=False),
                TemplateVariable("related_docs", "Related documentation", required=False)
            ],
            usage_guidelines="Use for standard documentation with overview, usage, and examples"
        )
        
        # API Documentation template
        api_template = ContentTemplate(
            template_id="api_documentation",
            name="API Documentation",
            description="Template for API endpoint documentation",
            template_type=TemplateType.API_DOC,
            content_structure="""# {{api_name}} API

## {{endpoint_name}}

**Endpoint**: `{{method}} {{path}}`

### Description
{{description}}

### Parameters
{{#parameters}}
#### {{name}}
- **Type**: {{type}}
- **Required**: {{required}}
- **Description**: {{description}}
{{#default_value}}
- **Default**: {{default_value}}
{{/default_value}}
{{/parameters}}

### Request Example
```{{request_format}}
{{request_example}}
```

### Response Example
```{{response_format}}
{{response_example}}
```

### Error Codes
{{#error_codes}}
- **{{code}}**: {{message}}
{{/error_codes}}

### Notes
{{notes}}
""",
            variables=[
                TemplateVariable("api_name", "API name", required=True),
                TemplateVariable("endpoint_name", "Endpoint name", required=True),
                TemplateVariable("method", "HTTP method", required=True),
                TemplateVariable("path", "API path", required=True),
                TemplateVariable("description", "Endpoint description", required=True),
                TemplateVariable("parameters", "Parameters", "array", required=False),
                TemplateVariable("request_format", "Request format", required=False, default_value="json"),
                TemplateVariable("request_example", "Request example", required=False),
                TemplateVariable("response_format", "Response format", required=False, default_value="json"),
                TemplateVariable("response_example", "Response example", required=False),
                TemplateVariable("error_codes", "Error codes", "array", required=False),
                TemplateVariable("notes", "Additional notes", required=False)
            ],
            usage_guidelines="Use for API endpoint documentation with examples and error codes"
        )
        
        # User Guide template
        guide_template = ContentTemplate(
            template_id="user_guide",
            name="User Guide",
            description="Template for user guides and tutorials",
            template_type=TemplateType.GUIDE,
            content_structure="""# {{title}}

## Introduction
{{introduction}}

## Prerequisites
{{#prerequisites}}
- {{.}}
{{/prerequisites}}

## Getting Started
{{getting_started}}

## Step-by-Step Instructions
{{#steps}}
### Step {{step_number}}: {{step_title}}
{{step_description}}

{{#step_example}}
**Example:**
```
{{step_example}}
```
{{/step_example}}
{{/steps}}

## Troubleshooting
{{#troubleshooting}}
### {{issue}}
{{solution}}

{{/troubleshooting}}

## Advanced Usage
{{advanced_usage}}

## Best Practices
{{#best_practices}}
- {{.}}
{{/best_practices}}

## FAQ
{{#faq}}
**Q: {{question}}**
A: {{answer}}

{{/faq}}
""",
            variables=[
                TemplateVariable("title", "Guide title", required=True),
                TemplateVariable("introduction", "Introduction", required=True),
                TemplateVariable("prerequisites", "Prerequisites", "array", required=False),
                TemplateVariable("getting_started", "Getting started section", required=True),
                TemplateVariable("steps", "Step-by-step instructions", "array", required=True),
                TemplateVariable("troubleshooting", "Troubleshooting section", "array", required=False),
                TemplateVariable("advanced_usage", "Advanced usage", required=False),
                TemplateVariable("best_practices", "Best practices", "array", required=False),
                TemplateVariable("faq", "FAQ", "array", required=False)
            ],
            usage_guidelines="Use for user guides with step-by-step instructions and troubleshooting"
        )
        
        # Store built-in templates
        self.built_in_templates = {
            "documentation_standard": doc_template,
            "api_documentation": api_template,
            "user_guide": guide_template
        }
        
        # Add to main templates collection
        self.templates.update(self.built_in_templates)
    
    @require(lambda template: template.name.strip(), "Template name required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns template or error")
    async def create_template(self, template: ContentTemplate) -> Either[str, ContentTemplate]:
        """Create new content template."""
        try:
            # Validate template structure
            validation_errors = template.validate_structure()
            if validation_errors:
                return Either.left(f"Template validation failed: {'; '.join(validation_errors)}")
            
            # Check for duplicate template ID
            if template.template_id in self.templates:
                return Either.left(f"Template with ID '{template.template_id}' already exists")
            
            # Store template
            self.templates[template.template_id] = template
            
            logger.info(f"Created template: {template.template_id}")
            return Either.right(template)
            
        except Exception as e:
            error_msg = f"Failed to create template: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def get_template(self, template_id: str) -> Either[str, ContentTemplate]:
        """Get template by ID."""
        try:
            if template_id not in self.templates:
                return Either.left(f"Template '{template_id}' not found")
            
            template = self.templates[template_id]
            return Either.right(template)
            
        except Exception as e:
            error_msg = f"Failed to get template: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def list_templates(self, 
                           template_type: Optional[TemplateType] = None,
                           tags: Optional[Set[str]] = None) -> List[ContentTemplate]:
        """List available templates with optional filtering."""
        try:
            templates = list(self.templates.values())
            
            # Filter by type
            if template_type:
                templates = [t for t in templates if t.template_type == template_type]
            
            # Filter by tags
            if tags:
                templates = [t for t in templates if tags.issubset(t.tags)]
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            return []
    
    @require(lambda template_id, context: template_id.strip(), "Template ID required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns content or error")
    async def render_template(self, 
                            template_id: str, 
                            context: TemplateRenderContext) -> Either[str, str]:
        """Render template with given context."""
        try:
            # Get template
            template_result = await self.get_template(template_id)
            if template_result.is_left():
                return template_result
            
            template = template_result.right()
            
            # Validate context variables
            if context.strict_validation:
                validation_result = self._validate_context(template, context)
                if validation_result.is_left():
                    return validation_result
            
            # Auto-populate variables if enabled
            if context.auto_populate_enabled and template.auto_populate:
                await self._auto_populate_variables(template, context)
            
            # Render template
            rendered_content = self._render_template_content(template, context)
            
            # Format for output format
            if context.output_format != ContentFormat.MARKDOWN:
                rendered_content = await self._convert_format(rendered_content, context.output_format)
            
            return Either.right(rendered_content)
            
        except Exception as e:
            error_msg = f"Failed to render template: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def update_template(self, 
                            template_id: str, 
                            updates: Dict[str, Any]) -> Either[str, ContentTemplate]:
        """Update existing template."""
        try:
            if template_id not in self.templates:
                return Either.left(f"Template '{template_id}' not found")
            
            template = self.templates[template_id]
            
            # Create updated template
            updated_template = ContentTemplate(
                template_id=template.template_id,
                name=updates.get('name', template.name),
                description=updates.get('description', template.description),
                template_type=template.template_type,
                content_structure=updates.get('content_structure', template.content_structure),
                variables=updates.get('variables', template.variables),
                output_formats=updates.get('output_formats', template.output_formats),
                usage_guidelines=updates.get('usage_guidelines', template.usage_guidelines),
                auto_populate=updates.get('auto_populate', template.auto_populate),
                validation_rules=updates.get('validation_rules', template.validation_rules),
                created_at=template.created_at,
                modified_at=datetime.utcnow(),
                author=updates.get('author', template.author),
                version=updates.get('version', template.version),
                tags=updates.get('tags', template.tags)
            )
            
            # Validate updated template
            validation_errors = updated_template.validate_structure()
            if validation_errors:
                return Either.left(f"Updated template validation failed: {'; '.join(validation_errors)}")
            
            # Store updated template
            self.templates[template_id] = updated_template
            
            # Clear cache
            self._clear_template_cache(template_id)
            
            logger.info(f"Updated template: {template_id}")
            return Either.right(updated_template)
            
        except Exception as e:
            error_msg = f"Failed to update template: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def delete_template(self, template_id: str) -> Either[str, bool]:
        """Delete template."""
        try:
            if template_id not in self.templates:
                return Either.left(f"Template '{template_id}' not found")
            
            # Check if it's a built-in template
            if template_id in self.built_in_templates:
                return Either.left(f"Cannot delete built-in template '{template_id}'")
            
            # Delete template
            del self.templates[template_id]
            
            # Clear cache
            self._clear_template_cache(template_id)
            
            logger.info(f"Deleted template: {template_id}")
            return Either.right(True)
            
        except Exception as e:
            error_msg = f"Failed to delete template: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def _validate_context(self, template: ContentTemplate, 
                         context: TemplateRenderContext) -> Either[str, bool]:
        """Validate template context variables."""
        errors = []
        
        # Check required variables
        for var in template.variables:
            if var.required and var.name not in context.variables:
                errors.append(f"Required variable '{var.name}' missing")
            
            # Validate variable values
            if var.name in context.variables:
                value = context.variables[var.name]
                if not var.validate_value(value):
                    errors.append(f"Invalid value for variable '{var.name}': {value}")
        
        if errors:
            return Either.left(f"Context validation failed: {'; '.join(errors)}")
        
        return Either.right(True)
    
    async def _auto_populate_variables(self, template: ContentTemplate, 
                                     context: TemplateRenderContext) -> None:
        """Auto-populate template variables with defaults."""
        for var in template.variables:
            if var.name not in context.variables and var.default_value is not None:
                context.variables[var.name] = var.default_value
    
    def _render_template_content(self, template: ContentTemplate, 
                               context: TemplateRenderContext) -> str:
        """Render template content with context variables."""
        content = template.content_structure
        
        # Simple variable substitution ({{variable}})
        for var_name, var_value in context.variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            
            if isinstance(var_value, str):
                content = content.replace(placeholder, var_value)
            elif isinstance(var_value, list):
                # Handle array variables (simplified)
                array_content = ""
                for item in var_value:
                    if isinstance(item, str):
                        array_content += f"- {item}\n"
                    elif isinstance(item, dict):
                        # Handle object arrays
                        for key, value in item.items():
                            array_content += f"- **{key}**: {value}\n"
                content = content.replace(placeholder, array_content)
            else:
                content = content.replace(placeholder, str(var_value))
        
        # Handle conditional sections ({{#variable}} content {{/variable}})
        content = self._render_conditional_sections(content, context)
        
        return content
    
    def _render_conditional_sections(self, content: str, 
                                   context: TemplateRenderContext) -> str:
        """Render conditional sections in template."""
        # Simple conditional rendering
        # {{#variable}} content {{/variable}}
        pattern = r'\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}'
        
        def replace_conditional(match):
            var_name = match.group(1)
            section_content = match.group(2)
            
            if var_name in context.variables:
                var_value = context.variables[var_name]
                if var_value and (not isinstance(var_value, list) or len(var_value) > 0):
                    return section_content
            
            return ""
        
        return re.sub(pattern, replace_conditional, content, flags=re.DOTALL)
    
    async def _convert_format(self, content: str, output_format: ContentFormat) -> str:
        """Convert content to specified format."""
        if output_format == ContentFormat.HTML:
            # Simple markdown to HTML conversion
            content = content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
            content = content.replace('## ', '<h2>').replace('\n', '</h2>\n', 1)
            content = content.replace('### ', '<h3>').replace('\n', '</h3>\n', 1)
            content = f"<html><body>{content}</body></html>"
        
        return content
    
    def _clear_template_cache(self, template_id: str) -> None:
        """Clear template cache."""
        keys_to_remove = [key for key in self.template_cache.keys() if key.startswith(template_id)]
        for key in keys_to_remove:
            del self.template_cache[key]
    
    async def get_template_analytics(self) -> Dict[str, Any]:
        """Get template usage analytics."""
        try:
            total_templates = len(self.templates)
            builtin_templates = len(self.built_in_templates)
            custom_templates = total_templates - builtin_templates
            
            # Template type distribution
            type_distribution = {}
            for template in self.templates.values():
                type_name = template.template_type.value
                type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
            
            # Tag distribution
            tag_distribution = {}
            for template in self.templates.values():
                for tag in template.tags:
                    tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
            
            return {
                "total_templates": total_templates,
                "builtin_templates": builtin_templates,
                "custom_templates": custom_templates,
                "type_distribution": type_distribution,
                "tag_distribution": tag_distribution,
                "cache_size": len(self.template_cache)
            }
            
        except Exception as e:
            logger.error(f"Failed to get template analytics: {str(e)}")
            return {}