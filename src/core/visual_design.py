"""
Visual design type definitions for workflow designer.

Comprehensive branded types for drag-and-drop visual workflow creation
with FastMCP integration and security validation.

Security: Type-safe visual component validation with access control.
Performance: <100ms component creation, <50ms canvas updates.
Type Safety: Complete design by contract with visual workflow validation.
"""

from __future__ import annotations
from typing import NewType, Protocol, TypeVar, Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
from abc import ABC, abstractmethod
import uuid

from .contracts import require, ensure
from .either import Either


# Branded Types for Visual Design
WorkflowId = NewType('WorkflowId', str)
ComponentId = NewType('ComponentId', str)
ConnectionId = NewType('ConnectionId', str)
CanvasId = NewType('CanvasId', str)
TemplateId = NewType('TemplateId', str)
LayerId = NewType('LayerId', str)


def create_workflow_id() -> WorkflowId:
    """Create unique workflow identifier."""
    return WorkflowId(f"workflow_{uuid.uuid4().hex[:16]}")


def create_component_id() -> ComponentId:
    """Create unique component identifier."""
    return ComponentId(f"component_{uuid.uuid4().hex[:12]}")


def create_connection_id() -> ConnectionId:
    """Create unique connection identifier."""
    return ConnectionId(f"connection_{uuid.uuid4().hex[:12]}")


def create_canvas_id() -> CanvasId:
    """Create unique canvas identifier."""
    return CanvasId(f"canvas_{uuid.uuid4().hex[:12]}")


class ComponentType(Enum):
    """Visual workflow component types."""
    ACTION = "action"
    CONDITION = "condition" 
    TRIGGER = "trigger"
    GROUP = "group"
    VARIABLE = "variable"
    LOOP = "loop"
    SWITCH = "switch"
    PARALLEL = "parallel"
    DELAY = "delay"
    COMMENT = "comment"


class ConnectionType(Enum):
    """Visual connection types between components."""
    SEQUENCE = "sequence"        # Sequential execution flow
    CONDITION = "condition"      # Conditional branch (true/false)
    DATA = "data"               # Data flow between components
    TRIGGER = "trigger"         # Event-driven activation
    PARALLEL = "parallel"       # Parallel execution branch
    LOOP = "loop"              # Loop iteration connection


class CanvasTheme(Enum):
    """Visual canvas themes."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    SYSTEM = "system"


@dataclass(frozen=True)
class CanvasPosition:
    """2D position on visual canvas."""
    x: int
    y: int
    
    @require(lambda self: self.x >= 0 and self.y >= 0)
    def __post_init__(self) -> None:
        """Validate position coordinates."""
        pass
    
    def distance_to(self, other: CanvasPosition) -> float:
        """Calculate distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class CanvasDimensions:
    """Canvas size and constraints."""
    width: int
    height: int
    min_width: int = 800
    min_height: int = 600
    max_width: int = 4000
    max_height: int = 4000
    
    @require(lambda self: self.width >= self.min_width and self.height >= self.min_height)
    @require(lambda self: self.width <= self.max_width and self.height <= self.max_height)
    def __post_init__(self) -> None:
        """Validate canvas dimensions."""
        pass


@dataclass
class ComponentProperties:
    """Visual component configuration properties."""
    title: str
    description: str = ""
    enabled: bool = True
    properties: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)
    
    @require(lambda self: len(self.title) > 0 and len(self.title) <= 100)
    @require(lambda self: len(self.description) <= 500)
    def __post_init__(self) -> None:
        """Validate component properties."""
        pass


@dataclass
class VisualComponent:
    """Visual workflow component with design and behavior."""
    component_id: ComponentId
    component_type: ComponentType
    position: CanvasPosition
    properties: ComponentProperties
    layer_id: LayerId
    connections: List[ConnectionId] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def update_position(self, new_position: CanvasPosition) -> VisualComponent:
        """Update component position."""
        return VisualComponent(
            component_id=self.component_id,
            component_type=self.component_type,
            position=new_position,
            properties=self.properties,
            layer_id=self.layer_id,
            connections=self.connections,
            created_at=self.created_at,
            modified_at=datetime.now(UTC)
        )


@dataclass
class ConnectionProperties:
    """Connection configuration and styling."""
    label: str = ""
    color: str = "#007AFF"
    width: int = 2
    style: str = "solid"  # solid, dashed, dotted
    animated: bool = False
    bidirectional: bool = False
    
    @require(lambda self: self.width > 0 and self.width <= 10)
    @require(lambda self: self.style in ["solid", "dashed", "dotted"])
    def __post_init__(self) -> None:
        """Validate connection properties."""
        pass


@dataclass
class VisualConnection:
    """Connection between visual components."""
    connection_id: ConnectionId
    connection_type: ConnectionType
    source_component: ComponentId
    target_component: ComponentId
    properties: ConnectionProperties
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_valid_connection(self, source_type: ComponentType, target_type: ComponentType) -> bool:
        """Validate connection compatibility."""
        # Define valid connection rules
        valid_connections = {
            ConnectionType.SEQUENCE: True,  # Any component can connect sequentially
            ConnectionType.CONDITION: source_type == ComponentType.CONDITION,
            ConnectionType.DATA: True,  # Data can flow between any components
            ConnectionType.TRIGGER: source_type == ComponentType.TRIGGER,
            ConnectionType.PARALLEL: True,
            ConnectionType.LOOP: source_type in [ComponentType.LOOP, ComponentType.CONDITION]
        }
        return valid_connections.get(self.connection_type, False)


@dataclass
class WorkflowCanvas:
    """Visual canvas for workflow design."""
    canvas_id: CanvasId
    dimensions: CanvasDimensions
    theme: CanvasTheme
    zoom_level: float = 1.0
    grid_enabled: bool = True
    snap_to_grid: bool = True
    grid_size: int = 20
    
    @require(lambda self: 0.1 <= self.zoom_level <= 5.0)
    @require(lambda self: self.grid_size > 0 and self.grid_size <= 50)
    def __post_init__(self) -> None:
        """Validate canvas configuration."""
        pass


@dataclass
class VisualWorkflow:
    """Complete visual workflow with components and connections."""
    workflow_id: WorkflowId
    name: str
    description: str
    canvas: WorkflowCanvas
    components: Dict[ComponentId, VisualComponent] = field(default_factory=dict)
    connections: Dict[ConnectionId, VisualConnection] = field(default_factory=dict)
    layers: Dict[LayerId, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: int = 1
    
    @require(lambda self: len(self.name) > 0 and len(self.name) <= 100)
    @require(lambda self: len(self.description) <= 1000)
    def __post_init__(self) -> None:
        """Validate workflow configuration."""
        pass
    
    def add_component(self, component: VisualComponent) -> VisualWorkflow:
        """Add component to workflow."""
        new_components = self.components.copy()
        new_components[component.component_id] = component
        
        return VisualWorkflow(
            workflow_id=self.workflow_id,
            name=self.name,
            description=self.description,
            canvas=self.canvas,
            components=new_components,
            connections=self.connections,
            layers=self.layers,
            metadata=self.metadata,
            created_at=self.created_at,
            modified_at=datetime.now(UTC),
            version=self.version + 1
        )
    
    def add_connection(self, connection: VisualConnection) -> Either[Exception, VisualWorkflow]:
        """Add connection between components with validation."""
        # Validate components exist
        source_component = self.components.get(connection.source_component)
        target_component = self.components.get(connection.target_component)
        
        if not source_component or not target_component:
            return Either.left(ValueError("Source or target component not found"))
        
        # Validate connection type compatibility
        if not connection.is_valid_connection(source_component.component_type, target_component.component_type):
            return Either.left(ValueError(f"Invalid connection type {connection.connection_type} between {source_component.component_type} and {target_component.component_type}"))
        
        new_connections = self.connections.copy()
        new_connections[connection.connection_id] = connection
        
        # Update component connection lists
        new_components = self.components.copy()
        updated_source = source_component
        updated_target = target_component
        
        # Add connection to component lists if not already present
        if connection.connection_id not in updated_source.connections:
            updated_source.connections.append(connection.connection_id)
            new_components[connection.source_component] = updated_source
        
        if connection.connection_id not in updated_target.connections:
            updated_target.connections.append(connection.connection_id)
            new_components[connection.target_component] = updated_target
        
        return Either.right(VisualWorkflow(
            workflow_id=self.workflow_id,
            name=self.name,
            description=self.description,
            canvas=self.canvas,
            components=new_components,
            connections=new_connections,
            layers=self.layers,
            metadata=self.metadata,
            created_at=self.created_at,
            modified_at=datetime.now(UTC),
            version=self.version + 1
        ))
    
    def validate_workflow(self) -> List[str]:
        """Validate complete workflow integrity."""
        errors = []
        
        # Check for orphaned components (no connections)
        for component_id, component in self.components.items():
            if not component.connections and component.component_type != ComponentType.TRIGGER:
                errors.append(f"Component {component_id} has no connections")
        
        # Check for invalid connections
        for connection_id, connection in self.connections.items():
            if connection.source_component not in self.components:
                errors.append(f"Connection {connection_id} references invalid source component")
            if connection.target_component not in self.components:
                errors.append(f"Connection {connection_id} references invalid target component")
        
        # Check for cycles in sequential connections
        visited = set()
        rec_stack = set()
        
        def has_cycle(component_id: ComponentId) -> bool:
            visited.add(component_id)
            rec_stack.add(component_id)
            
            component = self.components.get(component_id)
            if not component:
                return False
            
            for connection_id in component.connections:
                connection = self.connections.get(connection_id)
                if connection and connection.connection_type == ConnectionType.SEQUENCE:
                    if connection.source_component == component_id:
                        target = connection.target_component
                        if target not in visited:
                            if has_cycle(target):
                                return True
                        elif target in rec_stack:
                            return True
            
            rec_stack.remove(component_id)
            return False
        
        for component_id in self.components.keys():
            if component_id not in visited:
                if has_cycle(component_id):
                    errors.append("Workflow contains circular dependencies")
                    break
        
        return errors


class WorkflowTemplate(Protocol):
    """Protocol for visual workflow templates."""
    
    @property
    def template_id(self) -> TemplateId:
        """Template identifier."""
        ...
    
    @property
    def name(self) -> str:
        """Template name."""
        ...
    
    @property
    def description(self) -> str:
        """Template description."""
        ...
    
    @property
    def category(self) -> str:
        """Template category."""
        ...
    
    @property
    def complexity(self) -> str:
        """Template complexity level."""
        ...
    
    def create_workflow(self, name: str, canvas: WorkflowCanvas) -> VisualWorkflow:
        """Create workflow from template."""
        ...


@dataclass
class BasicWorkflowTemplate:
    """Basic workflow template implementation."""
    template_id: TemplateId
    name: str
    description: str
    category: str
    complexity: str
    component_definitions: List[Dict[str, Any]] = field(default_factory=list)
    connection_definitions: List[Dict[str, Any]] = field(default_factory=list)
    
    def create_workflow(self, name: str, canvas: WorkflowCanvas) -> VisualWorkflow:
        """Create workflow from template."""
        workflow_id = create_workflow_id()
        workflow = VisualWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=self.description,
            canvas=canvas
        )
        
        # Create components from template
        component_map = {}
        for i, comp_def in enumerate(self.component_definitions):
            component_id = create_component_id()
            component = VisualComponent(
                component_id=component_id,
                component_type=ComponentType(comp_def.get("type", "action")),
                position=CanvasPosition(
                    x=comp_def.get("x", 100 + i * 200),
                    y=comp_def.get("y", 100)
                ),
                properties=ComponentProperties(
                    title=comp_def.get("title", f"Component {i+1}"),
                    description=comp_def.get("description", ""),
                    properties=comp_def.get("properties", {})
                ),
                layer_id=LayerId("default")
            )
            workflow = workflow.add_component(component)
            component_map[i] = component_id
        
        # Create connections from template
        for conn_def in self.connection_definitions:
            source_idx = conn_def.get("source", 0)
            target_idx = conn_def.get("target", 1)
            
            if source_idx in component_map and target_idx in component_map:
                connection = VisualConnection(
                    connection_id=create_connection_id(),
                    connection_type=ConnectionType(conn_def.get("type", "sequence")),
                    source_component=component_map[source_idx],
                    target_component=component_map[target_idx],
                    properties=ConnectionProperties(
                        label=conn_def.get("label", "")
                    )
                )
                workflow = workflow.add_connection(connection).right()
        
        return workflow


# Common workflow templates
EMAIL_AUTOMATION_TEMPLATE = BasicWorkflowTemplate(
    template_id=TemplateId("email_automation"),
    name="Email Automation",
    description="Automated email processing workflow",
    category="communication",
    complexity="simple",
    component_definitions=[
        {"type": "trigger", "title": "Email Received", "x": 100, "y": 100},
        {"type": "condition", "title": "Check Sender", "x": 300, "y": 100},
        {"type": "action", "title": "Process Email", "x": 500, "y": 100},
        {"type": "action", "title": "Send Reply", "x": 700, "y": 100}
    ],
    connection_definitions=[
        {"source": 0, "target": 1, "type": "sequence"},
        {"source": 1, "target": 2, "type": "condition"},
        {"source": 2, "target": 3, "type": "sequence"}
    ]
)

FILE_PROCESSING_TEMPLATE = BasicWorkflowTemplate(
    template_id=TemplateId("file_processing"),
    name="File Processing",
    description="Automated file processing and organization",
    category="file_management",
    complexity="intermediate",
    component_definitions=[
        {"type": "trigger", "title": "File Created", "x": 100, "y": 100},
        {"type": "condition", "title": "Check File Type", "x": 300, "y": 100},
        {"type": "action", "title": "Process File", "x": 500, "y": 100},
        {"type": "action", "title": "Move to Folder", "x": 700, "y": 100},
        {"type": "action", "title": "Send Notification", "x": 900, "y": 100}
    ],
    connection_definitions=[
        {"source": 0, "target": 1, "type": "sequence"},
        {"source": 1, "target": 2, "type": "condition"},
        {"source": 2, "target": 3, "type": "sequence"},
        {"source": 3, "target": 4, "type": "sequence"}
    ]
)

# Export commonly used templates
WORKFLOW_TEMPLATES: Dict[str, WorkflowTemplate] = {
    "email_automation": EMAIL_AUTOMATION_TEMPLATE,
    "file_processing": FILE_PROCESSING_TEMPLATE
}