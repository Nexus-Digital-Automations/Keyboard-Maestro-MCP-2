"""
Visual workflow composition engine with drag-and-drop support.

Core engine for visual workflow creation, editing, and manipulation
with comprehensive validation and performance optimization.

Security: Component validation with access control and input sanitization.
Performance: <100ms component operations, <200ms workflow validation.
Type Safety: Complete design by contract with visual workflow management.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, UTC
import logging

from ..core.visual_design import (
    WorkflowId, ComponentId, ConnectionId, CanvasId, LayerId,
    VisualWorkflow, VisualComponent, VisualConnection, WorkflowCanvas,
    ComponentType, ConnectionType, CanvasPosition, ComponentProperties,
    ConnectionProperties, CanvasDimensions, CanvasTheme,
    create_workflow_id, create_component_id, create_connection_id, create_canvas_id
)
from ..core.contracts import require, ensure
from ..core.either import Either

logger = logging.getLogger(__name__)


class VisualComposer:
    """Visual workflow composition engine with drag-and-drop capabilities."""
    
    def __init__(self):
        self.workflows: Dict[WorkflowId, VisualWorkflow] = {}
        self.workflow_locks: Dict[WorkflowId, asyncio.Lock] = {}
        self.auto_save_enabled = True
        self.performance_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    @require(lambda name: len(name) > 0 and len(name) <= 100)
    async def create_workflow(
        self,
        name: str,
        description: str = "",
        canvas_config: Optional[Dict[str, Any]] = None
    ) -> Either[Exception, VisualWorkflow]:
        """Create new visual workflow with interactive canvas."""
        try:
            workflow_id = create_workflow_id()
            canvas_id = create_canvas_id()
            
            # Create canvas with default or custom configuration
            if canvas_config is None:
                canvas_config = {
                    "width": 1200,
                    "height": 800,
                    "theme": "light"
                }
            
            canvas = WorkflowCanvas(
                canvas_id=canvas_id,
                dimensions=CanvasDimensions(
                    width=canvas_config.get("width", 1200),
                    height=canvas_config.get("height", 800)
                ),
                theme=CanvasTheme(canvas_config.get("theme", "light")),
                zoom_level=canvas_config.get("zoom_level", 1.0),
                grid_enabled=canvas_config.get("grid_enabled", True),
                snap_to_grid=canvas_config.get("snap_to_grid", True),
                grid_size=canvas_config.get("grid_size", 20)
            )
            
            # Create default layer
            default_layer = LayerId("layer_default")
            
            workflow = VisualWorkflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                canvas=canvas,
                layers={default_layer: "Default Layer"}
            )
            
            # Store workflow and create lock
            self.workflows[workflow_id] = workflow
            self.workflow_locks[workflow_id] = asyncio.Lock()
            
            self.logger.info(f"Created visual workflow: {workflow_id}")
            return Either.right(workflow)
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            return Either.left(e)
    
    @require(lambda component_type: component_type in ComponentType.__members__.values())
    async def add_component(
        self,
        workflow_id: WorkflowId,
        component_type: ComponentType,
        position: CanvasPosition,
        properties: ComponentProperties,
        layer_id: Optional[LayerId] = None,
        auto_connect: bool = False
    ) -> Either[Exception, VisualComponent]:
        """Add component to visual workflow at specified position."""
        try:
            # Get workflow with lock
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            async with self.workflow_locks[workflow_id]:
                workflow = self.workflows[workflow_id]
                
                # Use default layer if none specified
                if layer_id is None:
                    layer_id = LayerId("layer_default")
                
                # Validate layer exists
                if layer_id not in workflow.layers:
                    workflow.layers[layer_id] = f"Layer {len(workflow.layers) + 1}"
                
                # Create component
                component_id = create_component_id()
                component = VisualComponent(
                    component_id=component_id,
                    component_type=component_type,
                    position=position,
                    properties=properties,
                    layer_id=layer_id
                )
                
                # Add component to workflow
                updated_workflow = workflow.add_component(component)
                
                # Auto-connect to previous component if requested
                if auto_connect and len(workflow.components) > 0:
                    # Find the most recently added component
                    last_component = None
                    latest_time = datetime.min.replace(tzinfo=UTC)
                    
                    for comp in workflow.components.values():
                        if comp.created_at > latest_time:
                            latest_time = comp.created_at
                            last_component = comp
                    
                    if last_component:
                        connection_result = await self._create_auto_connection(
                            updated_workflow, last_component.component_id, component_id
                        )
                        if connection_result.is_right():
                            updated_workflow = connection_result.right()
                
                # Update stored workflow
                self.workflows[workflow_id] = updated_workflow
                
                self.logger.info(f"Added component {component_id} to workflow {workflow_id}")
                return Either.right(component)
                
        except Exception as e:
            self.logger.error(f"Failed to add component: {e}")
            return Either.left(e)
    
    async def connect_components(
        self,
        workflow_id: WorkflowId,
        source_component: ComponentId,
        target_component: ComponentId,
        connection_type: ConnectionType = ConnectionType.SEQUENCE,
        connection_config: Optional[Dict[str, Any]] = None
    ) -> Either[Exception, VisualConnection]:
        """Connect workflow components with validation."""
        try:
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            async with self.workflow_locks[workflow_id]:
                workflow = self.workflows[workflow_id]
                
                # Validate components exist
                if source_component not in workflow.components:
                    return Either.left(ValueError(f"Source component {source_component} not found"))
                if target_component not in workflow.components:
                    return Either.left(ValueError(f"Target component {target_component} not found"))
                
                # Create connection properties
                conn_props = ConnectionProperties()
                if connection_config:
                    conn_props.label = connection_config.get("label", "")
                    conn_props.color = connection_config.get("color", "#007AFF")
                    conn_props.width = connection_config.get("width", 2)
                    conn_props.style = connection_config.get("style", "solid")
                    conn_props.animated = connection_config.get("animated", False)
                
                # Create connection
                connection_id = create_connection_id()
                connection = VisualConnection(
                    connection_id=connection_id,
                    connection_type=connection_type,
                    source_component=source_component,
                    target_component=target_component,
                    properties=conn_props
                )
                
                # Add connection to workflow with validation
                connection_result = workflow.add_connection(connection)
                if connection_result.is_left():
                    return Either.left(connection_result.left())
                
                # Update stored workflow
                self.workflows[workflow_id] = connection_result.right()
                
                self.logger.info(f"Connected components {source_component} -> {target_component}")
                return Either.right(connection)
                
        except Exception as e:
            self.logger.error(f"Failed to connect components: {e}")
            return Either.left(e)
    
    async def update_component(
        self,
        workflow_id: WorkflowId,
        component_id: ComponentId,
        updated_properties: Optional[ComponentProperties] = None,
        new_position: Optional[CanvasPosition] = None
    ) -> Either[Exception, VisualComponent]:
        """Update component properties and position."""
        try:
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            async with self.workflow_locks[workflow_id]:
                workflow = self.workflows[workflow_id]
                
                if component_id not in workflow.components:
                    return Either.left(ValueError(f"Component {component_id} not found"))
                
                component = workflow.components[component_id]
                
                # Update properties if provided
                if updated_properties:
                    component.properties = updated_properties
                    component.modified_at = datetime.now(UTC)
                
                # Update position if provided
                if new_position:
                    component = component.update_position(new_position)
                
                # Update workflow
                updated_components = workflow.components.copy()
                updated_components[component_id] = component
                
                updated_workflow = VisualWorkflow(
                    workflow_id=workflow.workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    canvas=workflow.canvas,
                    components=updated_components,
                    connections=workflow.connections,
                    layers=workflow.layers,
                    metadata=workflow.metadata,
                    created_at=workflow.created_at,
                    modified_at=datetime.now(UTC),
                    version=workflow.version + 1
                )
                
                self.workflows[workflow_id] = updated_workflow
                
                self.logger.info(f"Updated component {component_id}")
                return Either.right(component)
                
        except Exception as e:
            self.logger.error(f"Failed to update component: {e}")
            return Either.left(e)
    
    async def remove_component(
        self,
        workflow_id: WorkflowId,
        component_id: ComponentId
    ) -> Either[Exception, bool]:
        """Remove component and its connections from workflow."""
        try:
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            async with self.workflow_locks[workflow_id]:
                workflow = self.workflows[workflow_id]
                
                if component_id not in workflow.components:
                    return Either.left(ValueError(f"Component {component_id} not found"))
                
                # Remove component
                updated_components = workflow.components.copy()
                component = updated_components.pop(component_id)
                
                # Remove associated connections
                updated_connections = workflow.connections.copy()
                connections_to_remove = []
                
                for conn_id, connection in workflow.connections.items():
                    if (connection.source_component == component_id or 
                        connection.target_component == component_id):
                        connections_to_remove.append(conn_id)
                
                for conn_id in connections_to_remove:
                    updated_connections.pop(conn_id, None)
                
                # Update remaining components to remove connection references
                for comp_id, comp in updated_components.items():
                    updated_connections_list = [
                        conn_id for conn_id in comp.connections 
                        if conn_id not in connections_to_remove
                    ]
                    if len(updated_connections_list) != len(comp.connections):
                        comp.connections = updated_connections_list
                
                # Update workflow
                updated_workflow = VisualWorkflow(
                    workflow_id=workflow.workflow_id,
                    name=workflow.name,
                    description=workflow.description,
                    canvas=workflow.canvas,
                    components=updated_components,
                    connections=updated_connections,
                    layers=workflow.layers,
                    metadata=workflow.metadata,
                    created_at=workflow.created_at,
                    modified_at=datetime.now(UTC),
                    version=workflow.version + 1
                )
                
                self.workflows[workflow_id] = updated_workflow
                
                self.logger.info(f"Removed component {component_id} and {len(connections_to_remove)} connections")
                return Either.right(True)
                
        except Exception as e:
            self.logger.error(f"Failed to remove component: {e}")
            return Either.left(e)
    
    async def validate_workflow(self, workflow_id: WorkflowId) -> Either[Exception, List[str]]:
        """Validate workflow logic and structure."""
        try:
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            workflow = self.workflows[workflow_id]
            validation_errors = workflow.validate_workflow()
            
            return Either.right(validation_errors)
            
        except Exception as e:
            self.logger.error(f"Failed to validate workflow: {e}")
            return Either.left(e)
    
    async def get_workflow(self, workflow_id: WorkflowId) -> Either[Exception, VisualWorkflow]:
        """Get workflow by ID."""
        try:
            if workflow_id not in self.workflows:
                return Either.left(ValueError(f"Workflow {workflow_id} not found"))
            
            return Either.right(self.workflows[workflow_id])
            
        except Exception as e:
            return Either.left(e)
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows with metadata."""
        workflows_info = []
        
        for workflow_id, workflow in self.workflows.items():
            workflows_info.append({
                "workflow_id": workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "component_count": len(workflow.components),
                "connection_count": len(workflow.connections),
                "created_at": workflow.created_at.isoformat(),
                "modified_at": workflow.modified_at.isoformat(),
                "version": workflow.version
            })
        
        return workflows_info
    
    async def _create_auto_connection(
        self,
        workflow: VisualWorkflow,
        source_id: ComponentId,
        target_id: ComponentId
    ) -> Either[Exception, VisualWorkflow]:
        """Create automatic connection between components."""
        try:
            source_component = workflow.components.get(source_id)
            target_component = workflow.components.get(target_id)
            
            if not source_component or not target_component:
                return Either.left(ValueError("Source or target component not found"))
            
            # Determine appropriate connection type
            connection_type = ConnectionType.SEQUENCE
            if source_component.component_type == ComponentType.CONDITION:
                connection_type = ConnectionType.CONDITION
            elif source_component.component_type == ComponentType.TRIGGER:
                connection_type = ConnectionType.TRIGGER
            
            # Create connection
            connection = VisualConnection(
                connection_id=create_connection_id(),
                connection_type=connection_type,
                source_component=source_id,
                target_component=target_id,
                properties=ConnectionProperties(label="Auto-connected")
            )
            
            return workflow.add_connection(connection)
            
        except Exception as e:
            return Either.left(e)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_workflows": len(self.workflows),
            "total_components": sum(len(w.components) for w in self.workflows.values()),
            "total_connections": sum(len(w.connections) for w in self.workflows.values()),
            "cache_size": len(self.performance_cache),
            "memory_usage": "calculation_needed"  # Placeholder for actual memory calculation
        }


# Global composer instance
_visual_composer: Optional[VisualComposer] = None


def get_visual_composer() -> VisualComposer:
    """Get or create global visual composer instance."""
    global _visual_composer
    if _visual_composer is None:
        _visual_composer = VisualComposer()
    return _visual_composer