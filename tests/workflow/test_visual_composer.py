"""
Test suite for visual workflow composer.

Comprehensive testing for visual workflow creation, editing, and validation
with property-based testing and contract verification.

Security: Test input validation and access control mechanisms.
Performance: Validate sub-second response times for all operations.
Type Safety: Verify complete contract compliance and type validation.
"""

import pytest
import asyncio
from typing import Dict, List, Any
from datetime import datetime, UTC
from hypothesis import given, strategies as st

from src.workflow.visual_composer import VisualComposer, get_visual_composer
from src.core.visual_design import (
    WorkflowId, ComponentId, ConnectionId,
    ComponentType, ConnectionType, CanvasPosition, ComponentProperties,
    ConnectionProperties, CanvasDimensions, CanvasTheme
)


class TestVisualComposer:
    """Test visual workflow composer functionality."""
    
    @pytest.fixture
    def composer(self):
        """Create fresh composer instance for each test."""
        return VisualComposer()
    
    @pytest.mark.asyncio
    async def test_create_workflow_basic(self, composer):
        """Test basic workflow creation."""
        result = await composer.create_workflow(
            name="Test Workflow",
            description="Test workflow description"
        )
        
        assert result.is_right()
        workflow = result.right()
        
        assert workflow.name == "Test Workflow"
        assert workflow.description == "Test workflow description"
        assert len(workflow.components) == 0
        assert len(workflow.connections) == 0
        assert workflow.version == 1
    
    @pytest.mark.asyncio
    async def test_create_workflow_with_canvas_config(self, composer):
        """Test workflow creation with custom canvas configuration."""
        canvas_config = {
            "width": 1600,
            "height": 1000,
            "theme": "dark",
            "zoom_level": 1.5,
            "grid_enabled": False
        }
        
        result = await composer.create_workflow(
            name="Custom Canvas Workflow",
            canvas_config=canvas_config
        )
        
        assert result.is_right()
        workflow = result.right()
        
        assert workflow.canvas.dimensions.width == 1600
        assert workflow.canvas.dimensions.height == 1000
        assert workflow.canvas.theme == CanvasTheme.DARK
        assert workflow.canvas.zoom_level == 1.5
        assert workflow.canvas.grid_enabled == False
    
    @pytest.mark.asyncio
    async def test_add_component_basic(self, composer):
        """Test adding component to workflow."""
        # Create workflow first
        workflow_result = await composer.create_workflow("Test Workflow")
        assert workflow_result.is_right()
        workflow = workflow_result.right()
        
        # Add component
        position = CanvasPosition(x=100, y=200)
        properties = ComponentProperties(title="Test Action", description="Test description")
        
        result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=position,
            properties=properties
        )
        
        assert result.is_right()
        component = result.right()
        
        assert component.component_type == ComponentType.ACTION
        assert component.position.x == 100
        assert component.position.y == 200
        assert component.properties.title == "Test Action"
        
        # Verify component was added to workflow
        updated_workflow = composer.workflows[workflow.workflow_id]
        assert len(updated_workflow.components) == 1
        assert component.component_id in updated_workflow.components
    
    @pytest.mark.asyncio
    async def test_add_component_auto_connect(self, composer):
        """Test auto-connecting components."""
        # Create workflow and add first component
        workflow_result = await composer.create_workflow("Auto Connect Test")
        workflow = workflow_result.right()
        
        # Add first component
        await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.TRIGGER,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Trigger")
        )
        
        # Add second component with auto-connect
        result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=300, y=100),
            properties=ComponentProperties(title="Action"),
            auto_connect=True
        )
        
        assert result.is_right()
        
        # Verify connection was created
        updated_workflow = composer.workflows[workflow.workflow_id]
        assert len(updated_workflow.components) == 2
        assert len(updated_workflow.connections) == 1
    
    @pytest.mark.asyncio
    async def test_connect_components(self, composer):
        """Test connecting workflow components."""
        # Create workflow and components
        workflow_result = await composer.create_workflow("Connect Test")
        workflow = workflow_result.right()
        
        # Add two components
        comp1_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.TRIGGER,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Trigger")
        )
        comp1 = comp1_result.right()
        
        comp2_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=300, y=100),
            properties=ComponentProperties(title="Action")
        )
        comp2 = comp2_result.right()
        
        # Connect components
        connection_result = await composer.connect_components(
            workflow_id=workflow.workflow_id,
            source_component=comp1.component_id,
            target_component=comp2.component_id,
            connection_type=ConnectionType.TRIGGER
        )
        
        assert connection_result.is_right()
        connection = connection_result.right()
        
        assert connection.source_component == comp1.component_id
        assert connection.target_component == comp2.component_id
        assert connection.connection_type == ConnectionType.TRIGGER
        
        # Verify connection was added
        updated_workflow = composer.workflows[workflow.workflow_id]
        assert len(updated_workflow.connections) == 1
    
    @pytest.mark.asyncio
    async def test_update_component(self, composer):
        """Test updating component properties and position."""
        # Create workflow and component
        workflow_result = await composer.create_workflow("Update Test")
        workflow = workflow_result.right()
        
        comp_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Original Title")
        )
        component = comp_result.right()
        
        # Update component
        new_properties = ComponentProperties(
            title="Updated Title",
            description="Updated description",
            properties={"new_prop": "value"}
        )
        new_position = CanvasPosition(x=200, y=200)
        
        update_result = await composer.update_component(
            workflow_id=workflow.workflow_id,
            component_id=component.component_id,
            updated_properties=new_properties,
            new_position=new_position
        )
        
        assert update_result.is_right()
        updated_component = update_result.right()
        
        assert updated_component.properties.title == "Updated Title"
        assert updated_component.properties.description == "Updated description"
        assert updated_component.position.x == 200
        assert updated_component.position.y == 200
        assert updated_component.properties.properties["new_prop"] == "value"
    
    @pytest.mark.asyncio
    async def test_remove_component(self, composer):
        """Test removing component and its connections."""
        # Create workflow with connected components
        workflow_result = await composer.create_workflow("Remove Test")
        workflow = workflow_result.right()
        
        # Add components
        comp1_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.TRIGGER,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Trigger")
        )
        comp1 = comp1_result.right()
        
        comp2_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=300, y=100),
            properties=ComponentProperties(title="Action")
        )
        comp2 = comp2_result.right()
        
        # Connect components
        await composer.connect_components(
            workflow_id=workflow.workflow_id,
            source_component=comp1.component_id,
            target_component=comp2.component_id
        )
        
        # Verify initial state
        initial_workflow = composer.workflows[workflow.workflow_id]
        assert len(initial_workflow.components) == 2
        assert len(initial_workflow.connections) == 1
        
        # Remove component
        remove_result = await composer.remove_component(
            workflow_id=workflow.workflow_id,
            component_id=comp1.component_id
        )
        
        assert remove_result.is_right()
        assert remove_result.right() == True
        
        # Verify component and connections removed
        updated_workflow = composer.workflows[workflow.workflow_id]
        assert len(updated_workflow.components) == 1
        assert len(updated_workflow.connections) == 0
        assert comp1.component_id not in updated_workflow.components
        assert comp2.component_id in updated_workflow.components
    
    @pytest.mark.asyncio
    async def test_validate_workflow(self, composer):
        """Test workflow validation."""
        # Create valid workflow
        workflow_result = await composer.create_workflow("Validation Test")
        workflow = workflow_result.right()
        
        # Add connected components
        comp1_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.TRIGGER,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Trigger")
        )
        comp1 = comp1_result.right()
        
        comp2_result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=300, y=100),
            properties=ComponentProperties(title="Action")
        )
        comp2 = comp2_result.right()
        
        await composer.connect_components(
            workflow_id=workflow.workflow_id,
            source_component=comp1.component_id,
            target_component=comp2.component_id
        )
        
        # Validate workflow
        validation_result = await composer.validate_workflow(workflow.workflow_id)
        assert validation_result.is_right()
        
        errors = validation_result.right()
        assert len(errors) == 0  # Should be valid
    
    @pytest.mark.asyncio
    async def test_validate_workflow_with_errors(self, composer):
        """Test workflow validation with errors."""
        # Create workflow with orphaned component
        workflow_result = await composer.create_workflow("Error Test")
        workflow = workflow_result.right()
        
        # Add orphaned action component (no connections)
        await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Orphaned Action")
        )
        
        # Validate workflow
        validation_result = await composer.validate_workflow(workflow.workflow_id)
        assert validation_result.is_right()
        
        errors = validation_result.right()
        assert len(errors) > 0  # Should have validation errors
        assert any("no connections" in error.lower() for error in errors)
    
    @pytest.mark.asyncio
    async def test_workflow_not_found_errors(self, composer):
        """Test error handling for non-existent workflows."""
        fake_workflow_id = WorkflowId("nonexistent_workflow")
        
        # Test various operations with non-existent workflow
        add_result = await composer.add_component(
            workflow_id=fake_workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=0, y=0),
            properties=ComponentProperties(title="Test")
        )
        assert add_result.is_left()
        
        validate_result = await composer.validate_workflow(fake_workflow_id)
        assert validate_result.is_left()
        
        get_result = await composer.get_workflow(fake_workflow_id)
        assert get_result.is_left()
    
    def test_get_performance_stats(self, composer):
        """Test performance statistics retrieval."""
        stats = composer.get_performance_stats()
        
        assert "total_workflows" in stats
        assert "total_components" in stats
        assert "total_connections" in stats
        assert "cache_size" in stats
        assert isinstance(stats["total_workflows"], int)
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, composer):
        """Test listing all workflows."""
        # Create multiple workflows
        await composer.create_workflow("Workflow 1", "First workflow")
        await composer.create_workflow("Workflow 2", "Second workflow")
        
        workflows_list = await composer.list_workflows()
        
        assert len(workflows_list) == 2
        assert all("workflow_id" in w for w in workflows_list)
        assert all("name" in w for w in workflows_list)
        assert all("component_count" in w for w in workflows_list)
        
        names = [w["name"] for w in workflows_list]
        assert "Workflow 1" in names
        assert "Workflow 2" in names


class TestVisualComposerPropertyBased:
    """Property-based tests for visual composer."""
    
    @given(
        name=st.text(min_size=1, max_size=100),
        description=st.text(max_size=500)
    )
    @pytest.mark.asyncio
    async def test_create_workflow_with_valid_inputs(self, name, description):
        """Property: Valid inputs should always create successful workflows."""
        composer = VisualComposer()
        
        result = await composer.create_workflow(name=name, description=description)
        
        assert result.is_right()
        workflow = result.right()
        assert workflow.name == name
        assert workflow.description == description
    
    @given(
        x=st.integers(min_value=0, max_value=2000),
        y=st.integers(min_value=0, max_value=2000),
        title=st.text(min_size=1, max_size=100)
    )
    @pytest.mark.asyncio
    async def test_add_component_position_bounds(self, x, y, title):
        """Property: Components should be addable at any valid position."""
        composer = VisualComposer()
        
        # Create workflow
        workflow_result = await composer.create_workflow("Property Test")
        workflow = workflow_result.right()
        
        # Add component at position
        position = CanvasPosition(x=x, y=y)
        properties = ComponentProperties(title=title)
        
        result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=position,
            properties=properties
        )
        
        assert result.is_right()
        component = result.right()
        assert component.position.x == x
        assert component.position.y == y
    
    @given(
        component_count=st.integers(min_value=0, max_value=10)
    )
    @pytest.mark.asyncio
    async def test_workflow_component_count_invariant(self, component_count):
        """Property: Workflow component count should match actual components."""
        composer = VisualComposer()
        
        # Create workflow
        workflow_result = await composer.create_workflow("Count Test")
        workflow = workflow_result.right()
        
        # Add specified number of components
        for i in range(component_count):
            await composer.add_component(
                workflow_id=workflow.workflow_id,
                component_type=ComponentType.ACTION,
                position=CanvasPosition(x=i * 100, y=100),
                properties=ComponentProperties(title=f"Component {i}")
            )
        
        # Verify count invariant
        updated_workflow = composer.workflows[workflow.workflow_id]
        assert len(updated_workflow.components) == component_count
    
    @given(
        canvas_width=st.integers(min_value=800, max_value=4000),
        canvas_height=st.integers(min_value=600, max_value=4000)
    )
    @pytest.mark.asyncio
    async def test_canvas_dimensions_validation(self, canvas_width, canvas_height):
        """Property: Canvas dimensions should be validated and stored correctly."""
        composer = VisualComposer()
        
        canvas_config = {
            "width": canvas_width,
            "height": canvas_height,
            "theme": "light"
        }
        
        result = await composer.create_workflow(
            name="Canvas Test",
            canvas_config=canvas_config
        )
        
        assert result.is_right()
        workflow = result.right()
        assert workflow.canvas.dimensions.width == canvas_width
        assert workflow.canvas.dimensions.height == canvas_height


def test_global_composer_singleton():
    """Test global composer singleton pattern."""
    composer1 = get_visual_composer()
    composer2 = get_visual_composer()
    
    assert composer1 is composer2
    assert isinstance(composer1, VisualComposer)


@pytest.mark.performance
class TestVisualComposerPerformance:
    """Performance tests for visual composer operations."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation_performance(self):
        """Test workflow creation performance."""
        composer = VisualComposer()
        
        start_time = datetime.now(UTC)
        
        result = await composer.create_workflow("Performance Test")
        
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        
        assert result.is_right()
        assert duration < 0.1  # Should complete within 100ms
    
    @pytest.mark.asyncio
    async def test_component_addition_performance(self):
        """Test component addition performance."""
        composer = VisualComposer()
        
        # Create workflow
        workflow_result = await composer.create_workflow("Performance Test")
        workflow = workflow_result.right()
        
        start_time = datetime.now(UTC)
        
        # Add component
        result = await composer.add_component(
            workflow_id=workflow.workflow_id,
            component_type=ComponentType.ACTION,
            position=CanvasPosition(x=100, y=100),
            properties=ComponentProperties(title="Performance Test")
        )
        
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        
        assert result.is_right()
        assert duration < 0.1  # Should complete within 100ms
    
    @pytest.mark.asyncio
    async def test_large_workflow_validation_performance(self):
        """Test validation performance with large workflows."""
        composer = VisualComposer()
        
        # Create workflow with many components
        workflow_result = await composer.create_workflow("Large Workflow Test")
        workflow = workflow_result.right()
        
        # Add 20 components
        components = []
        for i in range(20):
            comp_result = await composer.add_component(
                workflow_id=workflow.workflow_id,
                component_type=ComponentType.ACTION,
                position=CanvasPosition(x=i * 50, y=100),
                properties=ComponentProperties(title=f"Component {i}")
            )
            components.append(comp_result.right())
        
        # Connect components sequentially
        for i in range(len(components) - 1):
            await composer.connect_components(
                workflow_id=workflow.workflow_id,
                source_component=components[i].component_id,
                target_component=components[i + 1].component_id
            )
        
        start_time = datetime.now(UTC)
        
        # Validate large workflow
        validation_result = await composer.validate_workflow(workflow.workflow_id)
        
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        
        assert validation_result.is_right()
        assert duration < 0.2  # Should complete within 200ms even for large workflows