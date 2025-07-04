"""
Test suite for workflow designer MCP tools.

Comprehensive testing for visual workflow designer MCP tools integration
with FastMCP protocol and Claude Desktop interaction validation.

Security: Test input validation and MCP protocol compliance.
Performance: Validate sub-second response times for all MCP operations.
Type Safety: Verify complete FastMCP integration and JSON response validation.
"""

import pytest
import json
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.server.tools.workflow_designer_tools import WorkflowDesignerTools, get_workflow_designer_tools
from src.workflow.visual_composer import VisualComposer
from src.workflow.component_library import ComponentLibrary
from src.core.visual_design import ComponentType, ConnectionType


class TestWorkflowDesignerTools:
    """Test workflow designer MCP tools functionality."""
    
    @pytest.fixture
    def tools(self):
        """Create fresh workflow designer tools instance."""
        return WorkflowDesignerTools()
    
    @pytest.fixture
    def mock_mcp(self):
        """Create mock FastMCP instance."""
        mock = Mock()
        mock.tool = Mock(return_value=lambda func: func)
        return mock
    
    def test_tools_initialization(self, tools):
        """Test tools initialization with dependencies."""
        assert isinstance(tools.visual_composer, VisualComposer)
        assert isinstance(tools.component_library, ComponentLibrary)
        assert tools.logger is not None
    
    def test_register_tools(self, tools, mock_mcp):
        """Test tool registration with FastMCP."""
        tools.register_tools(mock_mcp)
        
        # Verify all tools were registered
        expected_tools = [
            "km_create_visual_workflow",
            "km_add_workflow_component", 
            "km_connect_workflow_nodes",
            "km_edit_workflow_component",
            "km_export_visual_workflow",
            "km_get_workflow_templates",
            "km_validate_workflow"
        ]
        
        assert mock_mcp.tool.call_count == len(expected_tools)
    
    @pytest.mark.asyncio
    async def test_km_create_visual_workflow_basic(self, tools):
        """Test basic visual workflow creation."""
        # Create mock MCP and register tools
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Get the registered function
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        
        # Test workflow creation
        result = await create_workflow_func(
            name="Test Workflow",
            description="Test description",
            canvas_width=1200,
            canvas_height=800,
            theme="light"
        )
        
        # Parse JSON response
        assert result.startswith("```json")
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["workflow"]["name"] == "Test Workflow"
        assert response["workflow"]["description"] == "Test description"
        assert response["workflow"]["canvas"]["width"] == 1200
        assert response["workflow"]["canvas"]["height"] == 800
        assert response["workflow"]["canvas"]["theme"] == "light"
        assert "workflow_id" in response["workflow"]
        assert "component_categories" in response
        assert "next_steps" in response
    
    @pytest.mark.asyncio
    async def test_km_create_visual_workflow_with_template(self, tools):
        """Test workflow creation with template."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        
        # Test with email automation template
        result = await create_workflow_func(
            name="Email Workflow",
            template_id="email_automation",
            theme="dark"
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["workflow"]["name"] == "Email Workflow"
        assert response["workflow"]["canvas"]["theme"] == "dark"
    
    @pytest.mark.asyncio
    async def test_km_create_visual_workflow_invalid_theme(self, tools):
        """Test workflow creation with invalid theme."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        
        result = await create_workflow_func(
            name="Test Workflow",
            theme="invalid_theme"
        )
        
        assert result.startswith("Error:")
        assert "Invalid theme" in result
    
    @pytest.mark.asyncio
    async def test_km_add_workflow_component(self, tools):
        """Test adding component to workflow."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # First create a workflow
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Test Workflow")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Add component
        add_component_func = registered_tools["km_add_workflow_component"]
        result = await add_component_func(
            workflow_id=workflow_id,
            component_type="action",
            x_position=100,
            y_position=200,
            title="Test Action",
            description="Test action description",
            properties='{"test_prop": "test_value"}'
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["component"]["component_type"] == "action"
        assert response["component"]["title"] == "Test Action"
        assert response["component"]["position"]["x"] == 100
        assert response["component"]["position"]["y"] == 200
        assert response["component"]["properties"]["test_prop"] == "test_value"
        assert "component_id" in response["component"]
    
    @pytest.mark.asyncio
    async def test_km_add_workflow_component_invalid_type(self, tools):
        """Test adding component with invalid type."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow first
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Test Workflow")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Try to add component with invalid type
        add_component_func = registered_tools["km_add_workflow_component"]
        result = await add_component_func(
            workflow_id=workflow_id,
            component_type="invalid_type",
            x_position=100,
            y_position=200,
            title="Test"
        )
        
        assert result.startswith("Error:")
        assert "Invalid component_type" in result
    
    @pytest.mark.asyncio
    async def test_km_connect_workflow_nodes(self, tools):
        """Test connecting workflow components."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow and components
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Connection Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Add two components
        add_component_func = registered_tools["km_add_workflow_component"]
        
        comp1_result = await add_component_func(
            workflow_id=workflow_id,
            component_type="trigger",
            x_position=100,
            y_position=100,
            title="Trigger"
        )
        comp1_json = comp1_result.replace("```json\n", "").replace("\n```", "")
        comp1_response = json.loads(comp1_json)
        comp1_id = comp1_response["component"]["component_id"]
        
        comp2_result = await add_component_func(
            workflow_id=workflow_id,
            component_type="action",
            x_position=300,
            y_position=100,
            title="Action"
        )
        comp2_json = comp2_result.replace("```json\n", "").replace("\n```", "")
        comp2_response = json.loads(comp2_json)
        comp2_id = comp2_response["component"]["component_id"]
        
        # Connect components
        connect_func = registered_tools["km_connect_workflow_nodes"]
        result = await connect_func(
            workflow_id=workflow_id,
            source_component=comp1_id,
            target_component=comp2_id,
            connection_type="trigger",
            connection_label="Test Connection"
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["connection"]["connection_type"] == "trigger"
        assert response["connection"]["source_component"] == comp1_id
        assert response["connection"]["target_component"] == comp2_id
        assert response["connection"]["properties"]["label"] == "Test Connection"
        assert response["validation"]["is_valid"] == True
    
    @pytest.mark.asyncio
    async def test_km_edit_workflow_component(self, tools):
        """Test editing workflow component."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow and component
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Edit Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        add_component_func = registered_tools["km_add_workflow_component"]
        comp_result = await add_component_func(
            workflow_id=workflow_id,
            component_type="action",
            x_position=100,
            y_position=100,
            title="Original Title"
        )
        
        comp_json = comp_result.replace("```json\n", "").replace("\n```", "")
        comp_response = json.loads(comp_json)
        component_id = comp_response["component"]["component_id"]
        
        # Edit component
        edit_func = registered_tools["km_edit_workflow_component"]
        result = await edit_func(
            workflow_id=workflow_id,
            component_id=component_id,
            title="Updated Title",
            description="Updated description",
            properties='{"updated_prop": "updated_value"}',
            x_position=200,
            y_position=200
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["updated_component"]["title"] == "Updated Title"
        assert response["updated_component"]["description"] == "Updated description"
        assert response["updated_component"]["position"]["x"] == 200
        assert response["updated_component"]["position"]["y"] == 200
        assert response["updated_component"]["properties"]["updated_prop"] == "updated_value"
        assert response["changes_applied"]["title_changed"] == True
        assert response["changes_applied"]["position_changed"] == True
    
    @pytest.mark.asyncio
    async def test_km_get_workflow_templates(self, tools):
        """Test getting workflow templates."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        get_templates_func = registered_tools["km_get_workflow_templates"]
        result = await get_templates_func()
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert "templates" in response
        assert len(response["templates"]) > 0
        assert "template_categories" in response
        assert "complexity_levels" in response
        assert "component_library" in response
        
        # Check template structure
        for template in response["templates"]:
            assert "template_id" in template
            assert "name" in template
            assert "description" in template
            assert "category" in template
            assert "complexity" in template
            assert "preview_components" in template
    
    @pytest.mark.asyncio
    async def test_km_validate_workflow(self, tools):
        """Test workflow validation."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow with connected components
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Validation Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Add and connect components
        add_component_func = registered_tools["km_add_workflow_component"]
        
        comp1_result = await add_component_func(
            workflow_id=workflow_id,
            component_type="trigger",
            x_position=100,
            y_position=100,
            title="Trigger"
        )
        comp1_json = comp1_result.replace("```json\n", "").replace("\n```", "")
        comp1_response = json.loads(comp1_json)
        comp1_id = comp1_response["component"]["component_id"]
        
        comp2_result = await add_component_func(
            workflow_id=workflow_id,
            component_type="action",
            x_position=300,
            y_position=100,
            title="Action"
        )
        comp2_json = comp2_result.replace("```json\n", "").replace("\n```", "")
        comp2_response = json.loads(comp2_json)
        comp2_id = comp2_response["component"]["component_id"]
        
        connect_func = registered_tools["km_connect_workflow_nodes"]
        await connect_func(
            workflow_id=workflow_id,
            source_component=comp1_id,
            target_component=comp2_id
        )
        
        # Validate workflow
        validate_func = registered_tools["km_validate_workflow"]
        result = await validate_func(
            workflow_id=workflow_id,
            validation_level="full",
            check_performance=True,
            suggest_optimizations=True
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["validation_report"]["is_valid"] == True
        assert response["validation_report"]["validation_level"] == "full"
        assert "workflow_statistics" in response
        assert "performance_analysis" in response
        assert "optimization_suggestions" in response
        
        # Check workflow statistics
        stats = response["workflow_statistics"]
        assert stats["total_components"] == 2
        assert stats["total_connections"] == 1
        assert "component_types" in stats
    
    @pytest.mark.asyncio
    async def test_km_export_visual_workflow(self, tools):
        """Test workflow export functionality."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create simple workflow
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Export Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Export workflow
        export_func = registered_tools["km_export_visual_workflow"]
        result = await export_func(
            workflow_id=workflow_id,
            export_format="macro",
            include_metadata=True,
            validate_before_export=True,
            optimization_level="basic"
        )
        
        json_content = result.replace("```json\n", "").replace("\n```", "")
        response = json.loads(json_content)
        
        assert response["success"] == True
        assert response["export_result"]["export_format"] == "macro"
        assert response["export_result"]["optimization_applied"] == True
        assert "macro_id" in response["export_result"]
        assert "workflow_summary" in response
        assert "validation_report" in response
        
        # Check validation report
        validation = response["validation_report"]
        assert validation["is_valid"] == True
        assert len(validation["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_workflow_id(self, tools):
        """Test error handling for invalid workflow IDs."""
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        fake_workflow_id = "nonexistent_workflow_id"
        
        # Test various functions with invalid workflow ID
        add_component_func = registered_tools["km_add_workflow_component"]
        result = await add_component_func(
            workflow_id=fake_workflow_id,
            component_type="action",
            x_position=100,
            y_position=100,
            title="Test"
        )
        assert result.startswith("Error:")
        assert "not found" in result.lower()
        
        validate_func = registered_tools["km_validate_workflow"]
        result = await validate_func(workflow_id=fake_workflow_id)
        assert result.startswith("Error:")
        assert "not found" in result.lower()
        
        export_func = registered_tools["km_export_visual_workflow"]
        result = await export_func(workflow_id=fake_workflow_id)
        assert result.startswith("Error:")
        assert "not found" in result.lower()
    
    def test_global_tools_singleton(self):
        """Test global tools singleton pattern."""
        tools1 = get_workflow_designer_tools()
        tools2 = get_workflow_designer_tools()
        
        assert tools1 is tools2
        assert isinstance(tools1, WorkflowDesignerTools)


@pytest.mark.performance
class TestWorkflowDesignerToolsPerformance:
    """Performance tests for workflow designer MCP tools."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation_response_time(self):
        """Test workflow creation response time."""
        import time
        
        tools = WorkflowDesignerTools()
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        
        start_time = time.time()
        result = await create_workflow_func(name="Performance Test")
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete within 100ms
        assert duration < 0.1
        assert result.startswith("```json")
    
    @pytest.mark.asyncio
    async def test_component_addition_response_time(self):
        """Test component addition response time."""
        import time
        
        tools = WorkflowDesignerTools()
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow first
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Performance Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Test component addition performance
        add_component_func = registered_tools["km_add_workflow_component"]
        
        start_time = time.time()
        result = await add_component_func(
            workflow_id=workflow_id,
            component_type="action",
            x_position=100,
            y_position=100,
            title="Performance Test"
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete within 100ms
        assert duration < 0.1
        assert result.startswith("```json")
    
    @pytest.mark.asyncio
    async def test_validation_response_time(self):
        """Test workflow validation response time."""
        import time
        
        tools = WorkflowDesignerTools()
        mock_mcp = Mock()
        registered_tools = {}
        
        def mock_tool_decorator():
            def decorator(func):
                registered_tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp.tool = mock_tool_decorator
        tools.register_tools(mock_mcp)
        
        # Create workflow
        create_workflow_func = registered_tools["km_create_visual_workflow"]
        create_result = await create_workflow_func(name="Performance Test")
        
        create_json = create_result.replace("```json\n", "").replace("\n```", "")
        create_response = json.loads(create_json)
        workflow_id = create_response["workflow"]["workflow_id"]
        
        # Test validation performance
        validate_func = registered_tools["km_validate_workflow"]
        
        start_time = time.time()
        result = await validate_func(
            workflow_id=workflow_id,
            validation_level="full",
            check_performance=True
        )
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete within 200ms
        assert duration < 0.2
        assert result.startswith("```json")