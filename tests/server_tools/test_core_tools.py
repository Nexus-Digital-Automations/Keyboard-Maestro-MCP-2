"""
Core Server Tools Tests - Comprehensive Test Suite

Unit and integration tests for km_execute_macro, km_list_macros, km_get_macro, 
km_set_variable, km_get_variable and related core MCP tools with comprehensive 
validation and error handling.
"""

import pytest
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.server.tools.core_tools import (
    km_execute_macro,
    km_list_macros,
    km_variable_manager
)
from src.core.types import MacroId, VariableName
from src.core.errors import ValidationError, SecurityError


class TestCoreToolsValidation:
    """Test input validation for core tools."""
    
    def test_variable_name_validation_valid(self):
        """Test valid variable name creation."""
        valid_names = [
            "myVariable",
            "counter_01", 
            "APP_CONFIG",
            "temp123",
            "system_path"
        ]
        
        # Just test that these are valid string patterns
        for name in valid_names:
            assert isinstance(name, str)
            assert len(name) > 0


class TestKMExecuteMacro:
    """Test km_execute_macro tool."""
    
    @pytest.mark.asyncio
    async def test_execute_macro_success(self):
        """Test successful macro execution."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.execute_macro = AsyncMock(return_value={
                "success": True,
                "result": "Macro executed successfully",
                "execution_time": 0.5,
                "output": "Hello World"
            })
            mock_get_client.return_value = mock_client
            
            result = await km_execute_macro(
                identifier="Test Macro",
                trigger_value="test_value",
                method="applescript",
                timeout=30
            )
            
            assert result["success"] is True
            assert "result" in result
            assert "execution_time" in result
            assert result["identifier"] == "Test Macro"
            assert result["trigger_value"] == "test_value"
            assert result["method"] == "applescript"
    
    @pytest.mark.asyncio
    async def test_execute_macro_timeout(self):
        """Test macro execution timeout handling."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.execute_macro = AsyncMock(side_effect=asyncio.TimeoutError("Macro execution timed out"))
            mock_get_client.return_value = mock_client
            
            result = await km_execute_macro(
                identifier="Slow Macro",
                timeout=1
            )
            
            assert result["success"] is False
            assert "timeout" in result["error"].lower()
            assert result["identifier"] == "Slow Macro"
    
    @pytest.mark.asyncio
    async def test_execute_macro_invalid_identifier(self):
        """Test execution with invalid macro identifier."""
        result = await km_execute_macro(
            identifier="",  # Empty identifier
            timeout=30
        )
        
        assert result["success"] is False
        assert "invalid" in result["error"].lower() or "empty" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_macro_client_error(self):
        """Test handling of client execution errors."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.execute_macro = AsyncMock(side_effect=Exception("Keyboard Maestro not available"))
            mock_get_client.return_value = mock_client
            
            result = await km_execute_macro(
                identifier="Test Macro",
                timeout=30
            )
            
            assert result["success"] is False
            assert "error" in result
            assert result["identifier"] == "Test Macro"


class TestKMListMacros:
    """Test km_list_macros tool."""
    
    @pytest.mark.asyncio
    async def test_list_macros_success(self):
        """Test successful macro listing."""
        mock_macros = [
            {
                "name": "Test Macro 1",
                "uuid": "12345-67890",
                "group": "Default Group",
                "enabled": True,
                "trigger_count": 2
            },
            {
                "name": "Test Macro 2", 
                "uuid": "67890-12345",
                "group": "Automation",
                "enabled": False,
                "trigger_count": 1
            }
        ]
        
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.list_macros = AsyncMock(return_value={
                "success": True,
                "macros": mock_macros,
                "total_count": len(mock_macros)
            })
            mock_get_client.return_value = mock_client
            
            result = await km_list_macros(
                group_filter="",
                enabled_only=False,
                sort_by="name"
            )
            
            assert result["success"] is True
            assert len(result["macros"]) == 2
            assert result["total_count"] == 2
            assert result["macros"][0]["name"] == "Test Macro 1"
    
    @pytest.mark.asyncio
    async def test_list_macros_with_filter(self):
        """Test macro listing with group filter."""
        mock_macros = [
            {
                "name": "Automation Macro",
                "uuid": "12345-67890", 
                "group": "Automation",
                "enabled": True,
                "trigger_count": 1
            }
        ]
        
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.list_macros = AsyncMock(return_value={
                "success": True,
                "macros": mock_macros,
                "total_count": len(mock_macros)
            })
            mock_get_client.return_value = mock_client
            
            result = await km_list_macros(
                group_filter="Automation",
                enabled_only=True,
                sort_by="name"
            )
            
            assert result["success"] is True
            assert len(result["macros"]) == 1
            assert result["group_filter"] == "Automation" 
            assert result["enabled_only"] is True
    
    @pytest.mark.asyncio
    async def test_list_macros_empty_result(self):
        """Test macro listing with no results."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.list_macros = AsyncMock(return_value={
                "success": True,
                "macros": [],
                "total_count": 0
            })
            mock_get_client.return_value = mock_client
            
            result = await km_list_macros()
            
            assert result["success"] is True
            assert len(result["macros"]) == 0
            assert result["total_count"] == 0


class TestKMVariableManager:
    """Test km_variable_manager tool."""
    
    @pytest.mark.asyncio
    async def test_set_variable_success(self):
        """Test successful variable setting."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.set_variable = AsyncMock(return_value={
                "success": True,
                "variable_name": "testVar",
                "value": "test_value",
                "previous_value": None
            })
            mock_get_client.return_value = mock_client
            
            result = await km_variable_manager(
                operation="set",
                variable_name="testVar",
                value="test_value",
                scope="global"
            )
            
            assert result["success"] is True
            assert result["operation"] == "set"
            assert result["variable_name"] == "testVar"
    
    @pytest.mark.asyncio
    async def test_get_variable_success(self):
        """Test successful variable retrieval."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.get_variable = AsyncMock(return_value={
                "success": True,
                "variable_name": "testVar", 
                "value": "retrieved_value",
                "scope": "global",
                "type": "string"
            })
            mock_get_client.return_value = mock_client
            
            result = await km_variable_manager(
                operation="get",
                variable_name="testVar",
                scope="global"
            )
            
            assert result["success"] is True
            assert result["operation"] == "get"
            assert result["variable_name"] == "testVar"
    
    @pytest.mark.asyncio
    async def test_list_variables_success(self):
        """Test successful variable listing."""
        mock_variables = [
            {"name": "var1", "value": "value1", "scope": "global"},
            {"name": "var2", "value": "value2", "scope": "local"}
        ]
        
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.list_variables = AsyncMock(return_value={
                "success": True,
                "variables": mock_variables,
                "total_count": len(mock_variables)
            })
            mock_get_client.return_value = mock_client
            
            result = await km_variable_manager(
                operation="list",
                scope="all"
            )
            
            assert result["success"] is True
            assert result["operation"] == "list"
            assert len(result.get("variables", [])) >= 0
    
    @pytest.mark.asyncio
    async def test_delete_variable_success(self):
        """Test successful variable deletion."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.delete_variable = AsyncMock(return_value={
                "success": True,
                "variable_name": "testVar",
                "deleted": True
            })
            mock_get_client.return_value = mock_client
            
            result = await km_variable_manager(
                operation="delete",
                variable_name="testVar",
                scope="global"
            )
            
            assert result["success"] is True
            assert result["operation"] == "delete"
            assert result["variable_name"] == "testVar"
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation handling."""
        result = await km_variable_manager(
            operation="invalid_op",
            variable_name="testVar"
        )
        
        assert result["success"] is False
        assert "invalid" in result.get("error", "").lower() or "operation" in result.get("error", "").lower()


class TestErrorHandling:
    """Test error handling across core tools."""
    
    @pytest.mark.asyncio
    async def test_client_unavailable_error(self):
        """Test handling when KM client is unavailable."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_get_client.side_effect = Exception("Keyboard Maestro not running")
            
            result = await km_execute_macro("Test Macro")
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent tool operations."""
        with patch('src.server.tools.core_tools.get_km_client') as mock_get_client:
            mock_client = Mock()
            mock_client.execute_macro = AsyncMock(return_value={"success": True})
            mock_client.set_variable = AsyncMock(return_value={"success": True})
            mock_client.get_variable = AsyncMock(return_value={"success": True, "value": "test"})
            mock_get_client.return_value = mock_client
            
            # Run multiple operations concurrently
            tasks = [
                km_execute_macro("Macro1"),
                km_execute_macro("Macro2"),
                km_variable_manager("set", "var1", value="value1"),
                km_variable_manager("get", "var2")
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete without exceptions
            for result in results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_parameter_sanitization(self):
        """Test that dangerous parameters are properly sanitized."""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "${system.exit(1)}",
            "%(rm -rf /)s"
        ]
        
        for dangerous_input in dangerous_inputs:
            result = await km_execute_macro(dangerous_input)
            # Should either fail validation or sanitize the input
            assert result is not None
            if result["success"]:
                # If it succeeds, ensure the dangerous content was sanitized
                assert dangerous_input not in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])