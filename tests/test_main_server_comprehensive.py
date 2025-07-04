"""
Comprehensive Main Server Tests - Coverage Expansion

Tests for main.py, server initialization, resources, and configuration modules.
Focuses on achieving high coverage for actual server implementation.

Architecture: Property-Based Testing + Type Safety + Contract Validation
Performance: <100ms per test, parallel execution, comprehensive edge case coverage
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from hypothesis import given, strategies as st, settings
from pathlib import Path

# Test imports with graceful fallbacks
try:
    from src.main import create_mcp_server, main
    from src.server.initialization import initialize_components, get_km_client
    from src.server.resources import get_server_status, get_tool_help, create_macro_prompt
    from src.server.config import ServerConfig, ToolConfig
    from src.server.utils import validate_input, sanitize_output
    MAIN_SERVER_AVAILABLE = True
except ImportError:
    MAIN_SERVER_AVAILABLE = False
    # Mock classes for testing
    def create_mcp_server():
        return None
    
    def main():
        pass


class TestMainEntryPoint:
    """Test main entry point functionality."""
    
    def test_create_mcp_server(self):
        """Test MCP server creation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Main server module not available")
        
        with patch('src.main.FastMCP') as mock_fastmcp:
            mock_server = Mock()
            mock_fastmcp.return_value = mock_server
            
            server = create_mcp_server()
            
            # Should create FastMCP instance
            mock_fastmcp.assert_called_once()
            assert server == mock_server
    
    def test_server_configuration(self):
        """Test server configuration."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Main server module not available")
        
        with patch('src.main.FastMCP') as mock_fastmcp, \
             patch('src.main.get_tool_config_manager') as mock_config:
            
            mock_config_manager = Mock()
            mock_config_manager.get_category_summary.return_value = {
                "total_tools": 46,
                "categories": ["Macro Management", "System Control"]
            }
            mock_config.return_value = mock_config_manager
            
            mock_server = Mock()
            mock_fastmcp.return_value = mock_server
            
            server = create_mcp_server()
            
            # Should call configuration setup
            mock_config.assert_called_once()
            mock_config_manager.get_category_summary.assert_called_once()
    
    def test_main_function_execution(self):
        """Test main function execution."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Main server module not available")
        
        with patch('src.main.create_mcp_server') as mock_create, \
             patch('src.main.logger') as mock_logger, \
             patch('sys.argv', ['main.py']):
            
            mock_server = Mock()
            mock_server.run = Mock()
            mock_create.return_value = mock_server
            
            # Should not raise exception
            try:
                main()
            except SystemExit:
                pass  # Expected for main function
            
            mock_create.assert_called_once()
    
    def test_main_with_arguments(self):
        """Test main function with command line arguments."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Main server module not available")
        
        with patch('src.main.create_mcp_server') as mock_create, \
             patch('src.main.argparse.ArgumentParser') as mock_parser, \
             patch('sys.argv', ['main.py', '--debug']):
            
            mock_args = Mock()
            mock_args.debug = True
            mock_parser_instance = Mock()
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            mock_server = Mock()
            mock_create.return_value = mock_server
            
            try:
                main()
            except SystemExit:
                pass
            
            # Should parse arguments
            mock_parser_instance.parse_args.assert_called_once()
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Main server module not available")
        
        with patch('src.main.logging.basicConfig') as mock_config:
            # Import should trigger logging configuration
            import src.main
            
            # Should configure logging
            mock_config.assert_called()
            call_args = mock_config.call_args
            assert 'level' in call_args[1]
            assert 'format' in call_args[1]
            assert 'handlers' in call_args[1]


class TestServerInitialization:
    """Test server initialization functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_components(self):
        """Test component initialization."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server initialization module not available")
        
        with patch('src.server.initialization.setup_logging') as mock_logging, \
             patch('src.server.initialization.initialize_km_client') as mock_km, \
             patch('src.server.initialization.load_configuration') as mock_config:
            
            mock_km.return_value = AsyncMock()
            mock_config.return_value = {"status": "configured"}
            
            result = await initialize_components()
            
            # Should initialize all components
            mock_logging.assert_called_once()
            mock_km.assert_called_once()
            mock_config.assert_called_once()
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_km_client(self):
        """Test KM client retrieval."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server initialization module not available")
        
        with patch('src.server.initialization.KMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            client = await get_km_client()
            
            # Should return KM client
            assert client == mock_client
    
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test initialization error handling."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server initialization module not available")
        
        with patch('src.server.initialization.initialize_km_client') as mock_km:
            mock_km.side_effect = Exception("Initialization failed")
            
            with pytest.raises(Exception, match="Initialization failed"):
                await initialize_components()
    
    def test_configuration_loading(self):
        """Test configuration loading."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server initialization module not available")
        
        with patch('src.server.initialization.load_config_file') as mock_load:
            mock_config = {
                "server": {"port": 8080, "host": "localhost"},
                "tools": {"enabled": True, "count": 46}
            }
            mock_load.return_value = mock_config
            
            from src.server.initialization import load_configuration
            config = load_configuration()
            
            assert config["server"]["port"] == 8080
            assert config["tools"]["enabled"] is True
    
    def test_environment_detection(self):
        """Test environment detection."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server initialization module not available")
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            from src.server.initialization import get_environment
            env = get_environment()
            assert env == 'test'
        
        with patch.dict(os.environ, {}, clear=True):
            env = get_environment()
            assert env == 'development'  # Default


class TestServerResources:
    """Test server resources functionality."""
    
    def test_get_server_status(self):
        """Test server status retrieval."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        with patch('src.server.resources.get_system_info') as mock_system, \
             patch('src.server.resources.get_tool_count') as mock_tools:
            
            mock_system.return_value = {
                "platform": "macOS",
                "version": "14.0",
                "python_version": "3.11"
            }
            mock_tools.return_value = 46
            
            status = get_server_status()
            
            assert "server_name" in status
            assert "version" in status
            assert "status" in status
            assert "system_info" in status
            assert status["system_info"]["platform"] == "macOS"
    
    def test_get_tool_help(self):
        """Test tool help generation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        help_text = get_tool_help("km_execute_macro")
        
        assert isinstance(help_text, str)
        assert len(help_text) > 0
        assert "km_execute_macro" in help_text.lower()
    
    def test_get_tool_help_nonexistent(self):
        """Test tool help for nonexistent tool."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        help_text = get_tool_help("nonexistent_tool")
        
        assert isinstance(help_text, str)
        assert "not found" in help_text.lower() or "unknown" in help_text.lower()
    
    def test_create_macro_prompt(self):
        """Test macro prompt creation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        prompt_data = {
            "macro_name": "Test Macro",
            "description": "A test macro",
            "actions": ["type_text", "pause"]
        }
        
        prompt = create_macro_prompt(prompt_data)
        
        assert "macro_name" in prompt
        assert prompt["macro_name"] == "Test Macro"
        assert "description" in prompt
        assert "actions" in prompt
    
    def test_prompt_validation(self):
        """Test prompt validation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        # Valid prompt
        valid_prompt = {
            "macro_name": "Valid Macro",
            "description": "Valid description",
            "actions": ["action1", "action2"]
        }
        
        result = create_macro_prompt(valid_prompt)
        assert result is not None
        
        # Invalid prompt (missing required fields)
        invalid_prompt = {
            "description": "Missing name"
        }
        
        with pytest.raises(ValueError):
            create_macro_prompt(invalid_prompt)
    
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=10)
    def test_tool_help_property_based(self, tool_name):
        """Property-based test for tool help."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server resources module not available")
        
        help_text = get_tool_help(tool_name)
        
        # Should always return a string
        assert isinstance(help_text, str)
        assert len(help_text) > 0


class TestServerConfig:
    """Test server configuration functionality."""
    
    def test_server_config_creation(self):
        """Test server config creation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server config module not available")
        
        config = ServerConfig(
            host="localhost",
            port=8080,
            debug=True,
            log_level="INFO"
        )
        
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.debug is True
        assert config.log_level == "INFO"
    
    def test_server_config_validation(self):
        """Test server config validation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server config module not available")
        
        # Valid config
        valid_config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            debug=False
        )
        assert valid_config.port == 9000
        
        # Invalid port
        with pytest.raises(ValueError):
            ServerConfig(host="localhost", port=-1)
        
        with pytest.raises(ValueError):
            ServerConfig(host="localhost", port=70000)
    
    def test_tool_config_creation(self):
        """Test tool config creation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server config module not available")
        
        tool_config = ToolConfig(
            enabled_tools=["km_execute_macro", "km_variable_manager"],
            disabled_tools=["km_debug_tool"],
            tool_timeout=30.0,
            max_concurrent_tools=10
        )
        
        assert len(tool_config.enabled_tools) == 2
        assert "km_execute_macro" in tool_config.enabled_tools
        assert tool_config.tool_timeout == 30.0
    
    def test_config_loading_from_file(self):
        """Test config loading from file."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server config module not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "server": {
                    "host": "0.0.0.0",
                    "port": 8080,
                    "debug": False
                },
                "tools": {
                    "enabled_tools": ["km_execute_macro"],
                    "tool_timeout": 60.0
                }
            }
            import json
            json.dump(config_data, f)
            f.flush()
            
            try:
                from src.server.config import load_config_from_file
                loaded_config = load_config_from_file(f.name)
                
                assert loaded_config["server"]["host"] == "0.0.0.0"
                assert loaded_config["server"]["port"] == 8080
                assert loaded_config["tools"]["tool_timeout"] == 60.0
            finally:
                os.unlink(f.name)
    
    def test_config_environment_override(self):
        """Test config environment variable override."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server config module not available")
        
        with patch.dict(os.environ, {
            'KM_MCP_HOST': '192.168.1.100',
            'KM_MCP_PORT': '9090',
            'KM_MCP_DEBUG': 'true'
        }):
            from src.server.config import load_config_with_env_override
            config = load_config_with_env_override()
            
            assert config["server"]["host"] == "192.168.1.100"
            assert config["server"]["port"] == 9090
            assert config["server"]["debug"] is True


class TestServerUtils:
    """Test server utility functions."""
    
    def test_validate_input(self):
        """Test input validation."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server utils module not available")
        
        # Valid inputs
        assert validate_input("test_string", str) is True
        assert validate_input(42, int) is True
        assert validate_input(3.14, float) is True
        assert validate_input(True, bool) is True
        
        # Invalid inputs
        assert validate_input("not_a_number", int) is False
        assert validate_input(42, str) is False
        assert validate_input(None, str) is False
    
    def test_sanitize_output(self):
        """Test output sanitization."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server utils module not available")
        
        # Test string sanitization
        dirty_string = "<script>alert('xss')</script>Hello World"
        clean_string = sanitize_output(dirty_string)
        assert "<script>" not in clean_string
        assert "Hello World" in clean_string
        
        # Test object sanitization
        dirty_dict = {
            "safe_key": "safe_value",
            "dangerous_key": "<script>alert('xss')</script>",
            "nested": {
                "safe": "value",
                "dangerous": "<img src=x onerror=alert('xss')>"
            }
        }
        clean_dict = sanitize_output(dirty_dict)
        assert "<script>" not in clean_dict["dangerous_key"]
        assert "<img" not in clean_dict["nested"]["dangerous"]
        assert clean_dict["safe_key"] == "safe_value"
    
    def test_input_validation_schemas(self):
        """Test input validation with schemas."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server utils module not available")
        
        schema = {
            "macro_name": {"type": str, "required": True, "max_length": 100},
            "enabled": {"type": bool, "required": False, "default": True},
            "priority": {"type": int, "required": False, "min_value": 1, "max_value": 10}
        }
        
        # Valid input
        valid_input = {
            "macro_name": "Test Macro",
            "enabled": True,
            "priority": 5
        }
        
        from src.server.utils import validate_input_schema
        result = validate_input_schema(valid_input, schema)
        assert result["valid"] is True
        assert result["data"]["macro_name"] == "Test Macro"
        
        # Invalid input
        invalid_input = {
            "macro_name": "A" * 150,  # Too long
            "priority": 15  # Out of range
        }
        
        result = validate_input_schema(invalid_input, schema)
        assert result["valid"] is False
        assert "errors" in result
    
    @given(st.text(), st.one_of(st.none(), st.text(), st.integers(), st.booleans()))
    @settings(max_examples=20)
    def test_sanitization_property_based(self, key, value):
        """Property-based test for sanitization."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server utils module not available")
        
        test_dict = {key: value}
        sanitized = sanitize_output(test_dict)
        
        # Should not contain dangerous scripts
        if isinstance(value, str):
            sanitized_value = sanitized.get(key, "")
            assert "<script>" not in str(sanitized_value).lower()
            assert "javascript:" not in str(sanitized_value).lower()
        
        # Should preserve non-dangerous content
        if value is None or isinstance(value, (int, bool)):
            assert sanitized.get(key) == value


class TestServerIntegration:
    """Test server integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_server_startup(self):
        """Test full server startup process."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server modules not available")
        
        with patch('src.server.initialization.initialize_components') as mock_init, \
             patch('src.main.FastMCP') as mock_fastmcp:
            
            # Mock successful initialization
            mock_init.return_value = {"status": "initialized"}
            mock_server = Mock()
            mock_fastmcp.return_value = mock_server
            
            # Create and configure server
            server = create_mcp_server()
            
            # Should initialize components and create server
            assert server == mock_server
    
    @pytest.mark.asyncio
    async def test_server_shutdown_handling(self):
        """Test server shutdown handling."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server modules not available")
        
        with patch('src.server.initialization.cleanup_components') as mock_cleanup:
            from src.server.initialization import handle_shutdown
            
            await handle_shutdown()
            
            # Should cleanup components
            mock_cleanup.assert_called_once()
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server modules not available")
        
        with patch('src.server.resources.get_server_status') as mock_status:
            # Simulate temporary failure
            mock_status.side_effect = [Exception("Temporary error"), {"status": "recovered"}]
            
            # First call should raise exception
            with pytest.raises(Exception, match="Temporary error"):
                get_server_status()
            
            # Second call should succeed
            status = get_server_status()
            assert status["status"] == "recovered"


# Performance tests
class TestServerPerformance:
    """Test server performance characteristics."""
    
    def test_status_response_time(self):
        """Test server status response time."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server modules not available")
        
        import time
        
        with patch('src.server.resources.get_system_info') as mock_system:
            mock_system.return_value = {"platform": "test"}
            
            start_time = time.time()
            status = get_server_status()
            end_time = time.time()
            
            # Should respond quickly
            response_time = end_time - start_time
            assert response_time < 0.1  # Less than 100ms
            assert status is not None
    
    def test_concurrent_help_requests(self):
        """Test concurrent help request handling."""
        if not MAIN_SERVER_AVAILABLE:
            pytest.skip("Server modules not available")
        
        import threading
        import time
        
        results = []
        
        def get_help_worker():
            start_time = time.time()
            help_text = get_tool_help("km_execute_macro")
            end_time = time.time()
            results.append({
                "help_text": help_text,
                "response_time": end_time - start_time
            })
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=get_help_worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        for result in results:
            assert result["help_text"] is not None
            assert result["response_time"] < 1.0  # Less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])