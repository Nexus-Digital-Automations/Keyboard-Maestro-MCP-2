"""
Server configuration and FastMCP setup.

Contains server configuration, logging setup, and FastMCP initialization.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP


@dataclass
class ServerConfig:
    """Configuration for the KM MCP Server."""
    name: str = "KeyboardMaestroMCP"
    version: str = "1.0.0"
    log_level: str = "INFO"
    logs_dir: str = "logs"
    log_file: str = "km-mcp-server.log"
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000


def setup_logging(config: ServerConfig) -> logging.Logger:
    """Setup logging configuration for the server."""
    
    # Create logs directory if it doesn't exist
    logs_path = Path(config.logs_dir)
    logs_path.mkdir(exist_ok=True)
    
    # Configure logging to stderr to avoid corrupting MCP communications
    handlers = [logging.StreamHandler(sys.stderr)]
    
    # Add file handler if logs directory exists
    if logs_path.exists():
        handlers.append(logging.FileHandler(logs_path / config.log_file))
    else:
        handlers.append(logging.NullHandler())
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def create_fastmcp_server(config: ServerConfig) -> FastMCP:
    """Create and configure the FastMCP server instance."""
    
    instructions = f"""
This server provides comprehensive Keyboard Maestro automation capabilities through 9 production-ready MCP tools.

CAPABILITIES:
- Macro execution and management with enterprise-grade error handling
- Variable and dictionary operations with type safety
- Trigger and condition management with functional programming patterns
- Application control and window management
- File operations and system integration
- OCR and image recognition automation
- Plugin system support and custom action creation

SECURITY: All operations include input validation, permission checking, and audit logging.
PERFORMANCE: Sub-second response times with connection pooling and intelligent caching.
TYPE SAFETY: Complete branded type system with contract-driven development.

Use these tools to automate any macOS task that Keyboard Maestro can perform.
    """.strip()
    
    return FastMCP(
        name=config.name,
        instructions=instructions
    )


def get_server_config() -> ServerConfig:
    """Get the default server configuration."""
    return ServerConfig()