"""Server configuration and FastMCP setup.

Contains server configuration, logging setup, and FastMCP initialization.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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


@dataclass
class ToolConfig:
    """Configuration for tool management."""

    enabled_tools: list[str] = None
    disabled_tools: list[str] = None
    tool_timeout: float = 30.0
    max_concurrent_tools: int = 10

    def __post_init__(self) -> None:
        if self.enabled_tools is None:
            self.enabled_tools = []  # type: ignore[unreachable]  # runtime guard for callers passing None
        if self.disabled_tools is None:
            self.disabled_tools = []  # type: ignore[unreachable]  # runtime guard for callers passing None


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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logging.getLogger(__name__)


def create_fastmcp_server(config: ServerConfig) -> FastMCP:
    """Create and configure the FastMCP server instance."""
    instructions = """
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

    return FastMCP(name=config.name, instructions=instructions)


def get_server_config() -> ServerConfig:
    """Get the default server configuration."""
    return ServerConfig()


def load_config_from_file(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    import json

    try:
        with open(config_path) as f:
            return cast("dict[str, Any]", json.load(f))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def load_config_with_env_override() -> dict:
    """Load configuration with environment variable overrides."""
    import os

    config = {
        "server": {
            "host": os.getenv("KM_MCP_HOST", "127.0.0.1"),
            "port": int(os.getenv("KM_MCP_PORT", "8000")),
            "debug": os.getenv("KM_MCP_DEBUG", "false").lower() == "true",
        },
        "tools": {"enabled_tools": [], "tool_timeout": 30.0},
    }
    return config
