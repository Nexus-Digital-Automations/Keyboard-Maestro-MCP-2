#!/usr/bin/env python3
"""
Keyboard Maestro MCP Server - Main Entry Point (Dynamic Registration)

Advanced macOS automation through Model Context Protocol using FastMCP framework
with dynamic tool registration system. Provides 46+ production-ready tools for 
comprehensive Keyboard Maestro integration.

Security: All operations include input validation, permission checking, and audit logging.
Performance: Sub-second response times with connection pooling and intelligent caching.
Type Safety: Complete branded type system with contract-driven development.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.prompts import Message
from pydantic import Field
from typing_extensions import Annotated

# Import modular components
from .server.initialization import initialize_components, get_km_client
from .server.resources import get_server_status, get_tool_help, create_macro_prompt
from .server.dynamic_registration import register_tools_dynamically, DynamicToolRegistrar
from .server.tool_config import get_tool_config_manager

# Configure logging to stderr to avoid corrupting MCP communications
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('logs/km-mcp-server.log') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server with comprehensive instructions."""
    
    # Get tool configuration for dynamic capability description
    config_manager = get_tool_config_manager()
    category_summary = config_manager.get_category_summary()
    
    # Build dynamic capabilities description
    capabilities = [
        "CORE CAPABILITIES:",
        "- Macro execution and management with enterprise-grade error handling",
        "- Variable and dictionary operations with type safety",
        "- Advanced macro search and metadata analysis",
        "- Real-time synchronization and monitoring",
        "",
        "AUTOMATION FEATURES:",
        "- Macro group management and organization",
        "- Macro creation and template-based generation",
        "- Comprehensive clipboard operations with history and security validation",
        "- Application control and window management",
        "- Secure file operations with path validation and transaction safety",
        "",
        "INTELLIGENT AUTOMATION:",
        "- Conditional logic and control flow structures (if/then/else, loops, switch/case)",
        "- Advanced trigger system (time-based, file-based, system events)",
        "- Mathematical calculation engine with security validation",
        "- Token processing with variable substitution and context evaluation",
        "",
        "VISUAL & INTERFACE AUTOMATION:",
        "- OCR and image recognition automation",
        "- Interface automation with mouse/keyboard simulation",
        "- Advanced window management with multi-monitor support",
        "- Notification system with alerts, HUD displays, and user feedback",
        "",
        "ENTERPRISE INTEGRATION:",
        "- Enterprise audit logging with compliance reporting",
        "- Analytics engine with ROI calculation and performance monitoring",
        "- Workflow intelligence with natural language processing",
        "- Plugin ecosystem with custom action creation and third-party integration",
        "",
        "ADVANCED PLATFORMS:",
        "- IoT device control and automation with multi-protocol support",
        "- Smart home automation with scene management and scheduling",
        "- Voice control with speech recognition and processing",
        "- Biometric integration for authentication and personalization",
        "",
        f"TOOL SUMMARY: {sum(category_summary.values())} production-ready tools across {len(category_summary)} categories",
        "",
        "SECURITY: All operations include input validation, permission checking, and comprehensive audit logging.",
        "PERFORMANCE: Sub-second response times with connection pooling and intelligent caching.",
        "TYPE SAFETY: Complete branded type system with contract-driven development.",
        "",
        "Use these tools to automate any macOS task that Keyboard Maestro can perform."
    ]
    
    mcp = FastMCP(
        name="KeyboardMaestroMCP",
        instructions="\n".join(capabilities)
    )
    
    return mcp


def register_resources_and_prompts(mcp: FastMCP) -> None:
    """Register resources and prompts with the MCP server."""
    
    @mcp.resource("km://server/status")
    def get_server_status_resource() -> Dict[str, Any]:
        """Get current server status and configuration."""
        return get_server_status()

    @mcp.resource("km://help/tools")
    def get_tool_help_resource() -> str:
        """Get comprehensive help for all available tools."""
        return get_tool_help()

    @mcp.prompt()
    def create_macro_prompt_handler(
        task_description: Annotated[str, Field(
            description="Description of the automation task to create a macro for"
        )],
        app_context: Annotated[str, Field(
            default="",
            description="Specific application or context for the automation"
        )] = ""
    ) -> list[Message]:
        """Generate a structured prompt for creating Keyboard Maestro macros."""
        return create_macro_prompt(task_description, app_context)


def main():
    """Main entry point for the Keyboard Maestro MCP server."""
    parser = argparse.ArgumentParser(description="Keyboard Maestro MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport method (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--tool-filter",
        nargs="*",
        help="Filter tools to register (tool names or categories)"
    )
    parser.add_argument(
        "--disable-experimental",
        action="store_true",
        help="Disable experimental tools"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("🚀 Starting Keyboard Maestro MCP Server v2.0.0 (Dynamic Registration)")
    logger.info(f"📡 Transport: {args.transport}")
    
    # Initialize components
    try:
        initialize_components()
        logger.info("✅ Components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Create FastMCP server
    try:
        mcp = create_mcp_server()
        logger.info("✅ FastMCP server created")
    except Exception as e:
        logger.error(f"❌ Failed to create FastMCP server: {e}")
        return 1
    
    # Register resources and prompts
    try:
        register_resources_and_prompts(mcp)
        logger.info("✅ Resources and prompts registered")
    except Exception as e:
        logger.error(f"❌ Failed to register resources and prompts: {e}")
        return 1
    
    # Dynamic tool registration
    try:
        logger.info("🔧 Starting dynamic tool registration...")
        registrar = register_tools_dynamically(mcp)
        
        registered_tools = registrar.get_registered_tools()
        logger.info(f"✅ Successfully registered {len(registered_tools)} tools dynamically")
        
        # Log registration summary by category
        config_manager = get_tool_config_manager()
        category_summary = config_manager.get_category_summary()
        
        logger.info("📊 Tool Registration Summary:")
        for category, count in sorted(category_summary.items()):
            tools_in_category = registrar.get_tools_by_category(category)
            logger.info(f"   {category}: {len(tools_in_category)} tools")
        
        # Log any experimental or disabled tools
        disabled_count = len(config_manager.configurations) - len(registered_tools)
        if disabled_count > 0:
            logger.info(f"⚠️  {disabled_count} tools disabled or not registered")
            
    except Exception as e:
        logger.error(f"❌ Failed to register tools dynamically: {e}")
        return 1
    
    # Start the server
    try:
        if args.transport == "stdio":
            logger.info("🖥️  Running with stdio transport for local clients")
            mcp.run(transport="stdio")
        else:
            logger.info(f"🌐 Running SSE server on {args.host}:{args.port}")
            mcp.run(transport="sse", host=args.host, port=args.port)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())