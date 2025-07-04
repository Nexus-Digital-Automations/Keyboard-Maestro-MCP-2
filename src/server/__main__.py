#!/usr/bin/env python3
"""
MCP Server Entry Point

This module allows the server to be run as `python -m src.server`
by importing and executing the main server from src.main
"""

if __name__ == "__main__":
    # Import and run the main server
    from ..main import main
    main()