"""
Logging utilities for the Keyboard Maestro MCP server.

This module provides structured logging with security-conscious error handling
and performance monitoring for the macro automation system.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set level
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (stderr to avoid corrupting MCP communications)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if logs directory exists
        logs_dir = Path("logs")
        if logs_dir.exists():
            file_handler = logging.FileHandler(logs_dir / "km-mcp-server.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger