"""Advanced window management and positioning system.

import logging

logging.basicConfig(level=logging.DEBUG)
Provides comprehensive window control, grid management, and advanced
positioning capabilities for Keyboard Maestro automation.
"""

__all__ = [
    "AdvancedPositioning",
    "AdvancedGridManager",
]

from .advanced_positioning import AdvancedPositioning
from .grid_manager import AdvancedGridManager
