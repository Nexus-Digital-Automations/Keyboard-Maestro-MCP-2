"""Visual workflow composition and design system.

import logging

logging.basicConfig(level=logging.DEBUG)
Provides comprehensive workflow design, visual composition,
and component library management for Keyboard Maestro automation.
"""

__all__ = [
    "VisualComposer",
    "ComponentLibrary",
]

from .component_library import ComponentLibrary
from .visual_composer import VisualComposer
