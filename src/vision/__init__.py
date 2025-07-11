"""Computer vision and image recognition system.

import logging

logging.basicConfig(level=logging.DEBUG)
Provides comprehensive image analysis, object detection, OCR,
and screen analysis capabilities for Keyboard Maestro automation.
"""

__all__ = [
    "ImageRecognitionEngine",
    "ObjectDetector",
    "SceneAnalyzer",
    "ScreenAnalysisEngine",
]

from .image_recognition import ImageRecognitionEngine
from .object_detector import ObjectDetector
from .scene_analyzer import SceneAnalyzer
from .screen_analysis import ScreenAnalysisEngine
