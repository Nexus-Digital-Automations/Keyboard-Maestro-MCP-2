"""Voice control and speech recognition system.

import logging

logging.basicConfig(level=logging.DEBUG)
Provides comprehensive voice command processing, speech recognition,
and natural language understanding for Keyboard Maestro automation.
"""

__all__ = [
    "SpeechRecognizer",
    "VoiceFeedbackSystem",
    "VoiceCommandDispatcher",
    "IntentProcessor",
]

from .command_dispatcher import VoiceCommandDispatcher
from .intent_processor import IntentProcessor
from .speech_recognizer import SpeechRecognizer
from .voice_feedback import VoiceFeedbackSystem
