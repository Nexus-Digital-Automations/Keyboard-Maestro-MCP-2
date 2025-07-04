"""
File System Monitor for Keyboard Maestro Library Changes

Monitors KM preferences and library files for changes to trigger
immediate synchronization instead of relying solely on polling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Callable, Optional, List
from pathlib import Path
import asyncio
import logging
import time
import os

# Try to import watchdog for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEvent = object  # Fallback type

logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a file system change event."""
    file_path: Path
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float
    is_directory: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "file_path": str(self.file_path),
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "is_directory": self.is_directory
        }


class KMFileMonitor:
    """Monitor Keyboard Maestro files for changes."""
    
    def __init__(self, change_callback: Callable[[FileChangeEvent], None]):
        self.change_callback = change_callback
        self._observer: Optional[Observer] = None
        self._watched_paths: Set[Path] = set()
        self._last_change_time = 0.0
        self._debounce_interval = 1.0  # seconds
        
        # Default KM file locations
        self.km_prefs_path = self._find_km_preferences_path()
        self.km_macros_path = self._find_km_macros_path()
        
        # Monitoring status
        self._is_monitoring = False
        self._change_count = 0
    
    def start_monitoring(self) -> bool:
        """Start file system monitoring."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File monitoring unavailable - watchdog package not installed")
            logger.info("Install with: pip install watchdog")
            return False
        
        if self._is_monitoring:
            logger.warning("File monitoring already started")
            return True
        
        if not self.km_prefs_path.exists():
            logger.warning(f"KM preferences path not found: {self.km_prefs_path}")
            # Try to create a minimal monitoring setup anyway
        
        try:
            self._observer = Observer()
            
            # Monitor preferences directory
            if self.km_prefs_path.exists():
                event_handler = KMFileEventHandler(self._on_file_change)
                self._observer.schedule(event_handler, str(self.km_prefs_path), recursive=True)
                self._watched_paths.add(self.km_prefs_path)
                logger.info(f"Monitoring KM preferences: {self.km_prefs_path}")
            
            # Monitor macros directory if different and exists
            if (self.km_macros_path != self.km_prefs_path and 
                self.km_macros_path.exists()):
                event_handler = KMFileEventHandler(self._on_file_change)
                self._observer.schedule(event_handler, str(self.km_macros_path), recursive=True)
                self._watched_paths.add(self.km_macros_path)
                logger.info(f"Monitoring KM macros: {self.km_macros_path}")
            
            if self._watched_paths:
                self._observer.start()
                self._is_monitoring = True
                logger.info(f"File monitoring started for {len(self._watched_paths)} paths")
                return True
            else:
                logger.warning("No valid paths found for file monitoring")
                return False
            
        except Exception as e:
            logger.exception(f"Failed to start file monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        if self._observer and self._is_monitoring:
            try:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None
                self._is_monitoring = False
                logger.info(f"File monitoring stopped (processed {self._change_count} changes)")
            except Exception as e:
                logger.exception(f"Error stopping file monitor: {e}")
    
    def get_status(self) -> dict:
        """Get monitoring status."""
        return {
            "is_monitoring": self._is_monitoring,
            "watchdog_available": WATCHDOG_AVAILABLE,
            "watched_paths": [str(p) for p in self._watched_paths],
            "changes_detected": self._change_count,
            "km_prefs_path": str(self.km_prefs_path),
            "km_macros_path": str(self.km_macros_path)
        }
    
    def _on_file_change(self, event: FileSystemEvent):
        """Handle file system change events."""
        current_time = time.time()
        
        # Debounce rapid changes
        if current_time - self._last_change_time < self._debounce_interval:
            return
        
        self._last_change_time = current_time
        
        # Filter for relevant files
        if not self._is_relevant_file(Path(event.src_path)):
            return
        
        change_event = FileChangeEvent(
            file_path=Path(event.src_path),
            event_type=event.event_type,
            timestamp=current_time,
            is_directory=event.is_directory
        )
        
        try:
            self.change_callback(change_event)
            self._change_count += 1
            logger.debug(f"File change detected: {event.event_type} {event.src_path}")
        except Exception as e:
            logger.exception(f"Error in file change callback: {e}")
    
    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if file change is relevant to macro library."""
        if not file_path.exists() and "deleted" not in str(file_path):
            return False
        
        # File extensions that matter
        relevant_extensions = {'.kmmacros', '.plist', '.json', '.xml'}
        
        # File names that matter
        relevant_names = {
            'Keyboard Maestro Preferences.plist',
            'Keyboard Maestro Engine Preferences.plist',
            'macros.json',
            'macros.plist',
            'macro_groups.json'
        }
        
        # Path patterns that matter
        relevant_patterns = ['macro', 'keyboardmaestro', 'km_']
        
        return (
            file_path.suffix.lower() in relevant_extensions or
            file_path.name in relevant_names or
            any(pattern in file_path.name.lower() for pattern in relevant_patterns) or
            any(pattern in str(file_path).lower() for pattern in relevant_patterns)
        )
    
    def _find_km_preferences_path(self) -> Path:
        """Find Keyboard Maestro preferences directory."""
        home = Path.home()
        
        # Standard macOS application support locations
        possible_paths = [
            # Modern KM location
            home / "Library" / "Application Support" / "Keyboard Maestro",
            # Preferences location
            home / "Library" / "Preferences" / "com.stairways.keyboardmaestro.engine.plist",
            # System-wide installation
            Path("/Library/Application Support/Keyboard Maestro"),
            # Alternative preferences location
            home / "Library" / "Preferences" / "Keyboard Maestro",
        ]
        
        for path in possible_paths:
            if path.exists():
                if path.is_file():
                    return path.parent
                return path
        
        # Default fallback - create the directory structure
        default_path = home / "Library" / "Application Support" / "Keyboard Maestro"
        logger.info(f"Using default KM path: {default_path}")
        return default_path
    
    def _find_km_macros_path(self) -> Path:
        """Find Keyboard Maestro macros directory."""
        # For modern KM versions, macros are usually in the same location as preferences
        macros_in_prefs = self.km_prefs_path / "Macros"
        if macros_in_prefs.exists():
            return macros_in_prefs
        
        # Alternative locations
        home = Path.home()
        alternative_paths = [
            home / "Documents" / "Keyboard Maestro",
            home / "Library" / "Keyboard Maestro",
            self.km_prefs_path / "Library"
        ]
        
        for path in alternative_paths:
            if path.exists():
                return path
        
        # Default to same as preferences
        return self.km_prefs_path


# Only define the event handler if watchdog is available
if WATCHDOG_AVAILABLE:
    class KMFileEventHandler(FileSystemEventHandler):
        """Handle file system events for KM files."""
        
        def __init__(self, callback: Callable[[FileSystemEvent], None]):
            super().__init__()
            self.callback = callback
        
        def on_modified(self, event):
            self.callback(event)
        
        def on_created(self, event):
            self.callback(event)
        
        def on_deleted(self, event):
            self.callback(event)
        
        def on_moved(self, event):
            self.callback(event)
else:
    # Fallback implementation when watchdog is not available
    class KMFileEventHandler:
        def __init__(self, callback):
            self.callback = callback
            logger.warning("KMFileEventHandler created without watchdog support")


class SimpleFileMonitor:
    """Simplified file monitor using basic file system checks."""
    
    def __init__(self, change_callback: Callable[[FileChangeEvent], None]):
        self.change_callback = change_callback
        self._monitor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._file_states: dict = {}
        self._check_interval = 10.0  # Check every 10 seconds
        
    async def start_monitoring(self, paths: List[Path]):
        """Start simple file monitoring."""
        logger.info(f"Starting simple file monitoring for {len(paths)} paths")
        
        # Initialize file states
        for path in paths:
            if path.exists():
                self._file_states[str(path)] = path.stat().st_mtime
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop(paths))
        return True
    
    async def stop_monitoring(self):
        """Stop simple file monitoring."""
        if self._monitor_task:
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitor_task.cancel()
            self._monitor_task = None
            logger.info("Simple file monitoring stopped")
    
    async def _monitor_loop(self, paths: List[Path]):
        """Simple monitoring loop."""
        while not self._stop_event.is_set():
            try:
                for path in paths:
                    if path.exists():
                        current_mtime = path.stat().st_mtime
                        stored_mtime = self._file_states.get(str(path))
                        
                        if stored_mtime is None:
                            # New file
                            self._file_states[str(path)] = current_mtime
                            change_event = FileChangeEvent(
                                file_path=path,
                                event_type="created",
                                timestamp=time.time(),
                                is_directory=path.is_dir()
                            )
                            self.change_callback(change_event)
                        elif current_mtime > stored_mtime:
                            # Modified file
                            self._file_states[str(path)] = current_mtime
                            change_event = FileChangeEvent(
                                file_path=path,
                                event_type="modified",
                                timestamp=time.time(),
                                is_directory=path.is_dir()
                            )
                            self.change_callback(change_event)
                
                await asyncio.sleep(self._check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in simple monitor loop: {e}")
                await asyncio.sleep(self._check_interval)