# TASK_7: Real-time Macro State Synchronization

**Created By**: Agent_2 | **Priority**: MEDIUM | **Duration**: 2 hours
**Technique Focus**: Real-time Monitoring + Efficient Caching + Event-driven Updates
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED
**Assigned**: Agent_1
**Dependencies**: TASK_5 (Real KM API integration), TASK_6 (Enhanced metadata)
**Blocking**: None (Final enhancement task)

## 📖 Required Reading (Complete before starting)
- [ ] **TASK_5.md**: Real KM API integration and connection methods
- [ ] **TASK_6.md**: Enhanced metadata extraction and smart filtering
- [ ] **src/integration/km_client.py**: KM client implementation and async methods
- [ ] **src/integration/macro_metadata.py**: Enhanced metadata structures from TASK_6
- [ ] **Keyboard Maestro Documentation**: Macro change notifications and file monitoring
- [ ] **tests/TESTING.md**: Current integration test status and performance benchmarks

## 🎯 Implementation Overview
Implement real-time synchronization of macro library state to ensure AI clients always have current information about macro availability, status changes, and library modifications without performance degradation from constant polling.

<thinking>
Challenge: Keep macro information current without overwhelming the system
Solution approach:
1. Change Detection: Monitor KM library files and notification events
2. Intelligent Polling: Adaptive polling based on user activity and change frequency
3. Event-driven Updates: Push notifications when macros change state
4. Efficient Caching: Smart cache invalidation and selective updates
5. Performance Optimization: Minimize overhead while maintaining accuracy
6. State Management: Maintain synchronized local cache with change tracking
</thinking>

## ✅ Implementation Subtasks (Sequential completion)

### Phase 1: Change Detection Infrastructure ✅ COMPLETED
- [x] **File system monitoring**: Watch KM preferences and macro library files
- [x] **Macro state polling**: Implement adaptive polling strategy
- [x] **Change detection logic**: Identify macro additions, removals, and modifications
- [x] **Event notification system**: Create event bus for macro state changes

### Phase 2: Real-time Update Mechanisms ✅ COMPLETED
- [x] **Incremental updates**: Update only changed macros instead of full refresh
- [x] **Cache invalidation**: Smart cache expiration and selective updates
- [x] **Delta synchronization**: Track and apply only the changes since last update
- [x] **Conflict resolution**: Handle concurrent updates and state conflicts

### Phase 3: Performance and Monitoring ✅ COMPLETED
- [x] **Performance optimization**: Minimize overhead and resource usage
- [x] **Update throttling**: Rate limiting to prevent overwhelming the system
- [x] **Health monitoring**: Track synchronization health and performance
- [x] **Diagnostic tools**: Debugging and monitoring capabilities for sync status

## 🔧 Implementation Files & Specifications

### Core Synchronization Files to Create:

#### src/integration/sync_manager.py - Real-time Synchronization Manager
```python
"""
Real-time Macro State Synchronization Manager

Provides real-time monitoring and synchronization of Keyboard Maestro
macro library state with intelligent caching and change detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import time

from ..core.types import MacroId, Duration
from ..core.contracts import require, ensure
from .km_client import KMClient, Either, KMError
from .macro_metadata import EnhancedMacroMetadata, MacroMetadataExtractor


logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of macro library changes."""
    MACRO_ADDED = "macro_added"
    MACRO_REMOVED = "macro_removed"
    MACRO_MODIFIED = "macro_modified"
    MACRO_ENABLED = "macro_enabled"
    MACRO_DISABLED = "macro_disabled"
    GROUP_ADDED = "group_added"
    GROUP_REMOVED = "group_removed"
    GROUP_RENAMED = "group_renamed"
    LIBRARY_RELOADED = "library_reloaded"


class SyncStatus(Enum):
    """Synchronization status states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class MacroChange:
    """Represents a change in macro library state."""
    change_type: ChangeType
    macro_id: Optional[MacroId] = None
    macro_name: Optional[str] = None
    group_name: Optional[str] = None
    old_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    change_hash: str = field(init=False)
    
    def __post_init__(self):
        # Generate hash for change deduplication
        change_data = f"{self.change_type}:{self.macro_id}:{self.timestamp.isoformat()}"
        object.__setattr__(self, 'change_hash', hashlib.md5(change_data.encode()).hexdigest())


@dataclass
class SyncConfiguration:
    """Configuration for synchronization behavior."""
    # Polling intervals
    base_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    fast_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5))
    slow_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(120))
    
    # File monitoring
    enable_file_monitoring: bool = True
    km_prefs_path: Optional[Path] = None
    
    # Change detection
    change_batch_size: int = 10
    change_batch_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(2))
    
    # Performance limits
    max_concurrent_updates: int = 5
    cache_ttl: Duration = field(default_factory=lambda: Duration.from_minutes(10))
    max_memory_cache_size: int = 1000
    
    # Health monitoring
    health_check_interval: Duration = field(default_factory=lambda: Duration.from_minutes(5))
    max_consecutive_errors: int = 3


@dataclass
class SyncState:
    """Current state of synchronization system."""
    status: SyncStatus = SyncStatus.INITIALIZING
    last_full_sync: Optional[datetime] = None
    last_change_detected: Optional[datetime] = None
    total_changes_processed: int = 0
    consecutive_errors: int = 0
    current_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    active_listeners: Set[str] = field(default_factory=set)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class MacroSyncManager:
    """Real-time macro library synchronization manager."""
    
    def __init__(
        self, 
        km_client: KMClient,
        metadata_extractor: MacroMetadataExtractor,
        config: Optional[SyncConfiguration] = None
    ):
        self.km_client = km_client
        self.metadata_extractor = metadata_extractor
        self.config = config or SyncConfiguration()
        
        # State management
        self.sync_state = SyncState()
        self._macro_cache: Dict[MacroId, EnhancedMacroMetadata] = {}
        self._macro_hashes: Dict[MacroId, str] = {}
        self._change_queue: asyncio.Queue[MacroChange] = asyncio.Queue()
        self._change_listeners: List[Callable[[MacroChange], None]] = []
        
        # Synchronization control
        self._sync_task: Optional[asyncio.Task] = None
        self._file_monitor_task: Optional[asyncio.Task] = None
        self._change_processor_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Performance tracking
        self._performance_tracker = PerformanceTracker()
    
    async def start_sync(self) -> Either[KMError, bool]:
        """Start real-time synchronization."""
        try:
            logger.info("Starting macro synchronization manager")
            
            # Initialize with full sync
            initial_sync_result = await self._perform_full_sync()
            if initial_sync_result.is_left():
                return initial_sync_result
            
            # Start background tasks
            self._sync_task = asyncio.create_task(self._sync_loop())
            self._change_processor_task = asyncio.create_task(self._process_changes())
            self._health_monitor_task = asyncio.create_task(self._health_monitor())
            
            if self.config.enable_file_monitoring:
                self._file_monitor_task = asyncio.create_task(self._file_monitor())
            
            self.sync_state.status = SyncStatus.ACTIVE
            logger.info("Macro synchronization manager started successfully")
            
            return Either.right(True)
            
        except Exception as e:
            logger.exception("Failed to start synchronization manager")
            self.sync_state.status = SyncStatus.ERROR
            return Either.left(KMError.system_error(f"Sync start failed: {str(e)}"))
    
    async def stop_sync(self):
        """Stop real-time synchronization."""
        logger.info("Stopping macro synchronization manager")
        
        # Signal stop to all tasks
        self._stop_event.set()
        
        # Cancel all background tasks
        tasks = [
            self._sync_task,
            self._file_monitor_task,
            self._change_processor_task,
            self._health_monitor_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
        
        self.sync_state.status = SyncStatus.STOPPED
        logger.info("Macro synchronization manager stopped")
    
    def register_change_listener(self, listener: Callable[[MacroChange], None]):
        """Register a listener for macro change events."""
        listener_id = f"listener_{len(self._change_listeners)}"
        self._change_listeners.append(listener)
        self.sync_state.active_listeners.add(listener_id)
        return listener_id
    
    def unregister_change_listener(self, listener: Callable[[MacroChange], None]):
        """Unregister a change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    async def get_cached_macro(self, macro_id: MacroId) -> Optional[EnhancedMacroMetadata]:
        """Get macro from cache if available and current."""
        return self._macro_cache.get(macro_id)
    
    async def force_sync(self) -> Either[KMError, int]:
        """Force immediate full synchronization."""
        logger.info("Forcing full macro synchronization")
        return await self._perform_full_sync()
    
    def get_sync_status(self) -> SyncState:
        """Get current synchronization status."""
        return self.sync_state
    
    async def _sync_loop(self):
        """Main synchronization loop."""
        while not self._stop_event.is_set():
            try:
                await self._perform_incremental_sync()
                await asyncio.sleep(self.sync_state.current_poll_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in sync loop: {e}")
                self.sync_state.consecutive_errors += 1
                
                if self.sync_state.consecutive_errors >= self.config.max_consecutive_errors:
                    self.sync_state.status = SyncStatus.ERROR
                    await asyncio.sleep(60)  # Back off on persistent errors
                else:
                    await asyncio.sleep(self.sync_state.current_poll_interval.total_seconds())
    
    async def _perform_full_sync(self) -> Either[KMError, int]:
        """Perform complete macro library synchronization."""
        start_time = time.perf_counter()
        
        try:
            # Get all macros from KM
            macros_result = await self.km_client.list_macros_async()
            if macros_result.is_left():
                return macros_result.map(lambda _: 0)
            
            basic_macros = macros_result.get_right()
            
            # Extract enhanced metadata for all macros
            enhanced_macros = []
            for basic_macro in basic_macros:
                macro_id = MacroId(basic_macro["id"])
                
                metadata_result = await self.metadata_extractor.extract_enhanced_metadata(macro_id)
                if metadata_result.is_right():
                    enhanced_macros.append(metadata_result.get_right())
            
            # Update cache
            old_cache = self._macro_cache.copy()
            self._macro_cache.clear()
            self._macro_hashes.clear()
            
            for macro in enhanced_macros:
                self._macro_cache[macro.id] = macro
                self._macro_hashes[macro.id] = self._calculate_macro_hash(macro)
            
            # Generate change events for differences
            await self._generate_change_events(old_cache, self._macro_cache)
            
            # Update sync state
            self.sync_state.last_full_sync = datetime.utcnow()
            self.sync_state.consecutive_errors = 0
            
            # Update performance metrics
            sync_time = time.perf_counter() - start_time
            self._performance_tracker.record_sync_time(sync_time)
            
            logger.info(f"Full sync completed: {len(enhanced_macros)} macros in {sync_time:.2f}s")
            return Either.right(len(enhanced_macros))
            
        except Exception as e:
            logger.exception(f"Full sync failed: {e}")
            return Either.left(KMError.system_error(f"Full sync failed: {str(e)}"))
    
    async def _perform_incremental_sync(self):
        """Perform incremental synchronization to detect changes."""
        if not self._macro_cache:
            await self._perform_full_sync()
            return
        
        start_time = time.perf_counter()
        
        try:
            # Quick check for macro count changes
            macros_result = await self.km_client.list_macros_async()
            if macros_result.is_left():
                logger.warning("Incremental sync failed - KM connection error")
                return
            
            current_macros = macros_result.get_right()
            current_ids = {MacroId(m["id"]) for m in current_macros}
            cached_ids = set(self._macro_cache.keys())
            
            # Check for additions/removals
            added_ids = current_ids - cached_ids
            removed_ids = cached_ids - current_ids
            
            # Process additions
            for macro_id in added_ids:
                await self._process_added_macro(macro_id)
            
            # Process removals
            for macro_id in removed_ids:
                await self._process_removed_macro(macro_id)
            
            # Check for modifications in existing macros (sample-based for performance)
            existing_ids = current_ids & cached_ids
            sample_size = min(10, len(existing_ids))  # Check up to 10 macros per cycle
            
            if existing_ids:
                import random
                sample_ids = random.sample(list(existing_ids), min(sample_size, len(existing_ids)))
                
                for macro_id in sample_ids:
                    await self._check_macro_modification(macro_id)
            
            # Adjust polling interval based on activity
            if added_ids or removed_ids:
                self.sync_state.current_poll_interval = self.config.fast_poll_interval
            else:
                # Gradually increase interval if no changes
                current_interval = self.sync_state.current_poll_interval.total_seconds()
                max_interval = self.config.slow_poll_interval.total_seconds()
                new_interval = min(current_interval * 1.1, max_interval)
                self.sync_state.current_poll_interval = Duration.from_seconds(new_interval)
            
            # Update performance metrics
            sync_time = time.perf_counter() - start_time
            self._performance_tracker.record_incremental_sync_time(sync_time)
            
        except Exception as e:
            logger.exception(f"Incremental sync error: {e}")
    
    async def _process_changes(self):
        """Process change events in batches."""
        change_batch = []
        
        while not self._stop_event.is_set():
            try:
                # Collect changes for batch processing
                try:
                    timeout = self.config.change_batch_timeout.total_seconds()
                    change = await asyncio.wait_for(self._change_queue.get(), timeout=timeout)
                    change_batch.append(change)
                    
                    # Continue collecting until batch size or timeout
                    while len(change_batch) < self.config.change_batch_size:
                        try:
                            change = await asyncio.wait_for(self._change_queue.get(), timeout=0.1)
                            change_batch.append(change)
                        except asyncio.TimeoutError:
                            break
                    
                except asyncio.TimeoutError:
                    if not change_batch:
                        continue
                
                # Process batch
                if change_batch:
                    await self._notify_change_listeners(change_batch)
                    self.sync_state.total_changes_processed += len(change_batch)
                    change_batch.clear()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error processing changes: {e}")
    
    async def _notify_change_listeners(self, changes: List[MacroChange]):
        """Notify all registered listeners of changes."""
        for change in changes:
            for listener in self._change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(change)
                    else:
                        listener(change)
                except Exception as e:
                    logger.exception(f"Error in change listener: {e}")
    
    def _calculate_macro_hash(self, macro: EnhancedMacroMetadata) -> str:
        """Calculate hash for macro state to detect changes."""
        state_data = {
            "name": macro.name,
            "enabled": macro.enabled,
            "group": macro.group,
            "trigger_count": len(macro.triggers),
            "action_count": len(macro.actions),
            "last_analyzed": macro.last_analyzed.isoformat()
        }
        state_json = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_json.encode()).hexdigest()


class PerformanceTracker:
    """Track synchronization performance metrics."""
    
    def __init__(self):
        self.sync_times: List[float] = []
        self.incremental_sync_times: List[float] = []
        self.max_history = 100
    
    def record_sync_time(self, sync_time: float):
        """Record full sync time."""
        self.sync_times.append(sync_time)
        if len(self.sync_times) > self.max_history:
            self.sync_times.pop(0)
    
    def record_incremental_sync_time(self, sync_time: float):
        """Record incremental sync time."""
        self.incremental_sync_times.append(sync_time)
        if len(self.incremental_sync_times) > self.max_history:
            self.incremental_sync_times.pop(0)
    
    def get_average_sync_time(self) -> float:
        """Get average full sync time."""
        return sum(self.sync_times) / len(self.sync_times) if self.sync_times else 0.0
    
    def get_average_incremental_sync_time(self) -> float:
        """Get average incremental sync time."""
        return sum(self.incremental_sync_times) / len(self.incremental_sync_times) if self.incremental_sync_times else 0.0
```

#### src/integration/file_monitor.py - File System Change Detection
```python
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

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class FileChangeEvent:
    """Represents a file system change event."""
    file_path: Path
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float
    is_directory: bool = False


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
    
    def start_monitoring(self) -> bool:
        """Start file system monitoring."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("File monitoring unavailable - watchdog package not installed")
            return False
        
        if not self.km_prefs_path.exists():
            logger.warning(f"KM preferences path not found: {self.km_prefs_path}")
            return False
        
        try:
            self._observer = Observer()
            
            # Monitor preferences directory
            event_handler = KMFileEventHandler(self._on_file_change)
            self._observer.schedule(event_handler, str(self.km_prefs_path), recursive=True)
            
            # Monitor macros directory if different
            if self.km_macros_path != self.km_prefs_path and self.km_macros_path.exists():
                self._observer.schedule(event_handler, str(self.km_macros_path), recursive=True)
            
            self._observer.start()
            logger.info(f"File monitoring started for: {self.km_prefs_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to start file monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("File monitoring stopped")
    
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
        except Exception as e:
            logger.exception(f"Error in file change callback: {e}")
    
    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if file change is relevant to macro library."""
        relevant_extensions = {'.kmmacros', '.plist', '.json'}
        relevant_names = {'Keyboard Maestro Preferences.plist', 'macros.json'}
        
        return (
            file_path.suffix.lower() in relevant_extensions or
            file_path.name in relevant_names or
            'macro' in file_path.name.lower()
        )
    
    def _find_km_preferences_path(self) -> Path:
        """Find Keyboard Maestro preferences directory."""
        home = Path.home()
        possible_paths = [
            home / "Library" / "Preferences" / "com.stairways.keyboardmaestro.engine.plist",
            home / "Library" / "Application Support" / "Keyboard Maestro",
            Path("/Library/Application Support/Keyboard Maestro")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path.parent if path.is_file() else path
        
        # Default fallback
        return home / "Library" / "Application Support" / "Keyboard Maestro"
    
    def _find_km_macros_path(self) -> Path:
        """Find Keyboard Maestro macros directory."""
        # Usually same as preferences for modern KM versions
        return self.km_prefs_path


if WATCHDOG_AVAILABLE:
    class KMFileEventHandler(FileSystemEventHandler):
        """Handle file system events for KM files."""
        
        def __init__(self, callback: Callable[[FileSystemEvent], None]):
            self.callback = callback
        
        def on_modified(self, event):
            self.callback(event)
        
        def on_created(self, event):
            self.callback(event)
        
        def on_deleted(self, event):
            self.callback(event)
        
        def on_moved(self, event):
            self.callback(event)
```

## 🏗️ Modularity Strategy
- **sync_manager.py**: Main synchronization orchestration (target: 250 lines)
- **file_monitor.py**: File system change detection (target: 150 lines)  
- **change_detector.py**: Smart change detection algorithms (target: 125 lines)
- **cache_manager.py**: Intelligent caching and invalidation (target: 100 lines)
- **performance_monitor.py**: Sync performance tracking and optimization (target: 75 lines)

## ✅ Success Criteria
- Real-time macro state updates without performance degradation
- Intelligent polling adapts to user activity and change frequency
- File system monitoring provides immediate change detection
- Efficient caching minimizes redundant API calls and processing
- Change events enable reactive UI updates in AI clients
- Performance monitoring ensures sync overhead stays under 1% CPU usage
- Graceful degradation when Keyboard Maestro unavailable
- Smart batching prevents overwhelming listeners with rapid changes
- Delta synchronization updates only changed macros for efficiency
- Health monitoring detects and recovers from sync issues automatically
- Configuration allows tuning for different usage patterns
- Comprehensive logging enables debugging and optimization