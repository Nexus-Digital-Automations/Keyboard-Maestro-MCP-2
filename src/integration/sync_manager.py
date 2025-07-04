"""
Real-time Macro State Synchronization Manager

Provides real-time monitoring and synchronization of Keyboard Maestro
macro library state with intelligent caching and change detection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import hashlib
import json
import logging
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "change_type": self.change_type.value,
            "macro_id": str(self.macro_id) if self.macro_id else None,
            "macro_name": self.macro_name,
            "group_name": self.group_name,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SyncConfiguration:
    """Configuration for synchronization behavior."""
    # Polling intervals
    base_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(30))
    fast_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(5))
    slow_poll_interval: Duration = field(default_factory=lambda: Duration.from_seconds(120))
    
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
        self._change_processor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Performance tracking
        self._sync_times: List[float] = []
        self._last_activity_check = datetime.now(UTC)
    
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
            
            self.sync_state.status = SyncStatus.ACTIVE
            logger.info("Macro synchronization manager started successfully")
            
            return Either.right(True)
            
        except Exception as e:
            logger.exception("Failed to start synchronization manager")
            self.sync_state.status = SyncStatus.ERROR
            return Either.left(KMError.execution_error(f"Sync start failed: {str(e)}"))
    
    async def stop_sync(self):
        """Stop real-time synchronization."""
        logger.info("Stopping macro synchronization manager")
        
        # Signal stop to all tasks
        self._stop_event.set()
        
        # Cancel all background tasks
        tasks = [self._sync_task, self._change_processor_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
        
        self.sync_state.status = SyncStatus.STOPPED
        logger.info("Macro synchronization manager stopped")
    
    def register_change_listener(self, listener: Callable[[MacroChange], None]) -> str:
        """Register a listener for macro change events."""
        listener_id = f"listener_{len(self._change_listeners)}"
        self._change_listeners.append(listener)
        self.sync_state.active_listeners.add(listener_id)
        logger.info(f"Registered change listener: {listener_id}")
        return listener_id
    
    def unregister_change_listener(self, listener: Callable[[MacroChange], None]):
        """Unregister a change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            logger.info("Unregistered change listener")
    
    async def get_cached_macro(self, macro_id: MacroId) -> Optional[EnhancedMacroMetadata]:
        """Get macro from cache if available and current."""
        return self._macro_cache.get(macro_id)
    
    async def force_sync(self) -> Either[KMError, int]:
        """Force immediate full synchronization."""
        logger.info("Forcing full macro synchronization")
        return await self._perform_full_sync()
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        return {
            "status": self.sync_state.status.value,
            "last_full_sync": self.sync_state.last_full_sync.isoformat() if self.sync_state.last_full_sync else None,
            "last_change_detected": self.sync_state.last_change_detected.isoformat() if self.sync_state.last_change_detected else None,
            "total_changes_processed": self.sync_state.total_changes_processed,
            "consecutive_errors": self.sync_state.consecutive_errors,
            "current_poll_interval_seconds": self.sync_state.current_poll_interval.total_seconds(),
            "active_listeners": len(self._change_listeners),
            "cached_macros": len(self._macro_cache),
            "average_sync_time_seconds": sum(self._sync_times) / len(self._sync_times) if self._sync_times else 0.0
        }
    
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
            macros_result = await self.km_client.list_macros_async(
                group_filters=None,  # Get macros from all groups for sync
                enabled_only=False   # Include disabled macros for complete sync
            )
            if macros_result.is_left():
                return macros_result.map(lambda _: 0)
            
            basic_macros = macros_result.get_right()
            
            # Extract enhanced metadata for all macros
            enhanced_macros = []
            for basic_macro in basic_macros:
                macro_id = MacroId(basic_macro["id"])
                
                # Use cached metadata if available and recent
                cached_macro = self._macro_cache.get(macro_id)
                if cached_macro and self._is_cache_valid(cached_macro):
                    enhanced_macros.append(cached_macro)
                else:
                    metadata_result = await self.metadata_extractor.extract_enhanced_metadata(macro_id)
                    if metadata_result.is_right():
                        enhanced_macros.append(metadata_result.get_right())
            
            # Update cache and detect changes
            old_cache = self._macro_cache.copy()
            self._macro_cache.clear()
            self._macro_hashes.clear()
            
            for macro in enhanced_macros:
                self._macro_cache[macro.id] = macro
                self._macro_hashes[macro.id] = self._calculate_macro_hash(macro)
            
            # Generate change events for differences
            await self._generate_change_events(old_cache, self._macro_cache)
            
            # Update sync state
            self.sync_state.last_full_sync = datetime.now(UTC)
            self.sync_state.consecutive_errors = 0
            
            # Update performance metrics
            sync_time = time.perf_counter() - start_time
            self._sync_times.append(sync_time)
            if len(self._sync_times) > 50:  # Keep last 50 sync times
                self._sync_times.pop(0)
            
            logger.info(f"Full sync completed: {len(enhanced_macros)} macros in {sync_time:.2f}s")
            return Either.right(len(enhanced_macros))
            
        except Exception as e:
            logger.exception(f"Full sync failed: {e}")
            return Either.left(KMError.execution_error(f"Full sync failed: {str(e)}"))
    
    async def _perform_incremental_sync(self):
        """Perform incremental synchronization to detect changes."""
        if not self._macro_cache:
            await self._perform_full_sync()
            return
        
        start_time = time.perf_counter()
        
        try:
            # Quick check for macro count changes
            macros_result = await self.km_client.list_macros_async(
                group_filters=None,  # Get macros from all groups for incremental sync
                enabled_only=False   # Include disabled macros for complete count
            )
            if macros_result.is_left():
                logger.warning("Incremental sync failed - KM connection error")
                self.sync_state.consecutive_errors += 1
                return
            
            current_macros = macros_result.get_right()
            current_ids = {MacroId(m["id"]) for m in current_macros}
            cached_ids = set(self._macro_cache.keys())
            
            # Check for additions/removals
            added_ids = current_ids - cached_ids
            removed_ids = cached_ids - current_ids
            
            changes_detected = bool(added_ids or removed_ids)
            
            # Process additions
            for macro_id in added_ids:
                await self._process_added_macro(macro_id)
                changes_detected = True
            
            # Process removals
            for macro_id in removed_ids:
                await self._process_removed_macro(macro_id)
                changes_detected = True
            
            # Check for modifications in existing macros (sample-based for performance)
            existing_ids = current_ids & cached_ids
            if existing_ids:
                # Check a few random macros each cycle
                import random
                sample_size = min(5, len(existing_ids))
                sample_ids = random.sample(list(existing_ids), sample_size)
                
                for macro_id in sample_ids:
                    if await self._check_macro_modification(macro_id):
                        changes_detected = True
            
            # Adjust polling interval based on activity
            if changes_detected:
                self.sync_state.current_poll_interval = self.config.fast_poll_interval
                self.sync_state.last_change_detected = datetime.now(UTC)
            else:
                # Gradually increase interval if no changes
                current_interval = self.sync_state.current_poll_interval.total_seconds()
                max_interval = self.config.slow_poll_interval.total_seconds()
                new_interval = min(current_interval * 1.2, max_interval)
                self.sync_state.current_poll_interval = Duration.from_seconds(new_interval)
            
            # Reset error count on successful sync
            self.sync_state.consecutive_errors = 0
            
            sync_time = time.perf_counter() - start_time
            if sync_time > 1.0:  # Log slow syncs
                logger.warning(f"Slow incremental sync: {sync_time:.2f}s")
            
        except Exception as e:
            logger.exception(f"Incremental sync error: {e}")
            self.sync_state.consecutive_errors += 1
    
    async def _process_added_macro(self, macro_id: MacroId):
        """Process a newly added macro."""
        try:
            metadata_result = await self.metadata_extractor.extract_enhanced_metadata(macro_id)
            if metadata_result.is_right():
                macro = metadata_result.get_right()
                self._macro_cache[macro_id] = macro
                self._macro_hashes[macro_id] = self._calculate_macro_hash(macro)
                
                change = MacroChange(
                    change_type=ChangeType.MACRO_ADDED,
                    macro_id=macro_id,
                    macro_name=macro.name,
                    group_name=macro.group,
                    new_state={"name": macro.name, "enabled": macro.enabled}
                )
                await self._change_queue.put(change)
                logger.info(f"Detected new macro: {macro.name}")
        except Exception as e:
            logger.exception(f"Error processing added macro {macro_id}: {e}")
    
    async def _process_removed_macro(self, macro_id: MacroId):
        """Process a removed macro."""
        try:
            old_macro = self._macro_cache.pop(macro_id, None)
            self._macro_hashes.pop(macro_id, None)
            
            if old_macro:
                change = MacroChange(
                    change_type=ChangeType.MACRO_REMOVED,
                    macro_id=macro_id,
                    macro_name=old_macro.name,
                    group_name=old_macro.group,
                    old_state={"name": old_macro.name, "enabled": old_macro.enabled}
                )
                await self._change_queue.put(change)
                logger.info(f"Detected removed macro: {old_macro.name}")
        except Exception as e:
            logger.exception(f"Error processing removed macro {macro_id}: {e}")
    
    async def _check_macro_modification(self, macro_id: MacroId) -> bool:
        """Check if a macro has been modified."""
        try:
            metadata_result = await self.metadata_extractor.extract_enhanced_metadata(macro_id)
            if metadata_result.is_left():
                return False
            
            new_macro = metadata_result.get_right()
            new_hash = self._calculate_macro_hash(new_macro)
            old_hash = self._macro_hashes.get(macro_id)
            
            if old_hash != new_hash:
                old_macro = self._macro_cache.get(macro_id)
                self._macro_cache[macro_id] = new_macro
                self._macro_hashes[macro_id] = new_hash
                
                # Determine specific change type
                change_type = ChangeType.MACRO_MODIFIED
                if old_macro and old_macro.enabled != new_macro.enabled:
                    change_type = ChangeType.MACRO_ENABLED if new_macro.enabled else ChangeType.MACRO_DISABLED
                
                change = MacroChange(
                    change_type=change_type,
                    macro_id=macro_id,
                    macro_name=new_macro.name,
                    group_name=new_macro.group,
                    old_state={"enabled": old_macro.enabled} if old_macro else None,
                    new_state={"enabled": new_macro.enabled}
                )
                await self._change_queue.put(change)
                logger.info(f"Detected modified macro: {new_macro.name}")
                return True
                
        except Exception as e:
            logger.exception(f"Error checking macro modification {macro_id}: {e}")
        
        return False
    
    async def _generate_change_events(
        self, 
        old_cache: Dict[MacroId, EnhancedMacroMetadata],
        new_cache: Dict[MacroId, EnhancedMacroMetadata]
    ):
        """Generate change events by comparing cache states."""
        old_ids = set(old_cache.keys())
        new_ids = set(new_cache.keys())
        
        # Added macros
        for macro_id in new_ids - old_ids:
            macro = new_cache[macro_id]
            change = MacroChange(
                change_type=ChangeType.MACRO_ADDED,
                macro_id=macro_id,
                macro_name=macro.name,
                group_name=macro.group
            )
            await self._change_queue.put(change)
        
        # Removed macros
        for macro_id in old_ids - new_ids:
            macro = old_cache[macro_id]
            change = MacroChange(
                change_type=ChangeType.MACRO_REMOVED,
                macro_id=macro_id,
                macro_name=macro.name,
                group_name=macro.group
            )
            await self._change_queue.put(change)
    
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
            "action_count": len(macro.actions)
        }
        state_json = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_json.encode()).hexdigest()
    
    def _is_cache_valid(self, macro: EnhancedMacroMetadata) -> bool:
        """Check if cached macro metadata is still valid."""
        cache_age = datetime.now(UTC) - macro.last_analyzed
        return cache_age < self.config.cache_ttl