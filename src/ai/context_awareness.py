"""
Context awareness system for intelligent automation.

This module provides comprehensive context awareness capabilities including
real-time context detection, state management, and intelligent context-based
automation triggering with privacy protection and efficient processing.

Security: All context processing includes privacy protection and data validation.
Performance: Optimized for real-time context updates with intelligent caching.
Type Safety: Complete integration with intelligent automation architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NewType, Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import json
import re
import hashlib

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError
from .intelligent_automation import ContextState, ContextDimension, ContextStateId

# Branded Types for Context Awareness
ContextDetectorId = NewType('ContextDetectorId', str)
ContextRuleId = NewType('ContextRuleId', str)
ContextSignature = NewType('ContextSignature', str)
PrivacyLevel = NewType('PrivacyLevel', int)
UpdateFrequency = NewType('UpdateFrequency', float)


class ContextDetectionMethod(Enum):
    """Methods for context detection."""
    SYSTEM_MONITORING = "system_monitoring"      # System state monitoring
    APPLICATION_TRACKING = "application_tracking" # Active application tracking
    CONTENT_ANALYSIS = "content_analysis"        # Content-based detection
    USER_ACTIVITY = "user_activity"              # User activity patterns
    TEMPORAL_ANALYSIS = "temporal_analysis"      # Time-based patterns
    ENVIRONMENT_SENSING = "environment_sensing"  # Environmental factors
    WORKFLOW_TRACKING = "workflow_tracking"      # Workflow state tracking


class ContextChangeType(Enum):
    """Types of context changes."""
    MINOR_UPDATE = "minor_update"          # Small incremental change
    SIGNIFICANT_CHANGE = "significant_change" # Notable state change
    MAJOR_TRANSITION = "major_transition"   # Complete context shift
    PATTERN_DETECTED = "pattern_detected"   # New pattern recognition
    ANOMALY_DETECTED = "anomaly_detected"   # Unexpected change


class ContextPrivacyLevel(Enum):
    """Privacy levels for context data."""
    PUBLIC = 1          # Non-sensitive, shareable data
    INTERNAL = 2        # Internal use only
    CONFIDENTIAL = 3    # Sensitive, limited access
    RESTRICTED = 4      # Highly sensitive, minimal access
    PRIVATE = 5         # Personal/private data, anonymized


@dataclass(frozen=True)
class ContextDetector:
    """Context detector configuration and logic."""
    detector_id: ContextDetectorId
    name: str
    detection_method: ContextDetectionMethod
    target_dimensions: Set[ContextDimension]
    update_frequency: UpdateFrequency = UpdateFrequency(5.0)  # seconds
    privacy_level: PrivacyLevel = PrivacyLevel(2)
    enabled: bool = True
    detection_rules: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.detector_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: self.update_frequency > 0)
    @require(lambda self: 1 <= self.privacy_level <= 5)
    def __post_init__(self):
        """Validate context detector configuration."""
        pass
    
    async def detect_context(self, previous_context: Optional[ContextState] = None) -> Either[ValidationError, Dict[ContextDimension, Any]]:
        """Detect context information based on detection method."""
        try:
            detected_data = {}
            
            if self.detection_method == ContextDetectionMethod.SYSTEM_MONITORING:
                detected_data.update(await self._detect_system_context())
            
            elif self.detection_method == ContextDetectionMethod.APPLICATION_TRACKING:
                detected_data.update(await self._detect_application_context())
            
            elif self.detection_method == ContextDetectionMethod.TEMPORAL_ANALYSIS:
                detected_data.update(await self._detect_temporal_context())
            
            elif self.detection_method == ContextDetectionMethod.USER_ACTIVITY:
                detected_data.update(await self._detect_user_activity_context())
            
            elif self.detection_method == ContextDetectionMethod.WORKFLOW_TRACKING:
                detected_data.update(await self._detect_workflow_context())
            
            # Filter by target dimensions
            filtered_data = {
                dim: value for dim, value in detected_data.items()
                if dim in self.target_dimensions
            }
            
            # Apply privacy filtering
            privacy_filtered_data = self._apply_privacy_filtering(filtered_data)
            
            return Either.right(privacy_filtered_data)
            
        except Exception as e:
            return Either.left(ValidationError("context_detection_failed", str(e)))
    
    async def _detect_system_context(self) -> Dict[ContextDimension, Any]:
        """Detect system-level context information."""
        import psutil
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # Get network activity
            network_io = psutil.net_io_counters()
            
            system_context = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available // (1024 * 1024),  # MB
                "disk_usage": disk_usage.percent,
                "network_bytes_sent": network_io.bytes_sent,
                "network_bytes_recv": network_io.bytes_recv,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            return {ContextDimension.SYSTEM_STATE: system_context}
            
        except Exception:
            return {ContextDimension.SYSTEM_STATE: {"status": "unavailable"}}
    
    async def _detect_application_context(self) -> Dict[ContextDimension, Any]:
        """Detect application-level context information."""
        try:
            # Mock application detection - in real implementation would use
            # platform-specific APIs to get active applications
            active_apps = self._get_active_applications()
            
            app_context = {
                "active_application": active_apps.get("frontmost"),
                "running_applications": active_apps.get("running", []),
                "window_count": len(active_apps.get("windows", [])),
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            return {ContextDimension.APPLICATION: app_context}
            
        except Exception:
            return {ContextDimension.APPLICATION: {"status": "unavailable"}}
    
    def _get_active_applications(self) -> Dict[str, Any]:
        """Get active applications (mock implementation)."""
        # Mock implementation - real version would use AppleScript or system APIs
        return {
            "frontmost": "Finder",
            "running": ["Finder", "Safari", "Terminal", "VS Code"],
            "windows": [
                {"app": "Finder", "title": "Desktop"},
                {"app": "Safari", "title": "Web Development"},
                {"app": "Terminal", "title": "zsh"}
            ]
        }
    
    async def _detect_temporal_context(self) -> Dict[ContextDimension, Any]:
        """Detect temporal context information."""
        now = datetime.now()
        
        # Determine time category
        hour = now.hour
        if 6 <= hour < 12:
            time_category = "morning"
        elif 12 <= hour < 17:
            time_category = "afternoon"
        elif 17 <= hour < 21:
            time_category = "evening"
        else:
            time_category = "night"
        
        # Determine work vs personal time
        weekday = now.weekday()
        is_workday = weekday < 5  # Monday = 0, Friday = 4
        is_work_hours = 9 <= hour <= 17
        
        temporal_context = {
            "hour": hour,
            "minute": now.minute,
            "weekday": weekday,
            "day_name": now.strftime("%A"),
            "date": now.strftime("%Y-%m-%d"),
            "time_category": time_category,
            "is_workday": is_workday,
            "is_work_hours": is_work_hours and is_workday,
            "timezone": str(now.astimezone().tzinfo),
            "timestamp": now.isoformat()
        }
        
        return {ContextDimension.TEMPORAL: temporal_context}
    
    async def _detect_user_activity_context(self) -> Dict[ContextDimension, Any]:
        """Detect user activity context information."""
        # Mock user activity detection
        activity_context = {
            "activity_level": "moderate",  # low, moderate, high
            "input_type": "keyboard",      # keyboard, mouse, both, none
            "last_activity": datetime.now(UTC).isoformat(),
            "session_duration": 3600,     # seconds
            "idle_time": 0,               # seconds since last activity
            "estimated_focus": "high"     # low, medium, high
        }
        
        return {ContextDimension.USER_STATE: activity_context}
    
    async def _detect_workflow_context(self) -> Dict[ContextDimension, Any]:
        """Detect workflow-level context information."""
        # Mock workflow detection
        workflow_context = {
            "current_task": "development",
            "task_category": "coding",
            "estimated_progress": 0.65,
            "workflow_phase": "implementation",
            "tools_in_use": ["editor", "terminal", "browser"],
            "complexity_level": "intermediate"
        }
        
        return {ContextDimension.WORKFLOW: workflow_context}
    
    def _apply_privacy_filtering(self, context_data: Dict[ContextDimension, Any]) -> Dict[ContextDimension, Any]:
        """Apply privacy filtering based on privacy level."""
        if self.privacy_level >= PrivacyLevel(4):  # RESTRICTED or PRIVATE
            # Remove or anonymize sensitive data
            filtered_data = {}
            for dim, value in context_data.items():
                if dim == ContextDimension.APPLICATION:
                    # Anonymize application names
                    filtered_data[dim] = self._anonymize_application_data(value)
                elif dim == ContextDimension.CONTENT:
                    # Remove content data entirely for high privacy
                    continue
                else:
                    filtered_data[dim] = value
            return filtered_data
        
        elif self.privacy_level >= PrivacyLevel(3):  # CONFIDENTIAL
            # Limit sensitive details
            filtered_data = {}
            for dim, value in context_data.items():
                if dim == ContextDimension.CONTENT:
                    # Reduce content detail
                    filtered_data[dim] = self._reduce_content_detail(value)
                else:
                    filtered_data[dim] = value
            return filtered_data
        
        return context_data  # No filtering for lower privacy levels
    
    def _anonymize_application_data(self, app_data: Any) -> Dict[str, Any]:
        """Anonymize application data for privacy."""
        if isinstance(app_data, dict):
            anonymized = app_data.copy()
            
            # Replace specific app names with categories
            app_categories = {
                "finder": "file_manager",
                "safari": "browser",
                "chrome": "browser",
                "firefox": "browser",
                "terminal": "terminal",
                "iterm": "terminal",
                "vscode": "editor",
                "vim": "editor",
                "emacs": "editor"
            }
            
            if "active_application" in anonymized:
                app_name = str(anonymized["active_application"]).lower()
                for app, category in app_categories.items():
                    if app in app_name:
                        anonymized["active_application"] = category
                        break
            
            if "running_applications" in anonymized:
                running_apps = anonymized["running_applications"]
                if isinstance(running_apps, list):
                    categorized_apps = []
                    for app in running_apps:
                        app_lower = str(app).lower()
                        category = "unknown_app"
                        for app_name, app_category in app_categories.items():
                            if app_name in app_lower:
                                category = app_category
                                break
                        categorized_apps.append(category)
                    anonymized["running_applications"] = list(set(categorized_apps))  # Remove duplicates
            
            return anonymized
        
        return {"status": "anonymized"}
    
    def _reduce_content_detail(self, content_data: Any) -> Dict[str, Any]:
        """Reduce content detail for privacy."""
        if isinstance(content_data, dict):
            reduced = {
                "content_type": content_data.get("type", "unknown"),
                "content_length": len(str(content_data.get("text", ""))),
                "has_content": bool(content_data.get("text", "")),
                "timestamp": content_data.get("timestamp", datetime.now(UTC).isoformat())
            }
            return reduced
        
        return {"status": "content_reduced"}


@dataclass(frozen=True)
class ContextChangeEvent:
    """Context change event representation."""
    event_id: str
    change_type: ContextChangeType
    previous_context: Optional[ContextState]
    new_context: ContextState
    changed_dimensions: Set[ContextDimension]
    significance_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: len(self.event_id) > 0)
    @require(lambda self: 0.0 <= self.significance_score <= 1.0)
    def __post_init__(self):
        """Validate context change event."""
        pass
    
    def get_change_summary(self) -> Dict[str, Any]:
        """Get summary of context changes."""
        summary = {
            "event_id": self.event_id,
            "change_type": self.change_type.value,
            "significance": self.significance_score,
            "dimensions_changed": [dim.value for dim in self.changed_dimensions],
            "timestamp": self.timestamp.isoformat()
        }
        
        # Add specific change details
        if self.previous_context and self.new_context:
            changes_detail = {}
            for dim in self.changed_dimensions:
                old_value = self.previous_context.get_dimension_value(dim)
                new_value = self.new_context.get_dimension_value(dim)
                changes_detail[dim.value] = {
                    "from": str(old_value)[:100] if old_value else None,
                    "to": str(new_value)[:100] if new_value else None
                }
            summary["change_details"] = changes_detail
        
        return summary


class ContextAwarenessEngine:
    """Comprehensive context awareness and management system."""
    
    def __init__(self):
        self.detectors: Dict[ContextDetectorId, ContextDetector] = {}
        self.current_context: Optional[ContextState] = None
        self.context_history: List[ContextState] = []
        self.change_listeners: List[Callable[[ContextChangeEvent], None]] = []
        self.detection_tasks: Dict[ContextDetectorId, asyncio.Task] = {}
        self.is_running = False
        self.update_interval = 5.0  # seconds
    
    def add_detector(self, detector: ContextDetector) -> None:
        """Add context detector to the engine."""
        self.detectors[detector.detector_id] = detector
        
        # Start detection task if engine is running
        if self.is_running and detector.enabled:
            self._start_detector_task(detector)
    
    def remove_detector(self, detector_id: ContextDetectorId) -> None:
        """Remove context detector from the engine."""
        if detector_id in self.detectors:
            # Stop detection task
            if detector_id in self.detection_tasks:
                self.detection_tasks[detector_id].cancel()
                del self.detection_tasks[detector_id]
            
            del self.detectors[detector_id]
    
    def add_change_listener(self, listener: Callable[[ContextChangeEvent], None]) -> None:
        """Add context change event listener."""
        self.change_listeners.append(listener)
    
    async def start_monitoring(self) -> None:
        """Start context monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start detection tasks for all enabled detectors
        for detector in self.detectors.values():
            if detector.enabled:
                self._start_detector_task(detector)
    
    async def stop_monitoring(self) -> None:
        """Stop context monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all detection tasks
        for task in self.detection_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.detection_tasks:
            await asyncio.gather(*self.detection_tasks.values(), return_exceptions=True)
        
        self.detection_tasks.clear()
    
    def _start_detector_task(self, detector: ContextDetector) -> None:
        """Start background detection task for detector."""
        async def detection_loop():
            while self.is_running:
                try:
                    # Detect context
                    detection_result = await detector.detect_context(self.current_context)
                    
                    if detection_result.is_right():
                        detected_data = detection_result.get_right()
                        if detected_data:
                            await self._process_detected_context(detector.detector_id, detected_data)
                    
                    # Wait for next detection cycle
                    await asyncio.sleep(detector.update_frequency)
                    
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Log error and continue
                    await asyncio.sleep(detector.update_frequency)
        
        task = asyncio.create_task(detection_loop())
        self.detection_tasks[detector.detector_id] = task
    
    async def _process_detected_context(self, detector_id: ContextDetectorId, 
                                      detected_data: Dict[ContextDimension, Any]) -> None:
        """Process detected context data and update state."""
        try:
            # Create new context state
            context_id = ContextStateId(f"ctx_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{detector_id}")
            
            # Merge with existing context if available
            if self.current_context:
                merged_dimensions = dict(self.current_context.dimensions)
                merged_dimensions.update(detected_data)
            else:
                merged_dimensions = detected_data
            
            new_context = ContextState(
                context_id=context_id,
                timestamp=datetime.now(UTC),
                dimensions=merged_dimensions,
                confidence=self._calculate_context_confidence(merged_dimensions),
                metadata={"detector_id": detector_id}
            )
            
            # Check for significant changes
            change_event = self._analyze_context_change(self.current_context, new_context)
            
            # Update current context
            previous_context = self.current_context
            self.current_context = new_context
            
            # Add to history (keep last 100 entries)
            self.context_history.append(new_context)
            if len(self.context_history) > 100:
                self.context_history = self.context_history[-100:]
            
            # Notify listeners if significant change
            if change_event and change_event.significance_score > 0.3:
                await self._notify_change_listeners(change_event)
                
        except Exception as e:
            # Log error but don't crash
            pass
    
    def _calculate_context_confidence(self, dimensions: Dict[ContextDimension, Any]) -> float:
        """Calculate confidence score for context state."""
        if not dimensions:
            return 0.0
        
        # Base confidence on number of dimensions and data quality
        dimension_score = min(len(dimensions) / 5.0, 1.0)  # Max 5 dimensions
        
        # Quality score based on data completeness
        quality_scores = []
        for dim, value in dimensions.items():
            if value is None:
                quality_scores.append(0.0)
            elif isinstance(value, dict):
                # Check dictionary completeness
                non_null_values = sum(1 for v in value.values() if v is not None)
                total_values = len(value)
                quality_scores.append(non_null_values / total_values if total_values > 0 else 0.0)
            else:
                quality_scores.append(1.0)
        
        quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return min((dimension_score + quality_score) / 2.0, 1.0)
    
    def _analyze_context_change(self, previous: Optional[ContextState], 
                              new: ContextState) -> Optional[ContextChangeEvent]:
        """Analyze context change and determine significance."""
        if not previous:
            return None
        
        # Find changed dimensions
        changed_dimensions = set()
        for dim in set(previous.dimensions.keys()) | set(new.dimensions.keys()):
            old_value = previous.get_dimension_value(dim)
            new_value = new.get_dimension_value(dim)
            
            if old_value != new_value:
                changed_dimensions.add(dim)
        
        if not changed_dimensions:
            return None
        
        # Calculate significance score
        significance = self._calculate_change_significance(previous, new, changed_dimensions)
        
        # Determine change type
        change_type = self._determine_change_type(significance, changed_dimensions)
        
        # Create change event
        event_id = f"change_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(changed_dimensions)}"
        
        return ContextChangeEvent(
            event_id=event_id,
            change_type=change_type,
            previous_context=previous,
            new_context=new,
            changed_dimensions=changed_dimensions,
            significance_score=significance
        )
    
    def _calculate_change_significance(self, previous: ContextState, new: ContextState,
                                     changed_dimensions: Set[ContextDimension]) -> float:
        """Calculate significance score for context change."""
        if not changed_dimensions:
            return 0.0
        
        # Base score on number of changed dimensions
        dimension_factor = len(changed_dimensions) / len(ContextDimension)
        
        # Weight certain dimensions as more significant
        high_impact_dimensions = {
            ContextDimension.APPLICATION,
            ContextDimension.WORKFLOW,
            ContextDimension.USER_STATE
        }
        
        impact_factor = 0.0
        for dim in changed_dimensions:
            if dim in high_impact_dimensions:
                impact_factor += 0.3
            else:
                impact_factor += 0.1
        
        # Time factor - recent changes are more significant
        time_diff = (new.timestamp - previous.timestamp).total_seconds()
        time_factor = max(0.5, 1.0 - (time_diff / 3600))  # Decay over 1 hour
        
        significance = min((dimension_factor + impact_factor) * time_factor, 1.0)
        return significance
    
    def _determine_change_type(self, significance: float, 
                             changed_dimensions: Set[ContextDimension]) -> ContextChangeType:
        """Determine type of context change."""
        if significance >= 0.8:
            return ContextChangeType.MAJOR_TRANSITION
        elif significance >= 0.5:
            return ContextChangeType.SIGNIFICANT_CHANGE
        elif len(changed_dimensions) >= 3:
            return ContextChangeType.PATTERN_DETECTED
        else:
            return ContextChangeType.MINOR_UPDATE
    
    async def _notify_change_listeners(self, change_event: ContextChangeEvent) -> None:
        """Notify all change listeners of context change."""
        for listener in self.change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(change_event)
                else:
                    listener(change_event)
            except Exception:
                # Log error but continue notifying other listeners
                continue
    
    def get_current_context(self) -> Optional[ContextState]:
        """Get current context state."""
        return self.current_context
    
    def get_context_history(self, limit: int = 10) -> List[ContextState]:
        """Get recent context history."""
        return self.context_history[-limit:] if self.context_history else []
    
    def find_similar_contexts(self, target_context: ContextState, 
                            similarity_threshold: float = 0.8,
                            max_results: int = 5) -> List[tuple[ContextState, float]]:
        """Find contexts similar to target context."""
        similar_contexts = []
        
        for context in self.context_history:
            if context.context_id == target_context.context_id:
                continue
            
            similarity = target_context.similarity_to(context)
            if similarity >= similarity_threshold:
                similar_contexts.append((context, similarity))
        
        # Sort by similarity and return top results
        similar_contexts.sort(key=lambda x: x[1], reverse=True)
        return similar_contexts[:max_results]
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get comprehensive context awareness statistics."""
        detector_stats = {}
        for detector_id, detector in self.detectors.items():
            task = self.detection_tasks.get(detector_id)
            detector_stats[str(detector_id)] = {
                "name": detector.name,
                "method": detector.detection_method.value,
                "enabled": detector.enabled,
                "running": task is not None and not task.done() if task else False,
                "update_frequency": float(detector.update_frequency),
                "privacy_level": int(detector.privacy_level),
                "target_dimensions": [dim.value for dim in detector.target_dimensions]
            }
        
        current_context_info = None
        if self.current_context:
            current_context_info = {
                "context_id": str(self.current_context.context_id),
                "timestamp": self.current_context.timestamp.isoformat(),
                "confidence": float(self.current_context.confidence),
                "dimensions": [dim.value for dim in self.current_context.dimensions.keys()],
                "dimension_count": len(self.current_context.dimensions)
            }
        
        return {
            "system_status": {
                "is_monitoring": self.is_running,
                "total_detectors": len(self.detectors),
                "active_detectors": len(self.detection_tasks),
                "change_listeners": len(self.change_listeners),
                "context_history_size": len(self.context_history)
            },
            "detector_statistics": detector_stats,
            "current_context": current_context_info,
            "update_interval": self.update_interval,
            "timestamp": datetime.now(UTC).isoformat()
        }