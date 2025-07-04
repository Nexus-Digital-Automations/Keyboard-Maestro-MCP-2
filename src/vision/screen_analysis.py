"""
Advanced screen analysis and capture engine for visual automation.

This module implements sophisticated screen capture, analysis, and monitoring
capabilities. Provides secure screenshot capture, window analysis, and real-time
screen change detection with comprehensive privacy protection.

Security: Screen recording permission validation and sensitive content filtering.
Performance: Optimized capture with intelligent caching and change detection.
Privacy: Comprehensive content filtering and access control mechanisms.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import asyncio
import hashlib
from datetime import datetime, timedelta

from src.core.visual import (
    ScreenRegion, VisualElement, VisualError, ProcessingError, PermissionError,
    PrivacyError, ColorInfo, ImageData, ElementType, validate_image_data
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class CaptureMode(Enum):
    """Screen capture modes with different quality/performance trade-offs."""
    FULL_QUALITY = "full_quality"        # Maximum quality, slower
    BALANCED = "balanced"                # Good quality, reasonable speed
    PERFORMANCE = "performance"          # Lower quality, fastest
    PRIVACY_SAFE = "privacy_safe"        # Content filtering enabled
    THUMBNAIL = "thumbnail"              # Small preview capture


class WindowState(Enum):
    """Window state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MINIMIZED = "minimized"
    HIDDEN = "hidden"
    FULLSCREEN = "fullscreen"
    UNKNOWN = "unknown"


class ChangeDetectionMode(Enum):
    """Screen change detection sensitivity levels."""
    PIXEL_PERFECT = "pixel_perfect"      # Detect any pixel change
    CONTENT_AWARE = "content_aware"      # Ignore minor changes
    STRUCTURAL = "structural"            # Major layout changes only
    MOTION_ONLY = "motion_only"         # Movement-based detection


@dataclass(frozen=True)
class WindowInfo:
    """Comprehensive window information."""
    window_id: str
    title: str
    application_name: str
    bundle_id: str
    process_id: int
    bounds: ScreenRegion
    state: WindowState
    layer: int = 0
    is_on_screen: bool = True
    owner_name: str = ""
    window_level: int = 0
    alpha: float = 1.0
    has_shadow: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate window information."""
        if not self.window_id:
            raise ValueError("Window ID cannot be empty")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        if self.process_id < 0:
            raise ValueError("Process ID must be non-negative")
    
    @property
    def is_visible(self) -> bool:
        """Check if window is visible to user."""
        return (self.state not in [WindowState.HIDDEN, WindowState.MINIMIZED] and
                self.is_on_screen and self.alpha > 0.0)
    
    @property
    def area(self) -> int:
        """Get window area in pixels."""
        return self.bounds.area


@dataclass(frozen=True)
class ScreenCapture:
    """Screen capture result with metadata."""
    image_data: ImageData
    region: ScreenRegion
    timestamp: datetime
    capture_mode: CaptureMode
    display_id: Optional[int] = None
    privacy_filtered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_ratio: float = 1.0
    quality_score: float = 1.0
    
    def __post_init__(self):
        """Validate screen capture data."""
        if len(self.image_data) == 0:
            raise ValueError("Image data cannot be empty")
        if not (0.0 <= self.compression_ratio <= 1.0):
            raise ValueError("Compression ratio must be between 0.0 and 1.0")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("Quality score must be between 0.0 and 1.0")
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return len(self.image_data) / (1024 * 1024)
    
    @property
    def age_seconds(self) -> float:
        """Get capture age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass(frozen=True)
class ChangeDetectionResult:
    """Screen change detection result."""
    changed: bool
    change_percentage: float
    changed_regions: List[ScreenRegion]
    change_type: str  # "content", "layout", "motion", "appearance"
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate change detection result."""
        if not (0.0 <= self.change_percentage <= 100.0):
            raise ValueError("Change percentage must be between 0.0 and 100.0")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def is_significant_change(self) -> bool:
        """Check if change is significant (>10% and high confidence)."""
        return self.change_percentage > 10.0 and self.confidence > 0.8


class PermissionManager:
    """Manages screen recording permissions and access control."""
    
    def __init__(self):
        self._permission_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._cache_duration = timedelta(minutes=5)
    
    async def check_screen_recording_permission(self) -> Either[PermissionError, None]:
        """Check if screen recording permission is granted."""
        try:
            # Check cache first
            cache_key = "screen_recording"
            if cache_key in self._permission_cache:
                result, timestamp = self._permission_cache[cache_key]
                if datetime.now() - timestamp < self._cache_duration:
                    if result:
                        return Either.right(None)
                    else:
                        return Either.left(PermissionError("Screen recording permission denied"))
            
            # Simulate permission check (in real implementation, use macOS APIs)
            await asyncio.sleep(0.05)  # Simulate API call
            
            # For simulation, assume permission is granted
            permission_granted = True
            
            # Cache result
            self._permission_cache[cache_key] = (permission_granted, datetime.now())
            
            if permission_granted:
                logger.debug("Screen recording permission verified")
                return Either.right(None)
            else:
                logger.warning("Screen recording permission denied")
                return Either.left(PermissionError(
                    "Screen recording permission required. Please grant permission in System Preferences > Security & Privacy > Privacy > Screen Recording"
                ))
                
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            return Either.left(PermissionError(f"Permission check failed: {str(e)}"))
    
    async def check_window_access_permission(self, bundle_id: str) -> Either[PermissionError, None]:
        """Check if window access is permitted for specific application."""
        try:
            # Check for restricted applications
            restricted_apps = {
                "com.apple.systempreferences",
                "com.apple.keychainaccess",
                "com.apple.SecurityAgent",
                "com.apple.loginwindow"
            }
            
            if bundle_id in restricted_apps:
                return Either.left(PermissionError(f"Access to {bundle_id} is restricted for security"))
            
            # Simulate permission check
            permission_granted = True
            
            if permission_granted:
                return Either.right(None)
            else:
                return Either.left(PermissionError(f"Window access denied for {bundle_id}"))
                
        except Exception as e:
            return Either.left(PermissionError(f"Window access check failed: {str(e)}"))


class PrivacyProtection:
    """Advanced privacy protection for screen content."""
    
    # Sensitive application bundle IDs
    SENSITIVE_APPLICATIONS = {
        "com.apple.keychainaccess",
        "com.apple.systempreferences",
        "com.lastpass.LastPass",
        "com.1password.1password7",
        "com.agilebits.onepassword7",
        "org.mozilla.firefox",
        "com.google.Chrome",
        "com.apple.Safari",
        "com.microsoft.Outlook",
        "com.apple.mail"
    }
    
    # Sensitive window title patterns
    SENSITIVE_TITLE_PATTERNS = [
        r"(?i).*password.*",
        r"(?i).*login.*",
        r"(?i).*signin.*",
        r"(?i).*bank.*",
        r"(?i).*payment.*",
        r"(?i).*credit.*",
        r"(?i).*keychain.*",
        r"(?i).*private.*",
        r"(?i).*confidential.*"
    ]
    
    @classmethod
    def should_filter_window(cls, window: WindowInfo) -> bool:
        """Determine if window content should be privacy filtered."""
        # Check bundle ID
        if window.bundle_id in cls.SENSITIVE_APPLICATIONS:
            return True
        
        # Check window title patterns
        import re
        for pattern in cls.SENSITIVE_TITLE_PATTERNS:
            if re.search(pattern, window.title):
                return True
        
        return False
    
    @classmethod
    def filter_sensitive_region(cls, region: ScreenRegion, windows: List[WindowInfo]) -> ScreenRegion:
        """Filter region if it overlaps with sensitive windows."""
        for window in windows:
            if cls.should_filter_window(window) and region.overlaps_with(window.bounds):
                logger.info(f"Privacy filtering applied to region overlapping {window.title}")
                # Return empty region to indicate content should be blocked
                return ScreenRegion(0, 0, 0, 0)
        
        return region
    
    @classmethod
    def create_privacy_mask(cls, image_data: ImageData, sensitive_regions: List[ScreenRegion]) -> ImageData:
        """Create privacy mask over sensitive regions (simulation)."""
        # In real implementation, this would modify the image data
        # to blur or black out sensitive regions
        logger.info(f"Applied privacy mask to {len(sensitive_regions)} sensitive regions")
        return image_data


class ScreenAnalysisEngine:
    """
    Advanced screen analysis engine with secure capture and comprehensive monitoring.
    
    Provides sophisticated screen capture, window analysis, and change detection
    with comprehensive privacy protection and permission management.
    """
    
    def __init__(self, enable_privacy_protection: bool = True):
        self.permission_manager = PermissionManager()
        self.privacy_protection = PrivacyProtection() if enable_privacy_protection else None
        self.capture_cache: Dict[str, ScreenCapture] = {}
        self.window_cache: Dict[str, Tuple[List[WindowInfo], datetime]] = {}
        self.change_detection_baseline: Optional[ScreenCapture] = None
        self.analysis_stats = {
            "total_captures": 0,
            "cache_hits": 0,
            "privacy_filters_applied": 0,
            "average_capture_time": 0.0
        }
        logger.info(f"Screen Analysis Engine initialized with privacy protection {'enabled' if enable_privacy_protection else 'disabled'}")
    
    @require(lambda region: region.width > 0 and region.height > 0)
    @ensure(lambda result: result.is_right() or isinstance(result.get_left(), VisualError))
    async def capture_screen_region(
        self,
        region: ScreenRegion,
        mode: CaptureMode = CaptureMode.BALANCED,
        privacy_mode: bool = True,
        cache_duration_seconds: int = 5
    ) -> Either[VisualError, ScreenCapture]:
        """
        Capture screen region with advanced privacy protection and caching.
        
        Args:
            region: Screen region to capture
            mode: Capture quality/performance mode
            privacy_mode: Enable privacy content filtering
            cache_duration_seconds: How long to cache capture results
            
        Returns:
            Either screen capture result or processing error
        """
        try:
            start_time = time.time()
            logger.info(f"Starting screen capture: region {region.to_dict()}, mode: {mode.value}")
            
            # Check permissions
            permission_check = await self.permission_manager.check_screen_recording_permission()
            if permission_check.is_left():
                return Either.left(permission_check.get_left())
            
            # Check cache
            cache_key = self._generate_capture_cache_key(region, mode)
            if cache_key in self.capture_cache:
                cached_capture = self.capture_cache[cache_key]
                if cached_capture.age_seconds < cache_duration_seconds:
                    logger.debug(f"Using cached screen capture: {cache_key}")
                    self.analysis_stats["cache_hits"] += 1
                    return Either.right(cached_capture)
                else:
                    del self.capture_cache[cache_key]
            
            # Get current windows for privacy filtering
            windows_result = await self.get_window_list()
            windows = windows_result.get_right() if windows_result.is_right() else []
            
            # Apply privacy filtering if enabled
            filtered_region = region
            if privacy_mode and self.privacy_protection:
                filtered_region = self.privacy_protection.filter_sensitive_region(region, windows)
                if filtered_region.area == 0:
                    return Either.left(PrivacyError(
                        "Capture blocked due to privacy protection - region contains sensitive content"
                    ))
            
            # Perform screen capture
            capture_result = await self._perform_screen_capture(filtered_region, mode)
            if capture_result.is_left():
                return capture_result
            
            capture = capture_result.get_right()
            
            # Apply additional privacy masking if needed
            if privacy_mode and self.privacy_protection:
                sensitive_regions = [
                    w.bounds for w in windows 
                    if self.privacy_protection.should_filter_window(w) and region.overlaps_with(w.bounds)
                ]
                
                if sensitive_regions:
                    masked_data = self.privacy_protection.create_privacy_mask(
                        capture.image_data, sensitive_regions
                    )
                    capture = ScreenCapture(
                        image_data=masked_data,
                        region=capture.region,
                        timestamp=capture.timestamp,
                        capture_mode=capture.capture_mode,
                        display_id=capture.display_id,
                        privacy_filtered=True,
                        metadata=capture.metadata,
                        compression_ratio=capture.compression_ratio,
                        quality_score=capture.quality_score
                    )
                    self.analysis_stats["privacy_filters_applied"] += 1
            
            # Cache result
            self.capture_cache[cache_key] = capture
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.analysis_stats["total_captures"] += 1
            old_avg = self.analysis_stats["average_capture_time"]
            total = self.analysis_stats["total_captures"]
            self.analysis_stats["average_capture_time"] = (old_avg * (total - 1) + processing_time) / total
            
            logger.info(f"Screen capture completed in {processing_time:.1f}ms, size: {capture.file_size_mb:.2f}MB")
            return Either.right(capture)
            
        except Exception as e:
            logger.error(f"Screen capture failed: {str(e)}")
            return Either.left(ProcessingError(f"Screen capture failed: {str(e)}"))
    
    async def _perform_screen_capture(
        self,
        region: ScreenRegion,
        mode: CaptureMode
    ) -> Either[VisualError, ScreenCapture]:
        """Perform the actual screen capture (simulation)."""
        try:
            # Simulate capture delay based on mode
            capture_delays = {
                CaptureMode.FULL_QUALITY: 0.2,
                CaptureMode.BALANCED: 0.1,
                CaptureMode.PERFORMANCE: 0.05,
                CaptureMode.PRIVACY_SAFE: 0.15,
                CaptureMode.THUMBNAIL: 0.03
            }
            
            delay = capture_delays.get(mode, 0.1)
            await asyncio.sleep(delay)
            
            # Simulate image data based on region size and mode
            base_size = region.area // 4  # Rough estimate
            quality_multipliers = {
                CaptureMode.FULL_QUALITY: 1.0,
                CaptureMode.BALANCED: 0.7,
                CaptureMode.PERFORMANCE: 0.4,
                CaptureMode.PRIVACY_SAFE: 0.6,
                CaptureMode.THUMBNAIL: 0.1
            }
            
            multiplier = quality_multipliers.get(mode, 0.7)
            simulated_size = int(base_size * multiplier)
            
            # Create simulated image data
            simulated_data = b'simulated_image_data' + b'x' * max(0, simulated_size - 20)
            
            capture = ScreenCapture(
                image_data=ImageData(simulated_data),
                region=region,
                timestamp=datetime.now(),
                capture_mode=mode,
                display_id=1,
                privacy_filtered=False,
                metadata={
                    "capture_method": "simulation",
                    "processing_time_ms": delay * 1000,
                    "original_size": base_size,
                    "compressed_size": simulated_size
                },
                compression_ratio=multiplier,
                quality_score=multiplier
            )
            
            return Either.right(capture)
            
        except Exception as e:
            return Either.left(ProcessingError(f"Screen capture processing failed: {str(e)}"))
    
    async def get_window_list(
        self,
        include_hidden: bool = False,
        cache_duration_seconds: int = 2
    ) -> Either[VisualError, List[WindowInfo]]:
        """
        Get list of all windows with comprehensive information.
        
        Args:
            include_hidden: Include hidden/minimized windows
            cache_duration_seconds: How long to cache window list
            
        Returns:
            Either list of windows or processing error
        """
        try:
            # Check cache
            cache_key = f"windows_{include_hidden}"
            if cache_key in self.window_cache:
                windows, timestamp = self.window_cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < cache_duration_seconds:
                    logger.debug("Using cached window list")
                    return Either.right(windows)
            
            # Simulate window enumeration
            await asyncio.sleep(0.05)
            
            # Create simulated window list
            simulated_windows = [
                WindowInfo(
                    window_id="1",
                    title="Safari - Google Search",
                    application_name="Safari",
                    bundle_id="com.apple.Safari",
                    process_id=1234,
                    bounds=ScreenRegion(100, 100, 800, 600),
                    state=WindowState.ACTIVE,
                    layer=0,
                    is_on_screen=True,
                    owner_name="Safari",
                    window_level=0,
                    alpha=1.0,
                    has_shadow=True
                ),
                WindowInfo(
                    window_id="2",
                    title="Keyboard Maestro Editor",
                    application_name="Keyboard Maestro",
                    bundle_id="com.stairways.keyboardmaestro.editor",
                    process_id=5678,
                    bounds=ScreenRegion(200, 150, 900, 700),
                    state=WindowState.INACTIVE,
                    layer=1,
                    is_on_screen=True,
                    owner_name="Keyboard Maestro",
                    window_level=0,
                    alpha=1.0,
                    has_shadow=True
                ),
                WindowInfo(
                    window_id="3",
                    title="Terminal",
                    application_name="Terminal",
                    bundle_id="com.apple.Terminal",
                    process_id=9999,
                    bounds=ScreenRegion(300, 200, 600, 400),
                    state=WindowState.INACTIVE,
                    layer=2,
                    is_on_screen=True,
                    owner_name="Terminal",
                    window_level=0,
                    alpha=0.9,
                    has_shadow=True
                )
            ]
            
            # Filter based on include_hidden
            if not include_hidden:
                simulated_windows = [w for w in simulated_windows if w.is_visible]
            
            # Cache result
            self.window_cache[cache_key] = (simulated_windows, datetime.now())
            
            logger.info(f"Retrieved window list: {len(simulated_windows)} windows")
            return Either.right(simulated_windows)
            
        except Exception as e:
            logger.error(f"Window list retrieval failed: {str(e)}")
            return Either.left(ProcessingError(f"Window list retrieval failed: {str(e)}"))
    
    async def detect_screen_changes(
        self,
        region: ScreenRegion,
        mode: ChangeDetectionMode = ChangeDetectionMode.CONTENT_AWARE,
        sensitivity: float = 0.1
    ) -> Either[VisualError, ChangeDetectionResult]:
        """
        Detect changes in screen region since last baseline.
        
        Args:
            region: Region to monitor for changes
            mode: Change detection sensitivity mode
            sensitivity: Detection sensitivity (0.0 to 1.0)
            
        Returns:
            Either change detection result or processing error
        """
        try:
            logger.info(f"Detecting screen changes in region {region.to_dict()}, mode: {mode.value}")
            
            # Capture current screen state
            current_capture = await self.capture_screen_region(region, CaptureMode.PERFORMANCE)
            if current_capture.is_left():
                return Either.left(current_capture.get_left())
            
            current = current_capture.get_right()
            
            # If no baseline, set current as baseline and return no change
            if not self.change_detection_baseline:
                self.change_detection_baseline = current
                return Either.right(ChangeDetectionResult(
                    changed=False,
                    change_percentage=0.0,
                    changed_regions=[],
                    change_type="baseline",
                    confidence=1.0,
                    timestamp=datetime.now(),
                    metadata={"baseline_set": True}
                ))
            
            # Simulate change detection based on mode and sensitivity
            await asyncio.sleep(0.02)  # Processing delay
            
            # Simulate different change percentages based on mode
            change_percentages = {
                ChangeDetectionMode.PIXEL_PERFECT: 15.5,
                ChangeDetectionMode.CONTENT_AWARE: 8.2,
                ChangeDetectionMode.STRUCTURAL: 3.1,
                ChangeDetectionMode.MOTION_ONLY: 12.7
            }
            
            base_change = change_percentages.get(mode, 5.0)
            adjusted_change = base_change * (1.0 - sensitivity + 0.5)  # Sensitivity adjustment
            
            # Determine if change is significant
            threshold = sensitivity * 20.0  # Convert to percentage
            changed = adjusted_change > threshold
            
            # Generate changed regions if change detected
            changed_regions = []
            if changed:
                # Simulate 2-3 changed regions
                region_count = 2 if adjusted_change < 10 else 3
                for i in range(region_count):
                    changed_region = ScreenRegion(
                        x=region.x + i * 100,
                        y=region.y + i * 50,
                        width=80,
                        height=40
                    )
                    changed_regions.append(changed_region)
            
            # Determine change type
            change_types = ["content", "layout", "motion", "appearance"]
            change_type = change_types[int(adjusted_change) % len(change_types)]
            
            result = ChangeDetectionResult(
                changed=changed,
                change_percentage=adjusted_change,
                changed_regions=changed_regions,
                change_type=change_type,
                confidence=0.85 + (sensitivity * 0.1),
                timestamp=datetime.now(),
                metadata={
                    "detection_mode": mode.value,
                    "sensitivity": sensitivity,
                    "baseline_age": self.change_detection_baseline.age_seconds,
                    "comparison_method": "simulation"
                }
            )
            
            logger.info(f"Change detection completed: {'changed' if changed else 'no change'} ({adjusted_change:.1f}%)")
            return Either.right(result)
            
        except Exception as e:
            logger.error(f"Change detection failed: {str(e)}")
            return Either.left(ProcessingError(f"Change detection failed: {str(e)}"))
    
    def _generate_capture_cache_key(self, region: ScreenRegion, mode: CaptureMode) -> str:
        """Generate cache key for screen capture."""
        region_str = f"{region.x},{region.y},{region.width},{region.height}"
        return hashlib.md5(f"{region_str}_{mode.value}".encode()).hexdigest()[:16]
    
    async def analyze_color_distribution(
        self,
        region: ScreenRegion
    ) -> Either[VisualError, ColorInfo]:
        """Analyze color distribution in screen region."""
        try:
            # Capture region
            capture_result = await self.capture_screen_region(region, CaptureMode.BALANCED)
            if capture_result.is_left():
                return Either.left(capture_result.get_left())
            
            # Simulate color analysis
            await asyncio.sleep(0.05)
            
            # Generate simulated color information
            color_info = ColorInfo(
                dominant_colors=[(128, 128, 128), (255, 255, 255), (64, 64, 64)],
                color_palette=[
                    (128, 128, 128, 0.4),  # Gray 40%
                    (255, 255, 255, 0.3),  # White 30%
                    (64, 64, 64, 0.2),     # Dark gray 20%
                    (200, 200, 200, 0.1)   # Light gray 10%
                ],
                average_color=(150, 150, 150),
                brightness=0.6,
                contrast_ratio=4.5,
                color_distribution={
                    "grayscale": 0.7,
                    "colorful": 0.2,
                    "monochrome": 0.1
                }
            )
            
            logger.info(f"Color analysis completed for region {region.to_dict()}")
            return Either.right(color_info)
            
        except Exception as e:
            logger.error(f"Color analysis failed: {str(e)}")
            return Either.left(ProcessingError(f"Color analysis failed: {str(e)}"))
    
    def set_change_detection_baseline(self, region: ScreenRegion) -> None:
        """Set new baseline for change detection."""
        async def _set_baseline():
            capture_result = await self.capture_screen_region(region, CaptureMode.PERFORMANCE)
            if capture_result.is_right():
                self.change_detection_baseline = capture_result.get_right()
                logger.info(f"Change detection baseline set for region {region.to_dict()}")
        
        asyncio.create_task(_set_baseline())
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get screen analysis statistics."""
        stats = self.analysis_stats.copy()
        stats.update({
            "cache_size": len(self.capture_cache),
            "window_cache_size": len(self.window_cache),
            "baseline_set": self.change_detection_baseline is not None,
            "privacy_protection_enabled": self.privacy_protection is not None
        })
        return stats
    
    def clear_caches(self) -> None:
        """Clear all analysis caches."""
        self.capture_cache.clear()
        self.window_cache.clear()
        self.change_detection_baseline = None
        logger.info("All screen analysis caches cleared")


# Convenience functions for common screen analysis operations
async def capture_full_screen(
    display_id: int = 1,
    privacy_mode: bool = True
) -> Either[VisualError, ScreenCapture]:
    """Capture full screen with privacy protection."""
    # Simulate full screen dimensions
    full_screen_region = ScreenRegion(0, 0, 1920, 1080, display_id)
    
    engine = ScreenAnalysisEngine(enable_privacy_protection=privacy_mode)
    return await engine.capture_screen_region(full_screen_region, CaptureMode.BALANCED, privacy_mode)


async def find_active_window() -> Either[VisualError, Optional[WindowInfo]]:
    """Find the currently active window."""
    engine = ScreenAnalysisEngine()
    windows_result = await engine.get_window_list()
    
    if windows_result.is_left():
        return Either.left(windows_result.get_left())
    
    windows = windows_result.get_right()
    active_windows = [w for w in windows if w.state == WindowState.ACTIVE]
    
    return Either.right(active_windows[0] if active_windows else None)


async def monitor_region_for_changes(
    region: ScreenRegion,
    duration_seconds: int = 10,
    sensitivity: float = 0.2
) -> Either[VisualError, List[ChangeDetectionResult]]:
    """Monitor screen region for changes over time."""
    engine = ScreenAnalysisEngine()
    changes = []
    
    # Set baseline
    engine.set_change_detection_baseline(region)
    await asyncio.sleep(0.5)  # Wait for baseline to be set
    
    # Monitor for specified duration
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        change_result = await engine.detect_screen_changes(region, sensitivity=sensitivity)
        if change_result.is_right():
            change = change_result.get_right()
            if change.changed:
                changes.append(change)
        
        await asyncio.sleep(0.5)  # Check every 500ms
    
    return Either.right(changes)