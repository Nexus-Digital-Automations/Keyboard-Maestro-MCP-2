"""
Real-time synchronization tools.

Contains the TASK_7 implementation tools for real-time macro library
synchronization, file monitoring, and change detection.
"""

import logging
from datetime import datetime, UTC
from typing import Any, Dict

from fastmcp import Context
from pydantic import Field
from typing_extensions import Annotated

from ...integration.file_monitor import WATCHDOG_AVAILABLE
from ...integration.sync_manager import SyncStatus
from ...core.types import Duration
from ..initialization import get_sync_manager, get_file_monitor

logger = logging.getLogger(__name__)


async def km_start_realtime_sync(
    enable_file_monitoring: Annotated[bool, Field(
        default=True,
        description="Enable file system monitoring for immediate change detection"
    )] = True,
    poll_interval_seconds: Annotated[int, Field(
        default=30,
        ge=5,
        le=300,
        description="Base polling interval in seconds"
    )] = 30,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Start real-time macro library synchronization and monitoring.
    
    TASK_7 IMPLEMENTATION: Real-time state synchronization with intelligent
    polling, file monitoring, and change detection for live macro updates.
    """
    if ctx:
        await ctx.info("Starting real-time macro synchronization")
    
    try:
        sync_mgr = get_sync_manager()
        
        # Check if already running
        if sync_mgr.sync_state.status == SyncStatus.ACTIVE:
            return {
                "success": True,
                "data": {
                    "message": "Real-time sync already active",
                    "status": sync_mgr.get_sync_status()
                }
            }
        
        if ctx:
            await ctx.report_progress(25, 100, "Configuring synchronization")
        
        # Update configuration if provided
        if poll_interval_seconds != 30:
            sync_mgr.config.base_poll_interval = Duration.from_seconds(poll_interval_seconds)
            sync_mgr.config.slow_poll_interval = Duration.from_seconds(poll_interval_seconds * 4)
        
        # Start synchronization
        start_result = await sync_mgr.start_sync()
        
        if start_result.is_left():
            error = start_result.get_left()
            if ctx:
                await ctx.error(f"Failed to start sync: {error}")
            return {
                "success": False,
                "error": {
                    "code": "SYNC_START_FAILED",
                    "message": "Failed to start real-time synchronization",
                    "details": str(error),
                    "recovery_suggestion": "Check Keyboard Maestro connection and try again"
                }
            }
        
        if ctx:
            await ctx.report_progress(75, 100, "Starting file monitoring")
        
        # Start file monitoring if enabled
        file_monitor_status = None
        if enable_file_monitoring:
            monitor = get_file_monitor()
            if monitor.start_monitoring():
                file_monitor_status = "active"
                if ctx:
                    await ctx.info("File system monitoring enabled")
            else:
                file_monitor_status = "unavailable"
                if ctx:
                    await ctx.warn("File monitoring unavailable - using polling only")
        else:
            file_monitor_status = "disabled"
        
        if ctx:
            await ctx.report_progress(100, 100, "Real-time sync started")
            await ctx.info("Real-time macro synchronization is now active")
        
        return {
            "success": True,
            "data": {
                "message": "Real-time synchronization started successfully",
                "sync_status": sync_mgr.get_sync_status(),
                "file_monitoring": {
                    "status": file_monitor_status,
                    "watchdog_available": WATCHDOG_AVAILABLE,
                    "details": get_file_monitor().get_status() if file_monitor_status == "active" else None
                },
                "configuration": {
                    "poll_interval_seconds": sync_mgr.config.base_poll_interval.total_seconds(),
                    "fast_poll_interval_seconds": sync_mgr.config.fast_poll_interval.total_seconds(),
                    "slow_poll_interval_seconds": sync_mgr.config.slow_poll_interval.total_seconds(),
                    "cache_ttl_minutes": sync_mgr.config.cache_ttl.total_seconds() / 60
                }
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "server_version": "1.0.0",
                "feature": "real_time_sync_task_7"
            }
        }
        
    except Exception as e:
        logger.exception("Error starting real-time sync")
        if ctx:
            await ctx.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Failed to start real-time synchronization",
                "details": str(e),
                "recovery_suggestion": "Check logs and system configuration"
            }
        }


async def km_stop_realtime_sync(
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Stop real-time macro library synchronization and monitoring.
    
    TASK_7 IMPLEMENTATION: Gracefully stop all real-time sync processes.
    """
    if ctx:
        await ctx.info("Stopping real-time macro synchronization")
    
    try:
        sync_mgr = get_sync_manager()
        
        if sync_mgr.sync_state.status == SyncStatus.STOPPED:
            return {
                "success": True,
                "data": {
                    "message": "Real-time sync already stopped",
                    "status": sync_mgr.get_sync_status()
                }
            }
        
        if ctx:
            await ctx.report_progress(33, 100, "Stopping synchronization")
        
        # Stop sync manager
        await sync_mgr.stop_sync()
        
        if ctx:
            await ctx.report_progress(66, 100, "Stopping file monitoring")
        
        # Stop file monitoring
        monitor = get_file_monitor()
        monitor.stop_monitoring()
        
        if ctx:
            await ctx.report_progress(100, 100, "Real-time sync stopped")
            await ctx.info("Real-time synchronization stopped")
        
        return {
            "success": True,
            "data": {
                "message": "Real-time synchronization stopped successfully",
                "final_status": sync_mgr.get_sync_status()
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "server_version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.exception("Error stopping real-time sync")
        if ctx:
            await ctx.error(f"Error stopping sync: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Failed to stop real-time synchronization",
                "details": str(e)
            }
        }


async def km_sync_status(
    include_performance_metrics: Annotated[bool, Field(
        default=True,
        description="Include detailed performance metrics"
    )] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get current status of real-time macro synchronization including
    performance metrics, change detection, and monitoring health.
    
    TASK_7 IMPLEMENTATION: Comprehensive sync status with diagnostics.
    """
    if ctx:
        await ctx.info("Retrieving real-time sync status")
    
    try:
        sync_mgr = get_sync_manager()
        monitor = get_file_monitor()
        
        # Get basic sync status
        sync_status = sync_mgr.get_sync_status()
        
        # Get file monitoring status
        file_status = monitor.get_status()
        
        # Build comprehensive status
        status_data = {
            "synchronization": sync_status,
            "file_monitoring": file_status,
            "system_info": {
                "watchdog_available": WATCHDOG_AVAILABLE,
                "server_uptime": "unknown",  # Could track this
                "sync_feature_active": sync_status["status"] == "active"
            }
        }
        
        if include_performance_metrics:
            status_data["performance"] = {
                "average_sync_time": sync_status.get("average_sync_time_seconds", 0.0),
                "poll_interval_seconds": sync_status.get("current_poll_interval_seconds", 30),
                "changes_per_hour": "unknown",  # Could calculate this
                "cache_hit_rate": "unknown"  # Could track this
            }
        
        # Determine overall health
        is_healthy = (
            sync_status["status"] in ["active", "stopped"] and
            sync_status["consecutive_errors"] < 3
        )
        
        status_data["health"] = {
            "overall_status": "healthy" if is_healthy else "degraded",
            "consecutive_errors": sync_status["consecutive_errors"],
            "last_sync": sync_status["last_full_sync"],
            "recommendations": []
        }
        
        # Add recommendations
        if sync_status["status"] == "error":
            status_data["health"]["recommendations"].append("Check Keyboard Maestro connection")
        
        if sync_status["consecutive_errors"] > 0:
            status_data["health"]["recommendations"].append("Monitor error logs for connection issues")
        
        if not file_status["is_monitoring"] and WATCHDOG_AVAILABLE:
            status_data["health"]["recommendations"].append("Enable file monitoring for faster change detection")
        
        return {
            "success": True,
            "data": status_data,
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "server_version": "1.0.0",
                "feature": "sync_status_monitoring"
            }
        }
        
    except Exception as e:
        logger.exception("Error getting sync status")
        if ctx:
            await ctx.error(f"Error retrieving status: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Failed to retrieve synchronization status",
                "details": str(e)
            }
        }


async def km_force_sync(
    full_resync: Annotated[bool, Field(
        default=False,
        description="Force complete resynchronization instead of incremental"
    )] = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Force immediate synchronization of macro library state.
    
    TASK_7 IMPLEMENTATION: Manual sync trigger for immediate updates.
    """
    if ctx:
        await ctx.info("Forcing macro library synchronization")
    
    try:
        sync_mgr = get_sync_manager()
        
        if ctx:
            await ctx.report_progress(25, 100, "Starting forced synchronization")
        
        # Perform synchronization
        sync_result = await sync_mgr.force_sync()
        
        if sync_result.is_left():
            error = sync_result.get_left()
            if ctx:
                await ctx.error(f"Sync failed: {error}")
            return {
                "success": False,
                "error": {
                    "code": "SYNC_FAILED",
                    "message": "Forced synchronization failed",
                    "details": str(error),
                    "recovery_suggestion": "Check Keyboard Maestro connection"
                }
            }
        
        macro_count = sync_result.get_right()
        
        if ctx:
            await ctx.report_progress(100, 100, "Synchronization complete")
            await ctx.info(f"Synchronized {macro_count} macros")
        
        return {
            "success": True,
            "data": {
                "message": "Forced synchronization completed successfully",
                "macros_synchronized": macro_count,
                "sync_type": "full" if full_resync else "incremental",
                "sync_status": sync_mgr.get_sync_status()
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "server_version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.exception("Error in forced sync")
        if ctx:
            await ctx.error(f"Force sync error: {e}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": "Failed to force synchronization",
                "details": str(e)
            }
        }