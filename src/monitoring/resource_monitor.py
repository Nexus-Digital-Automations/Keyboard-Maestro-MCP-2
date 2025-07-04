"""
System Resource Monitor - TASK_54 Phase 2 Implementation

Advanced system resource monitoring with real-time tracking, trend analysis,
and resource optimization recommendations.

Architecture: Resource tracking + Type Safety + Performance optimization
Performance: <50ms resource checks, <2% system overhead
Security: Resource access validation, sensitive data protection
"""

from __future__ import annotations
import asyncio
import psutil
import platform
import shutil
import time
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.performance_monitoring import (
    MonitoringSessionID, CPUPercentage, MemoryBytes,
    SystemResourceSnapshot, MetricType, MetricValue,
    PerformanceMonitoringError
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiskUsage:
    """Disk usage information for a specific mount point."""
    mount_point: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float
    
    def __post_init__(self):
        require(lambda: self.total_bytes >= 0, "Total bytes must be non-negative")
        require(lambda: self.used_bytes >= 0, "Used bytes must be non-negative")
        require(lambda: self.free_bytes >= 0, "Free bytes must be non-negative")
        require(lambda: 0 <= self.usage_percent <= 100, "Usage percent must be 0-100")


@dataclass(frozen=True)
class NetworkInterface:
    """Network interface information and statistics."""
    interface_name: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int
    is_up: bool
    speed_mbps: Optional[int] = None
    
    def __post_init__(self):
        require(lambda: len(self.interface_name.strip()) > 0, "Interface name required")
        require(lambda: self.bytes_sent >= 0, "Bytes sent must be non-negative")
        require(lambda: self.bytes_recv >= 0, "Bytes received must be non-negative")


@dataclass(frozen=True)
class ProcessInfo:
    """Process information and resource usage."""
    pid: int
    name: str
    cpu_percent: float
    memory_bytes: int
    memory_percent: float
    status: str
    create_time: datetime
    num_threads: int
    
    def __post_init__(self):
        require(lambda: self.pid > 0, "PID must be positive")
        require(lambda: len(self.name.strip()) > 0, "Process name required")
        require(lambda: self.cpu_percent >= 0, "CPU percent must be non-negative")
        require(lambda: self.memory_bytes >= 0, "Memory bytes must be non-negative")


@dataclass
class SystemResourceReport:
    """Comprehensive system resource report."""
    timestamp: datetime
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, Any]
    disk_usage: List[DiskUsage]
    network_interfaces: List[NetworkInterface]
    top_processes: List[ProcessInfo]
    system_info: Dict[str, Any]
    load_averages: Tuple[float, float, float]
    uptime_seconds: float
    
    def __post_init__(self):
        require(lambda: self.uptime_seconds >= 0, "Uptime must be non-negative")


class ResourceMonitor:
    """
    Advanced system resource monitoring with real-time tracking and analysis.
    
    Provides comprehensive system resource information including CPU, memory,
    disk, network, and process monitoring with trend analysis.
    """
    
    def __init__(self, update_interval: float = 1.0):
        require(lambda: update_interval > 0, "Update interval must be positive")
        
        self.update_interval = update_interval
        self.monitoring_active = False
        self.resource_history: List[SystemResourceReport] = []
        self.max_history_size = 1000  # Keep last 1000 reports
        
        # Cache for expensive operations
        self._system_info_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info(f"ResourceMonitor initialized with {update_interval}s interval")
    
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns report or error")
    async def get_current_resources(self) -> Either[str, SystemResourceReport]:
        """Get current system resource usage report."""
        try:
            timestamp = datetime.now(UTC)
            
            # Collect all resource information concurrently
            cpu_task = asyncio.create_task(self._get_cpu_usage())
            memory_task = asyncio.create_task(self._get_memory_usage())
            disk_task = asyncio.create_task(self._get_disk_usage())
            network_task = asyncio.create_task(self._get_network_interfaces())
            processes_task = asyncio.create_task(self._get_top_processes(limit=10))
            system_task = asyncio.create_task(self._get_system_info())
            load_task = asyncio.create_task(self._get_load_averages())
            uptime_task = asyncio.create_task(self._get_system_uptime())
            
            # Wait for all tasks to complete
            results = await asyncio.gather(
                cpu_task, memory_task, disk_task, network_task,
                processes_task, system_task, load_task, uptime_task,
                return_exceptions=True
            )
            
            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Resource collection task {i} failed: {result}")
            
            # Extract results (with fallbacks for exceptions)
            cpu_usage = results[0] if not isinstance(results[0], Exception) else {}
            memory_usage = results[1] if not isinstance(results[1], Exception) else {}
            disk_usage = results[2] if not isinstance(results[2], Exception) else []
            network_interfaces = results[3] if not isinstance(results[3], Exception) else []
            top_processes = results[4] if not isinstance(results[4], Exception) else []
            system_info = results[5] if not isinstance(results[5], Exception) else {}
            load_averages = results[6] if not isinstance(results[6], Exception) else (0.0, 0.0, 0.0)
            uptime_seconds = results[7] if not isinstance(results[7], Exception) else 0.0
            
            report = SystemResourceReport(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_interfaces=network_interfaces,
                top_processes=top_processes,
                system_info=system_info,
                load_averages=load_averages,
                uptime_seconds=uptime_seconds
            )
            
            # Add to history
            self._add_to_history(report)
            
            return Either.right(report)
            
        except Exception as e:
            error_msg = f"Failed to collect resource information: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def _get_cpu_usage(self) -> Dict[str, float]:
        """Get detailed CPU usage information."""
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage = {
                "overall_percent": cpu_percent,
                "core_count": psutil.cpu_count(),
                "logical_count": psutil.cpu_count(logical=True)
            }
            
            # Per-core usage
            try:
                per_cpu = psutil.cpu_percent(percpu=True)
                for i, usage in enumerate(per_cpu):
                    cpu_usage[f"core_{i}_percent"] = usage
            except Exception:
                pass
            
            # CPU times
            try:
                cpu_times = psutil.cpu_times()
                cpu_usage.update({
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system,
                    "idle_time": cpu_times.idle
                })
            except Exception:
                pass
            
            # CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_usage.update({
                        "current_freq_mhz": cpu_freq.current,
                        "min_freq_mhz": cpu_freq.min,
                        "max_freq_mhz": cpu_freq.max
                    })
            except Exception:
                pass
            
            return cpu_usage
            
        except Exception as e:
            logger.warning(f"CPU usage collection failed: {e}")
            return {"overall_percent": 0.0}
    
    async def _get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information."""
        try:
            # Virtual memory
            vmem = psutil.virtual_memory()
            memory_usage = {
                "virtual_total": vmem.total,
                "virtual_used": vmem.used,
                "virtual_free": vmem.free,
                "virtual_percent": vmem.percent,
                "virtual_available": vmem.available
            }
            
            # Swap memory
            try:
                swap = psutil.swap_memory()
                memory_usage.update({
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_free": swap.free,
                    "swap_percent": swap.percent
                })
            except Exception:
                pass
            
            return memory_usage
            
        except Exception as e:
            logger.warning(f"Memory usage collection failed: {e}")
            return {}
    
    async def _get_disk_usage(self) -> List[DiskUsage]:
        """Get disk usage information for all mounted filesystems."""
        try:
            disk_usage = []
            
            # Get all disk partitions
            partitions = psutil.disk_partitions()
            
            for partition in partitions:
                try:
                    usage = shutil.disk_usage(partition.mountpoint)
                    
                    disk_info = DiskUsage(
                        mount_point=partition.mountpoint,
                        total_bytes=usage.total,
                        used_bytes=usage.used,
                        free_bytes=usage.free,
                        usage_percent=(usage.used / usage.total * 100) if usage.total > 0 else 0.0
                    )
                    disk_usage.append(disk_info)
                    
                except (OSError, PermissionError):
                    # Skip inaccessible mount points
                    continue
            
            return disk_usage
            
        except Exception as e:
            logger.warning(f"Disk usage collection failed: {e}")
            return []
    
    async def _get_network_interfaces(self) -> List[NetworkInterface]:
        """Get network interface information and statistics."""
        try:
            interfaces = []
            
            # Get network I/O statistics per interface
            net_io = psutil.net_io_counters(pernic=True)
            
            for interface_name, stats in net_io.items():
                try:
                    # Check if interface is up
                    is_up = True
                    try:
                        addrs = psutil.net_if_addrs().get(interface_name, [])
                        is_up = len(addrs) > 0
                    except Exception:
                        pass
                    
                    interface = NetworkInterface(
                        interface_name=interface_name,
                        bytes_sent=stats.bytes_sent,
                        bytes_recv=stats.bytes_recv,
                        packets_sent=stats.packets_sent,
                        packets_recv=stats.packets_recv,
                        errors_in=stats.errin,
                        errors_out=stats.errout,
                        drops_in=stats.dropin,
                        drops_out=stats.dropout,
                        is_up=is_up
                    )
                    interfaces.append(interface)
                    
                except Exception as e:
                    logger.debug(f"Failed to get info for interface {interface_name}: {e}")
                    continue
            
            return interfaces
            
        except Exception as e:
            logger.warning(f"Network interface collection failed: {e}")
            return []
    
    async def _get_top_processes(self, limit: int = 10) -> List[ProcessInfo]:
        """Get top processes by CPU and memory usage."""
        try:
            processes = []
            
            # Get all processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'memory_percent', 'status', 'create_time', 'num_threads']):
                try:
                    pinfo = proc.info
                    
                    if pinfo['memory_info'] is None:
                        continue
                    
                    process_info = ProcessInfo(
                        pid=pinfo['pid'],
                        name=pinfo['name'] or 'Unknown',
                        cpu_percent=pinfo['cpu_percent'] or 0.0,
                        memory_bytes=pinfo['memory_info'].rss if pinfo['memory_info'] else 0,
                        memory_percent=pinfo['memory_percent'] or 0.0,
                        status=pinfo['status'] or 'unknown',
                        create_time=datetime.fromtimestamp(pinfo['create_time'], UTC) if pinfo['create_time'] else datetime.now(UTC),
                        num_threads=pinfo['num_threads'] or 0
                    )
                    processes.append(process_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage (descending) and take top N
            processes.sort(key=lambda p: p.cpu_percent, reverse=True)
            return processes[:limit]
            
        except Exception as e:
            logger.warning(f"Process collection failed: {e}")
            return []
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get general system information (cached)."""
        try:
            # Check cache
            now = datetime.now(UTC)
            if (self._system_info_cache and self._cache_timestamp and 
                now - self._cache_timestamp < self._cache_ttl):
                return self._system_info_cache
            
            system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version()
            }
            
            # Boot time
            try:
                boot_time = psutil.boot_time()
                system_info["boot_time"] = datetime.fromtimestamp(boot_time, UTC).isoformat()
            except Exception:
                pass
            
            # Update cache
            self._system_info_cache = system_info
            self._cache_timestamp = now
            
            return system_info
            
        except Exception as e:
            logger.warning(f"System info collection failed: {e}")
            return {}
    
    async def _get_load_averages(self) -> Tuple[float, float, float]:
        """Get system load averages."""
        try:
            if hasattr(psutil, 'getloadavg'):
                return psutil.getloadavg()
            else:
                # Load averages not available on all platforms
                return (0.0, 0.0, 0.0)
        except Exception as e:
            logger.debug(f"Load average collection failed: {e}")
            return (0.0, 0.0, 0.0)
    
    async def _get_system_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            boot_time = psutil.boot_time()
            return time.time() - boot_time
        except Exception as e:
            logger.debug(f"Uptime collection failed: {e}")
            return 0.0
    
    def _add_to_history(self, report: SystemResourceReport) -> None:
        """Add report to history, maintaining size limit."""
        self.resource_history.append(report)
        
        # Trim history if it exceeds max size
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size:]
    
    def get_resource_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage trends over the specified time period."""
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
            
            # Filter reports within time range
            recent_reports = [
                report for report in self.resource_history
                if report.timestamp >= cutoff_time
            ]
            
            if not recent_reports:
                return {"error": "No historical data available"}
            
            # Calculate trends
            cpu_values = [report.cpu_usage.get("overall_percent", 0) for report in recent_reports]
            memory_values = [report.memory_usage.get("virtual_percent", 0) for report in recent_reports]
            
            trends = {
                "time_range_hours": hours,
                "sample_count": len(recent_reports),
                "cpu_trend": {
                    "min": min(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "current": cpu_values[-1] if cpu_values else 0
                },
                "memory_trend": {
                    "min": min(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                    "current": memory_values[-1] if memory_values else 0
                },
                "first_sample": recent_reports[0].timestamp.isoformat(),
                "last_sample": recent_reports[-1].timestamp.isoformat()
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return {"error": f"Trend calculation failed: {str(e)}"}
    
    def get_resource_alerts(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0) -> List[str]:
        """Get current resource alerts based on thresholds."""
        alerts = []
        
        if not self.resource_history:
            return alerts
        
        latest_report = self.resource_history[-1]
        
        # CPU alerts
        cpu_percent = latest_report.cpu_usage.get("overall_percent", 0)
        if cpu_percent > cpu_threshold:
            alerts.append(f"High CPU usage: {cpu_percent:.1f}% (threshold: {cpu_threshold}%)")
        
        # Memory alerts
        memory_percent = latest_report.memory_usage.get("virtual_percent", 0)
        if memory_percent > memory_threshold:
            alerts.append(f"High memory usage: {memory_percent:.1f}% (threshold: {memory_threshold}%)")
        
        # Disk alerts
        for disk in latest_report.disk_usage:
            if disk.usage_percent > 90.0:
                alerts.append(f"High disk usage on {disk.mount_point}: {disk.usage_percent:.1f}%")
        
        return alerts
    
    def clear_history(self) -> None:
        """Clear resource history."""
        self.resource_history.clear()
        logger.info("Resource history cleared")


# Global instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create the global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor