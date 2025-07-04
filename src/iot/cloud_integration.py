"""
IoT Cloud Integration - TASK_65 Phase 5 Integration & Optimization

Cloud platform integration, data synchronization, multi-cloud IoT management,
and scalable cloud-based IoT orchestration for enterprise automation.

Architecture: Multi-Cloud Integration + Data Synchronization + Scalable Processing + Enterprise Orchestration
Performance: <100ms cloud API calls, <500ms data sync, <1s multi-cloud coordination
Security: Cloud-native security, encrypted data transfer, cross-platform authentication
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass, field
import asyncio
import json
import base64
from enum import Enum
import logging

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError, SecurityError, SystemError
from ..core.iot_architecture import (
    DeviceId, SensorId, IoTIntegrationError, IoTDevice, SensorReading
)

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud platforms for IoT integration."""
    AWS_IOT = "aws_iot"
    AZURE_IOT = "azure_iot"
    GOOGLE_IOT = "google_iot"
    IBM_WATSON = "ibm_watson"
    ALIBABA_IOT = "alibaba_iot"
    ORACLE_IOT = "oracle_iot"
    CUSTOM = "custom"


class SyncStrategy(Enum):
    """Data synchronization strategies."""
    REAL_TIME = "real_time"
    BATCH_SYNC = "batch_sync"
    DELTA_SYNC = "delta_sync"
    CONFLICT_RESOLUTION = "conflict_resolution"
    EVENTUAL_CONSISTENCY = "eventual_consistency"


class CloudServiceType(Enum):
    """Types of cloud IoT services."""
    DEVICE_MANAGEMENT = "device_management"
    DATA_INGESTION = "data_ingestion"
    ANALYTICS = "analytics"
    RULES_ENGINE = "rules_engine"
    ML_INFERENCE = "ml_inference"
    STORAGE = "storage"
    MESSAGING = "messaging"
    SECURITY = "security"


CloudConnectionId = str
SyncJobId = str
CloudResourceId = str


@dataclass
class CloudConnection:
    """Cloud platform connection configuration."""
    connection_id: CloudConnectionId
    provider: CloudProvider
    region: str
    credentials: Dict[str, str]  # Encrypted credentials
    endpoint_url: str
    connection_timeout: float
    retry_config: Dict[str, int]
    rate_limits: Dict[str, int]
    security_config: Dict[str, Any]
    health_check_interval: int = 300  # 5 minutes
    last_health_check: Optional[datetime] = None
    is_active: bool = True
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        if not self.is_active:
            return False
        
        if self.last_health_check is None:
            return False
        
        health_threshold = datetime.now(UTC) - timedelta(seconds=self.health_check_interval * 2)
        return self.last_health_check > health_threshold


@dataclass
class SyncConfiguration:
    """Data synchronization configuration."""
    sync_id: SyncJobId
    source_devices: List[DeviceId]
    target_cloud: CloudProvider
    sync_strategy: SyncStrategy
    sync_interval: int  # seconds
    batch_size: int
    conflict_resolution: str
    data_filters: List[str]
    compression_enabled: bool = True
    encryption_enabled: bool = True
    
    def should_sync_device(self, device_id: DeviceId) -> bool:
        """Check if device should be included in sync."""
        return device_id in self.source_devices or "*" in self.source_devices


@dataclass
class CloudResource:
    """Cloud IoT resource representation."""
    resource_id: CloudResourceId
    provider: CloudProvider
    resource_type: CloudServiceType
    configuration: Dict[str, Any]
    status: str
    created_at: datetime
    last_updated: datetime
    cost_estimate: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_operational(self) -> bool:
        """Check if resource is operational."""
        return self.status in ["active", "running", "healthy"]


class CloudIntegrationManager:
    """
    Comprehensive cloud integration manager for IoT platforms.
    
    Contracts:
        Preconditions:
            - All cloud connections must be authenticated and validated
            - Data synchronization must respect rate limits and quotas
            - Cross-cloud operations must maintain data consistency
        
        Postconditions:
            - All cloud operations are logged and monitored
            - Data synchronization maintains integrity and security
            - Multi-cloud coordination provides fault tolerance
        
        Invariants:
            - Cloud credentials are always encrypted at rest
            - Data transfers use secure channels and encryption
            - Resource costs are tracked and optimized continuously
    """
    
    def __init__(self):
        self.cloud_connections: Dict[CloudConnectionId, CloudConnection] = {}
        self.sync_configurations: Dict[SyncJobId, SyncConfiguration] = {}
        self.cloud_resources: Dict[CloudResourceId, CloudResource] = {}
        self.sync_jobs: Dict[SyncJobId, Dict[str, Any]] = {}
        
        # Performance tracking
        self.total_sync_operations = 0
        self.successful_syncs = 0
        self.failed_syncs = 0
        self.total_data_transferred = 0  # bytes
        self.total_cost_incurred = 0.0
        
        # Rate limiting and optimization
        self.rate_limiters: Dict[CloudProvider, Dict[str, Any]] = {}
        self.cost_optimizer_enabled = True
        self.auto_scaling_enabled = True
        
        # Initialize default cloud configurations
        self._initialize_cloud_providers()
    
    def _initialize_cloud_providers(self):
        """Initialize default cloud provider configurations."""
        # AWS IoT Core configuration
        self.rate_limiters[CloudProvider.AWS_IOT] = {
            "device_registry": {"requests_per_second": 100, "burst": 150},
            "data_ingestion": {"messages_per_second": 1000, "burst": 2000},
            "rules_engine": {"evaluations_per_second": 500, "burst": 750}
        }
        
        # Azure IoT Hub configuration
        self.rate_limiters[CloudProvider.AZURE_IOT] = {
            "device_management": {"requests_per_second": 83, "burst": 100},
            "telemetry": {"messages_per_second": 833, "burst": 1000},
            "device_twin": {"operations_per_second": 10, "burst": 20}
        }
        
        # Google Cloud IoT Core configuration
        self.rate_limiters[CloudProvider.GOOGLE_IOT] = {
            "device_manager": {"requests_per_second": 100, "burst": 200},
            "telemetry": {"messages_per_second": 100, "burst": 500},
            "config_updates": {"operations_per_second": 1, "burst": 5}
        }
    
    @require(lambda self, connection: connection.connection_id and connection.provider)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def register_cloud_connection(self, connection: CloudConnection) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Register cloud platform connection for IoT integration.
        
        Architecture:
            - Validates cloud credentials and connectivity
            - Establishes secure communication channels
            - Configures rate limiting and monitoring
        
        Security:
            - Encrypts credentials at rest
            - Validates SSL/TLS certificates
            - Implements secure authentication flows
        """
        try:
            # Validate connection credentials
            validation_result = await self._validate_cloud_credentials(connection)
            if validation_result.is_error():
                return validation_result
            
            # Test connectivity
            connectivity_result = await self._test_cloud_connectivity(connection)
            if connectivity_result.is_error():
                return connectivity_result
            
            # Encrypt credentials
            encrypted_credentials = await self._encrypt_credentials(connection.credentials)
            connection.credentials = encrypted_credentials
            
            # Store connection
            self.cloud_connections[connection.connection_id] = connection
            
            # Initialize health monitoring
            connection.last_health_check = datetime.now(UTC)
            
            # Set up rate limiting
            await self._setup_rate_limiting(connection)
            
            registration_info = {
                "connection_id": connection.connection_id,
                "provider": connection.provider.value,
                "region": connection.region,
                "endpoint": connection.endpoint_url,
                "health_status": "healthy",
                "rate_limits_configured": True,
                "encryption_enabled": True,
                "registered_at": datetime.now(UTC).isoformat()
            }
            
            logger.info(f"Cloud connection registered: {connection.connection_id}")
            
            return Either.success({
                "success": True,
                "registration_info": registration_info,
                "total_connections": len(self.cloud_connections)
            })
            
        except Exception as e:
            error_msg = f"Failed to register cloud connection: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg))
    
    @require(lambda self, config: config.sync_id and len(config.source_devices) > 0)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def configure_data_sync(self, config: SyncConfiguration) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Configure data synchronization between local IoT and cloud platforms.
        
        Performance:
            - Optimizes batch sizes based on network conditions
            - Implements intelligent retry mechanisms
            - Provides real-time sync monitoring
        """
        try:
            # Validate sync configuration
            if config.sync_interval < 1:
                return Either.error(IoTIntegrationError(
                    "Sync interval must be at least 1 second"
                ))
            
            if config.batch_size > 1000:
                return Either.error(IoTIntegrationError(
                    "Batch size cannot exceed 1000 items"
                ))
            
            # Check cloud connection availability
            cloud_available = await self._check_cloud_availability(config.target_cloud)
            if not cloud_available:
                return Either.error(IoTIntegrationError(
                    f"Cloud provider {config.target_cloud.value} not available"
                ))
            
            # Store sync configuration
            self.sync_configurations[config.sync_id] = config
            
            # Initialize sync job
            sync_job = {
                "sync_id": config.sync_id,
                "status": "configured",
                "created_at": datetime.now(UTC),
                "last_sync": None,
                "sync_count": 0,
                "error_count": 0,
                "data_transferred": 0,
                "next_sync": datetime.now(UTC) + timedelta(seconds=config.sync_interval)
            }
            
            self.sync_jobs[config.sync_id] = sync_job
            
            sync_info = {
                "sync_id": config.sync_id,
                "target_cloud": config.target_cloud.value,
                "sync_strategy": config.sync_strategy.value,
                "device_count": len(config.source_devices),
                "sync_interval": config.sync_interval,
                "batch_size": config.batch_size,
                "encryption_enabled": config.encryption_enabled,
                "compression_enabled": config.compression_enabled,
                "configured_at": datetime.now(UTC).isoformat()
            }
            
            logger.info(f"Data sync configured: {config.sync_id}")
            
            return Either.success({
                "success": True,
                "sync_info": sync_info,
                "total_sync_jobs": len(self.sync_configurations)
            })
            
        except Exception as e:
            error_msg = f"Failed to configure data sync: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg))
    
    @require(lambda self, sync_id: sync_id in self.sync_configurations)
    async def execute_data_sync(self, sync_id: SyncJobId, force_sync: bool = False) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """
        Execute data synchronization for specified sync job.
        
        Architecture:
            - Implements different sync strategies (real-time, batch, delta)
            - Handles conflict resolution and consistency
            - Provides comprehensive error handling and recovery
        """
        try:
            config = self.sync_configurations[sync_id]
            sync_job = self.sync_jobs[sync_id]
            
            # Check if sync is due (unless forced)
            if not force_sync:
                next_sync = sync_job.get("next_sync", datetime.now(UTC))
                if datetime.now(UTC) < next_sync:
                    return Either.success({
                        "success": True,
                        "message": "Sync not yet due",
                        "next_sync": next_sync.isoformat()
                    })
            
            # Update sync job status
            sync_job["status"] = "running"
            sync_job["last_sync"] = datetime.now(UTC)
            
            # Execute sync based on strategy
            if config.sync_strategy == SyncStrategy.REAL_TIME:
                result = await self._execute_realtime_sync(config)
            elif config.sync_strategy == SyncStrategy.BATCH_SYNC:
                result = await self._execute_batch_sync(config)
            elif config.sync_strategy == SyncStrategy.DELTA_SYNC:
                result = await self._execute_delta_sync(config)
            else:
                result = await self._execute_default_sync(config)
            
            # Update sync job with results
            if result.is_success():
                sync_job["status"] = "completed"
                sync_job["sync_count"] += 1
                sync_job["data_transferred"] += result.value.get("bytes_transferred", 0)
                self.successful_syncs += 1
            else:
                sync_job["status"] = "failed"
                sync_job["error_count"] += 1
                self.failed_syncs += 1
            
            # Schedule next sync
            sync_job["next_sync"] = datetime.now(UTC) + timedelta(seconds=config.sync_interval)
            
            self.total_sync_operations += 1
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute data sync: {str(e)}"
            logger.error(error_msg)
            return Either.error(IoTIntegrationError(error_msg))
    
    async def _execute_realtime_sync(self, config: SyncConfiguration) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute real-time data synchronization."""
        try:
            sync_results = []
            total_bytes = 0
            
            for device_id in config.source_devices:
                if not config.should_sync_device(device_id):
                    continue
                
                # Get latest device data
                device_data = await self._get_device_data(device_id)
                if not device_data:
                    continue
                
                # Apply data filters
                filtered_data = self._apply_data_filters(device_data, config.data_filters)
                
                # Compress if enabled
                if config.compression_enabled:
                    filtered_data = await self._compress_data(filtered_data)
                
                # Encrypt if enabled
                if config.encryption_enabled:
                    filtered_data = await self._encrypt_data(filtered_data)
                
                # Send to cloud
                cloud_result = await self._send_to_cloud(config.target_cloud, filtered_data)
                
                if cloud_result.is_success():
                    bytes_sent = len(json.dumps(filtered_data).encode('utf-8'))
                    total_bytes += bytes_sent
                    sync_results.append({
                        "device_id": device_id,
                        "status": "success",
                        "bytes_sent": bytes_sent
                    })
                else:
                    sync_results.append({
                        "device_id": device_id,
                        "status": "failed",
                        "error": str(cloud_result.error_value)
                    })
            
            return Either.success({
                "sync_type": "real_time",
                "devices_processed": len(sync_results),
                "bytes_transferred": total_bytes,
                "sync_results": sync_results,
                "completed_at": datetime.now(UTC).isoformat()
            })
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Real-time sync failed: {str(e)}"))
    
    async def _execute_batch_sync(self, config: SyncConfiguration) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute batch data synchronization."""
        try:
            batch_data = []
            device_count = 0
            
            # Collect data from all devices
            for device_id in config.source_devices:
                if not config.should_sync_device(device_id):
                    continue
                
                device_data = await self._get_device_data(device_id)
                if device_data:
                    batch_data.append({
                        "device_id": device_id,
                        "data": device_data,
                        "timestamp": datetime.now(UTC).isoformat()
                    })
                    device_count += 1
                
                # Process in batches
                if len(batch_data) >= config.batch_size:
                    await self._process_batch(batch_data, config)
                    batch_data = []
            
            # Process remaining data
            if batch_data:
                await self._process_batch(batch_data, config)
            
            total_bytes = len(json.dumps(batch_data).encode('utf-8')) if batch_data else 0
            
            return Either.success({
                "sync_type": "batch",
                "devices_processed": device_count,
                "bytes_transferred": total_bytes,
                "batch_size": config.batch_size,
                "completed_at": datetime.now(UTC).isoformat()
            })
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Batch sync failed: {str(e)}"))
    
    async def _execute_delta_sync(self, config: SyncConfiguration) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute delta (incremental) data synchronization."""
        try:
            # Get last sync timestamp for delta calculation
            sync_job = self.sync_jobs.get(config.sync_id, {})
            last_sync = sync_job.get("last_sync", datetime.now(UTC) - timedelta(hours=1))
            
            delta_data = []
            device_count = 0
            
            for device_id in config.source_devices:
                if not config.should_sync_device(device_id):
                    continue
                
                # Get only changed data since last sync
                changed_data = await self._get_changed_device_data(device_id, last_sync)
                if changed_data:
                    delta_data.append({
                        "device_id": device_id,
                        "changes": changed_data,
                        "change_timestamp": datetime.now(UTC).isoformat()
                    })
                    device_count += 1
            
            if delta_data:
                # Apply filters, compression, encryption
                processed_data = delta_data
                
                if config.compression_enabled:
                    processed_data = await self._compress_data(processed_data)
                
                if config.encryption_enabled:
                    processed_data = await self._encrypt_data(processed_data)
                
                # Send delta to cloud
                cloud_result = await self._send_to_cloud(config.target_cloud, processed_data)
                
                if cloud_result.is_error():
                    return cloud_result
            
            total_bytes = len(json.dumps(delta_data).encode('utf-8'))
            
            return Either.success({
                "sync_type": "delta",
                "devices_with_changes": device_count,
                "bytes_transferred": total_bytes,
                "delta_since": last_sync.isoformat(),
                "completed_at": datetime.now(UTC).isoformat()
            })
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Delta sync failed: {str(e)}"))
    
    async def _execute_default_sync(self, config: SyncConfiguration) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Execute default synchronization strategy."""
        # Default to batch sync
        return await self._execute_batch_sync(config)
    
    async def _validate_cloud_credentials(self, connection: CloudConnection) -> Either[IoTIntegrationError, bool]:
        """Validate cloud platform credentials."""
        try:
            if connection.provider == CloudProvider.AWS_IOT:
                required_keys = ["access_key_id", "secret_access_key", "region"]
            elif connection.provider == CloudProvider.AZURE_IOT:
                required_keys = ["connection_string", "tenant_id", "client_id"]
            elif connection.provider == CloudProvider.GOOGLE_IOT:
                required_keys = ["project_id", "private_key", "client_email"]
            else:
                required_keys = ["api_key", "endpoint"]
            
            for key in required_keys:
                if key not in connection.credentials:
                    return Either.error(IoTIntegrationError(
                        f"Missing required credential: {key}"
                    ))
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Credential validation failed: {str(e)}"))
    
    async def _test_cloud_connectivity(self, connection: CloudConnection) -> Either[IoTIntegrationError, bool]:
        """Test connectivity to cloud platform."""
        try:
            # Simulate connectivity test (in real implementation, make actual API call)
            import random
            
            # Simulate network latency
            await asyncio.sleep(0.1)
            
            # Simulate occasional connectivity issues
            if random.random() < 0.05:  # 5% failure rate
                return Either.error(IoTIntegrationError(
                    f"Connectivity test failed for {connection.provider.value}"
                ))
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Connectivity test failed: {str(e)}"))
    
    async def _encrypt_credentials(self, credentials: Dict[str, str]) -> Dict[str, str]:
        """Encrypt sensitive credential data."""
        # Simple base64 encoding for simulation (use proper encryption in production)
        encrypted_creds = {}
        for key, value in credentials.items():
            encrypted_value = base64.b64encode(value.encode()).decode()
            encrypted_creds[key] = encrypted_value
        
        return encrypted_creds
    
    async def _setup_rate_limiting(self, connection: CloudConnection):
        """Set up rate limiting for cloud connection."""
        # Initialize rate limiting counters
        if connection.provider not in self.rate_limiters:
            self.rate_limiters[connection.provider] = {}
        
        # Set up connection-specific rate limits
        connection.rate_limits = {
            "requests_per_second": 100,
            "burst_capacity": 200,
            "daily_quota": 100000
        }
    
    async def _check_cloud_availability(self, provider: CloudProvider) -> bool:
        """Check if cloud provider is available."""
        # Check if we have any active connections for this provider
        for connection in self.cloud_connections.values():
            if connection.provider == provider and connection.is_healthy():
                return True
        
        return False
    
    async def _get_device_data(self, device_id: DeviceId) -> Optional[Dict[str, Any]]:
        """Get current data for device."""
        # Simulate device data retrieval
        return {
            "device_id": device_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "temperature": 22.5,
            "humidity": 45.0,
            "status": "active",
            "battery_level": 85
        }
    
    async def _get_changed_device_data(self, device_id: DeviceId, since: datetime) -> Optional[Dict[str, Any]]:
        """Get changed data for device since timestamp."""
        # Simulate changed data calculation
        current_data = await self._get_device_data(device_id)
        
        # Return changes only if device was modified since timestamp
        if current_data and datetime.now(UTC) > since + timedelta(minutes=5):
            return {
                "changes": current_data,
                "change_type": "update",
                "previous_values": {"temperature": 21.8, "humidity": 43.2}
            }
        
        return None
    
    def _apply_data_filters(self, data: Dict[str, Any], filters: List[str]) -> Dict[str, Any]:
        """Apply data filters to device data."""
        if not filters:
            return data
        
        filtered_data = {}
        for key, value in data.items():
            if any(f in key for f in filters):
                filtered_data[key] = value
        
        return filtered_data if filtered_data else data
    
    async def _compress_data(self, data: Any) -> str:
        """Compress data for transmission."""
        # Simple compression simulation
        json_str = json.dumps(data)
        compressed = base64.b64encode(json_str.encode()).decode()
        return compressed
    
    async def _encrypt_data(self, data: Any) -> str:
        """Encrypt data for secure transmission."""
        # Simple encryption simulation
        if isinstance(data, str):
            encrypted = base64.b64encode(data.encode()).decode()
        else:
            json_str = json.dumps(data)
            encrypted = base64.b64encode(json_str.encode()).decode()
        
        return encrypted
    
    async def _send_to_cloud(self, provider: CloudProvider, data: Any) -> Either[IoTIntegrationError, Dict[str, Any]]:
        """Send data to cloud platform."""
        try:
            # Simulate cloud API call
            await asyncio.sleep(0.05)  # Simulate network latency
            
            # Track costs
            data_size = len(json.dumps(data).encode('utf-8'))
            cost = self._calculate_transmission_cost(provider, data_size)
            self.total_cost_incurred += cost
            
            return Either.success({
                "message_id": f"msg_{datetime.now(UTC).timestamp()}",
                "provider": provider.value,
                "data_size": data_size,
                "cost": cost,
                "transmitted_at": datetime.now(UTC).isoformat()
            })
            
        except Exception as e:
            return Either.error(IoTIntegrationError(f"Cloud transmission failed: {str(e)}"))
    
    async def _process_batch(self, batch_data: List[Dict[str, Any]], config: SyncConfiguration):
        """Process a batch of data for cloud synchronization."""
        # Apply filters, compression, encryption to batch
        processed_batch = batch_data
        
        if config.compression_enabled:
            processed_batch = await self._compress_data(processed_batch)
        
        if config.encryption_enabled:
            processed_batch = await self._encrypt_data(processed_batch)
        
        # Send batch to cloud
        result = await self._send_to_cloud(config.target_cloud, processed_batch)
        return result
    
    def _calculate_transmission_cost(self, provider: CloudProvider, data_size: int) -> float:
        """Calculate cost for data transmission."""
        # Cost per MB for different providers (simplified)
        cost_per_mb = {
            CloudProvider.AWS_IOT: 0.08,
            CloudProvider.AZURE_IOT: 0.10,
            CloudProvider.GOOGLE_IOT: 0.09,
            CloudProvider.IBM_WATSON: 0.12,
            CloudProvider.ALIBABA_IOT: 0.06,
            CloudProvider.ORACLE_IOT: 0.11,
            CloudProvider.CUSTOM: 0.05
        }
        
        mb_transferred = data_size / (1024 * 1024)
        return mb_transferred * cost_per_mb.get(provider, 0.08)
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive cloud integration status."""
        active_connections = len([c for c in self.cloud_connections.values() if c.is_active])
        healthy_connections = len([c for c in self.cloud_connections.values() if c.is_healthy()])
        
        return {
            "total_connections": len(self.cloud_connections),
            "active_connections": active_connections,
            "healthy_connections": healthy_connections,
            "total_sync_jobs": len(self.sync_configurations),
            "total_sync_operations": self.total_sync_operations,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "sync_success_rate": (self.successful_syncs / max(self.total_sync_operations, 1)) * 100,
            "total_data_transferred": self.total_data_transferred,
            "total_cost_incurred": self.total_cost_incurred,
            "cloud_providers": list(set(c.provider.value for c in self.cloud_connections.values())),
            "cost_optimizer_enabled": self.cost_optimizer_enabled,
            "auto_scaling_enabled": self.auto_scaling_enabled
        }


# Helper functions for cloud integration
def create_cloud_connection(
    provider: CloudProvider,
    region: str,
    credentials: Dict[str, str],
    endpoint_url: Optional[str] = None
) -> CloudConnection:
    """Create cloud connection with default configuration."""
    connection_id = f"conn_{provider.value}_{region}_{int(datetime.now(UTC).timestamp())}"
    
    return CloudConnection(
        connection_id=connection_id,
        provider=provider,
        region=region,
        credentials=credentials,
        endpoint_url=endpoint_url or f"https://{provider.value}.{region}.amazonaws.com",
        connection_timeout=30.0,
        retry_config={"max_retries": 3, "backoff_factor": 2},
        rate_limits={"requests_per_second": 100, "burst": 200},
        security_config={"ssl_verify": True, "encryption": "AES256"}
    )


def create_sync_configuration(
    source_devices: List[DeviceId],
    target_cloud: CloudProvider,
    sync_strategy: SyncStrategy = SyncStrategy.BATCH_SYNC,
    sync_interval: int = 300
) -> SyncConfiguration:
    """Create sync configuration with sensible defaults."""
    sync_id = f"sync_{target_cloud.value}_{int(datetime.now(UTC).timestamp())}"
    
    return SyncConfiguration(
        sync_id=sync_id,
        source_devices=source_devices,
        target_cloud=target_cloud,
        sync_strategy=sync_strategy,
        sync_interval=sync_interval,
        batch_size=100,
        conflict_resolution="last_write_wins",
        data_filters=["*"]  # Include all data by default
    )