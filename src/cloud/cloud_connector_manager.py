"""
Cloud connector manager for comprehensive multi-cloud platform integration.

This module provides the main manager for coordinating cloud connections,
orchestrating multi-cloud operations, and managing cloud resources with
enterprise-grade security and performance optimization.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, UTC
import asyncio
import logging

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.cloud_integration import (
    CloudProvider, CloudCredentials, CloudResource, CloudError
)
from .aws_connector import AWSConnector
from .azure_connector import AzureConnector  
from .gcp_connector import GCPConnector
from .cloud_orchestrator import CloudOrchestrator
from .cost_optimizer import CloudCostOptimizer

logger = logging.getLogger(__name__)

# Global cloud manager instance
_cloud_manager: Optional[CloudConnectorManager] = None


class CloudConnectorManager:
    """Comprehensive cloud integration management system."""
    
    def __init__(self):
        self.aws_connector = AWSConnector()
        self.azure_connector = AzureConnector()
        self.gcp_connector = GCPConnector()
        self.orchestrator = CloudOrchestrator()
        self.cost_optimizer = CloudCostOptimizer()
        
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.connection_pool: Dict[CloudProvider, List[str]] = {}
        self.initialized = False
        self.max_connections_per_provider = 5
        
        # Performance tracking
        self.operation_metrics: Dict[str, List[float]] = {}
        self.connection_cache: Dict[str, datetime] = {}
    
    async def initialize(self) -> Either[CloudError, None]:
        """Initialize cloud connector manager."""
        try:
            logger.info("Initializing cloud connector manager...")
            
            # Initialize connection pools
            for provider in CloudProvider:
                if provider not in [CloudProvider.GENERIC, CloudProvider.MULTI_CLOUD]:
                    self.connection_pool[provider] = []
            
            # Initialize sub-components
            await self.orchestrator.initialize()
            await self.cost_optimizer.initialize()
            
            self.initialized = True
            logger.info("Cloud connector manager initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Failed to initialize cloud connector manager: {e}")
            return Either.left(CloudError.connection_establishment_failed(str(e)))
    
    @require(lambda self: self.initialized)
    async def establish_cloud_connection(self, provider: CloudProvider, 
                                       credentials: CloudCredentials) -> Either[CloudError, str]:
        """Establish connection to cloud provider with session management."""
        try:
            # Check connection pool limits
            if len(self.connection_pool.get(provider, [])) >= self.max_connections_per_provider:
                # Clean up expired connections
                await self._cleanup_expired_connections(provider)
                
                if len(self.connection_pool.get(provider, [])) >= self.max_connections_per_provider:
                    return Either.left(CloudError.connection_failed("Maximum connections reached for provider"))
            
            # Validate credentials security
            if not self._validate_credentials_security(credentials):
                return Either.left(CloudError.insecure_auth_method(credentials.auth_method))
            
            start_time = datetime.now(UTC)
            
            # Establish connection based on provider
            if provider == CloudProvider.AWS:
                result = await self.aws_connector.connect(credentials)
            elif provider == CloudProvider.AZURE:
                result = await self.azure_connector.connect(credentials)
            elif provider == CloudProvider.GOOGLE_CLOUD:
                result = await self.gcp_connector.connect(credentials)
            else:
                return Either.left(CloudError.unsupported_provider(provider))
            
            if result.is_right():
                session_id = result.get_right()
                
                # Track session
                self.active_sessions[session_id] = {
                    "provider": provider,
                    "created_at": datetime.now(UTC),
                    "last_used": datetime.now(UTC),
                    "credentials_hash": credentials.get_credential_hash(),
                    "region": credentials.region
                }
                
                # Add to connection pool
                if provider not in self.connection_pool:
                    self.connection_pool[provider] = []
                self.connection_pool[provider].append(session_id)
                
                # Track performance
                duration = (datetime.now(UTC) - start_time).total_seconds()
                self._track_operation_metric("connection_time", duration)
                
                logger.info(f"Cloud connection established: {provider.value} - {session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error establishing cloud connection: {e}")
            return Either.left(CloudError.connection_establishment_failed(str(e)))
    
    async def sync_cloud_data(self, session_id: str, sync_config: Dict[str, Any]) -> Either[CloudError, Dict[str, Any]]:
        """Synchronize data with cloud storage."""
        try:
            if session_id not in self.active_sessions:
                return Either.left(CloudError.session_not_found(session_id))
            
            session_info = self.active_sessions[session_id]
            provider = session_info["provider"]
            
            # Update last used timestamp
            session_info["last_used"] = datetime.now(UTC)
            
            start_time = datetime.now(UTC)
            
            # Route to appropriate connector
            if provider == CloudProvider.AWS:
                result = await self.aws_connector.sync_storage_data(
                    session_id,
                    sync_config.get('source_path', ''),
                    sync_config.get('bucket_name', ''),
                    sync_config.get('destination_prefix', '')
                )
            elif provider == CloudProvider.AZURE:
                result = await self.azure_connector.sync_blob_data(
                    session_id,
                    sync_config.get('source_path', ''),
                    sync_config.get('container_name', ''),
                    sync_config.get('destination_prefix', '')
                )
            elif provider == CloudProvider.GOOGLE_CLOUD:
                result = await self.gcp_connector.sync_storage_data(
                    session_id,
                    sync_config.get('source_path', ''),
                    sync_config.get('bucket_name', ''),
                    sync_config.get('destination_prefix', '')
                )
            else:
                return Either.left(CloudError.sync_not_supported_for_provider(provider))
            
            # Track performance
            if result.is_right():
                duration = (datetime.now(UTC) - start_time).total_seconds()
                self._track_operation_metric("sync_time", duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cloud data sync: {e}")
            return Either.left(CloudError.sync_operation_failed(str(e)))
    
    async def get_monitoring_data(self, provider: CloudProvider, 
                                config: Dict[str, Any]) -> Either[CloudError, Dict[str, Any]]:
        """Get monitoring data for cloud resources."""
        try:
            # Find active sessions for provider
            provider_sessions = [
                session_id for session_id, info in self.active_sessions.items()
                if info["provider"] == provider
            ]
            
            if not provider_sessions:
                return Either.left(CloudError.session_not_found(f"No active sessions for {provider.value}"))
            
            monitoring_data = {
                "provider": provider.value,
                "active_sessions": len(provider_sessions),
                "resources": [],
                "performance_metrics": self._get_performance_summary(),
                "connection_health": self._get_connection_health(provider)
            }
            
            # Get resource monitoring from first available session
            session_id = provider_sessions[0]
            
            if provider == CloudProvider.AWS:
                resources_result = await self.aws_connector.list_resources(session_id)
            elif provider == CloudProvider.AZURE:
                resources_result = await self.azure_connector.list_resources(session_id)
            elif provider == CloudProvider.GOOGLE_CLOUD:
                resources_result = await self.gcp_connector.list_resources(session_id)
            else:
                return Either.left(CloudError.monitoring_failed(f"Monitoring not supported for {provider.value}"))
            
            if resources_result.is_right():
                monitoring_data["resources"] = resources_result.get_right()
            
            return Either.right(monitoring_data)
            
        except Exception as e:
            logger.error(f"Error getting monitoring data: {e}")
            return Either.left(CloudError.monitoring_failed(str(e)))
    
    async def disconnect_session(self, session_id: str) -> Either[CloudError, None]:
        """Disconnect cloud session and cleanup resources."""
        try:
            if session_id not in self.active_sessions:
                return Either.left(CloudError.session_not_found(session_id))
            
            session_info = self.active_sessions[session_id]
            provider = session_info["provider"]
            
            # Remove from connection pool
            if provider in self.connection_pool:
                if session_id in self.connection_pool[provider]:
                    self.connection_pool[provider].remove(session_id)
            
            # Disconnect from provider
            if provider == CloudProvider.AWS:
                await self.aws_connector.disconnect(session_id)
            elif provider == CloudProvider.AZURE:
                await self.azure_connector.disconnect(session_id)
            elif provider == CloudProvider.GOOGLE_CLOUD:
                await self.gcp_connector.disconnect(session_id)
            
            # Remove session
            del self.active_sessions[session_id]
            
            logger.info(f"Cloud session disconnected: {session_id}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Error disconnecting session: {e}")
            return Either.left(CloudError.connection_failed(str(e)))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "initialized": self.initialized,
            "active_sessions": len(self.active_sessions),
            "connection_pools": {
                provider.value: len(sessions) 
                for provider, sessions in self.connection_pool.items()
            },
            "performance_metrics": self._get_performance_summary(),
            "session_details": [
                {
                    "session_id": session_id,
                    "provider": info["provider"].value,
                    "created_at": info["created_at"].isoformat(),
                    "last_used": info["last_used"].isoformat(),
                    "region": info.get("region")
                }
                for session_id, info in self.active_sessions.items()
            ]
        }
    
    def _validate_credentials_security(self, credentials: CloudCredentials) -> bool:
        """Validate credentials meet security requirements."""
        # Check for secure authentication methods
        secure_methods = {
            CloudProvider.AWS: ['role_based', 'api_key'],
            CloudProvider.AZURE: ['service_account', 'managed_identity'],
            CloudProvider.GOOGLE_CLOUD: ['service_account']
        }
        
        provider_methods = secure_methods.get(credentials.provider, [])
        return credentials.auth_method.value in provider_methods
    
    async def _cleanup_expired_connections(self, provider: CloudProvider):
        """Clean up expired connections for provider."""
        if provider not in self.connection_pool:
            return
        
        current_time = datetime.now(UTC)
        expired_sessions = []
        
        for session_id in self.connection_pool[provider]:
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                last_used = session_info["last_used"]
                
                # Consider session expired after 1 hour of inactivity
                if current_time - last_used > timedelta(hours=1):
                    expired_sessions.append(session_id)
        
        # Disconnect expired sessions
        for session_id in expired_sessions:
            await self.disconnect_session(session_id)
    
    def _track_operation_metric(self, metric_name: str, value: float):
        """Track operation performance metrics."""
        if metric_name not in self.operation_metrics:
            self.operation_metrics[metric_name] = []
        
        self.operation_metrics[metric_name].append(value)
        
        # Keep only last 100 measurements
        if len(self.operation_metrics[metric_name]) > 100:
            self.operation_metrics[metric_name] = self.operation_metrics[metric_name][-100:]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for metric_name, values in self.operation_metrics.items():
            if values:
                summary[metric_name] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def _get_connection_health(self, provider: CloudProvider) -> Dict[str, Any]:
        """Get connection health status for provider."""
        provider_sessions = [
            session_id for session_id, info in self.active_sessions.items()
            if info["provider"] == provider
        ]
        
        if not provider_sessions:
            return {"status": "no_connections", "health_score": 0}
        
        # Simple health calculation based on active sessions and recent activity
        current_time = datetime.now(UTC)
        active_count = 0
        
        for session_id in provider_sessions:
            session_info = self.active_sessions[session_id]
            last_used = session_info["last_used"]
            
            # Consider active if used within last 10 minutes
            if current_time - last_used < timedelta(minutes=10):
                active_count += 1
        
        health_score = min(100, (active_count / len(provider_sessions)) * 100)
        
        return {
            "status": "healthy" if health_score > 70 else "degraded" if health_score > 30 else "unhealthy",
            "health_score": health_score,
            "active_sessions": active_count,
            "total_sessions": len(provider_sessions)
        }


async def initialize_cloud_manager() -> Either[CloudError, CloudConnectorManager]:
    """Initialize the global cloud manager instance."""
    global _cloud_manager
    
    if _cloud_manager is None:
        _cloud_manager = CloudConnectorManager()
        result = await _cloud_manager.initialize()
        
        if result.is_left():
            _cloud_manager = None
            return result
    
    return Either.right(_cloud_manager)


def get_cloud_manager() -> Optional[CloudConnectorManager]:
    """Get the global cloud manager instance."""
    return _cloud_manager