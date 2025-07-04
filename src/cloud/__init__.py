"""
Cloud integration package for multi-cloud platform automation.

This package provides comprehensive cloud integration capabilities including
AWS, Azure, Google Cloud Platform connectivity, multi-cloud orchestration,
cost optimization, and enterprise-grade security management.
"""

from .cloud_connector_manager import CloudConnectorManager, get_cloud_manager, initialize_cloud_manager
from .aws_connector import AWSConnector
from .azure_connector import AzureConnector
from .gcp_connector import GCPConnector
from .cloud_orchestrator import CloudOrchestrator
from .cost_optimizer import CloudCostOptimizer

__all__ = [
    'CloudConnectorManager',
    'get_cloud_manager', 
    'initialize_cloud_manager',
    'AWSConnector',
    'AzureConnector', 
    'GCPConnector',
    'CloudOrchestrator',
    'CloudCostOptimizer'
]