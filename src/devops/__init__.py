"""
DevOps integration package for developer toolkit and CI/CD automation.

This package provides comprehensive DevOps capabilities including:
- Git integration and version control automation
- CI/CD pipeline automation and deployment management
- API management, documentation, and governance
- Infrastructure as Code (IaC) support
- Developer collaboration and code quality automation
"""

from .git_connector import GitConnector, GitOperation, BranchInfo
from .cicd_pipeline import CICDPipeline, PipelineConfig, BuildResult
from .api_manager import APIManager, APIDocumentation, APIGovernance
from .deployment_engine import DeploymentEngine, DeploymentStrategy, DeploymentResult

__all__ = [
    # Core classes
    "GitConnector", "CICDPipeline", "APIManager", "DeploymentEngine",
    
    # Data structures
    "GitOperation", "BranchInfo", "PipelineConfig", "BuildResult",
    "APIDocumentation", "APIGovernance", "DeploymentStrategy", "DeploymentResult"
]