"""
DevOps integration package for developer toolkit and CI/CD automation.

This package provides comprehensive DevOps capabilities including:
- Git integration and version control automation
- CI/CD pipeline automation and deployment management
- API management, documentation, and governance
- Infrastructure as Code (IaC) support
- Developer collaboration and code quality automation
"""

from .api_manager import APIDocumentation, APIGovernance, APIManager
from .cicd_pipeline import BuildResult, CICDPipeline, PipelineConfig
from .deployment_engine import DeploymentEngine, DeploymentResult, DeploymentStrategy
from .git_connector import BranchInfo, GitConnector, GitOperation

__all__ = [
    # Core classes
    "GitConnector",
    "CICDPipeline",
    "APIManager",
    "DeploymentEngine",
    # Data structures
    "GitOperation",
    "BranchInfo",
    "PipelineConfig",
    "BuildResult",
    "APIDocumentation",
    "APIGovernance",
    "DeploymentStrategy",
    "DeploymentResult",
]
