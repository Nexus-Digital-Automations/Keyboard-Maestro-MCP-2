"""
Developer toolkit type definitions for comprehensive DevOps integration.

This module provides enterprise-grade developer toolkit architecture including:
- Git operations and version control automation
- CI/CD pipeline management and deployment automation  
- API management, documentation, and governance
- Code quality automation and security scanning
- Infrastructure as Code operations and management

Security: Enterprise-grade authentication, secure credential management, audit logging.
Performance: <2s Git operations, <5s pipeline execution, <1s API discovery.
Type Safety: Complete branded type system with contracts and validation.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import uuid

from .contracts import require, ensure
from .either import Either
from .errors import ValidationError


# Branded types for developer toolkit operations
class GitRepositoryId(str):
    """Branded type for Git repository identifiers."""
    pass

class PipelineId(str):
    """Branded type for CI/CD pipeline identifiers."""
    pass

class DeploymentId(str):
    """Branded type for deployment identifiers."""
    pass

class APIEndpointId(str):
    """Branded type for API endpoint identifiers."""
    pass

class CodeReviewId(str):
    """Branded type for code review identifiers."""
    pass


def create_git_repository_id() -> GitRepositoryId:
    """Create a new Git repository identifier."""
    return GitRepositoryId(f"git_repo_{uuid.uuid4().hex[:12]}")

def create_pipeline_id() -> PipelineId:
    """Create a new CI/CD pipeline identifier."""
    return PipelineId(f"pipeline_{uuid.uuid4().hex[:12]}")

def create_deployment_id() -> DeploymentId:
    """Create a new deployment identifier."""
    return DeploymentId(f"deploy_{uuid.uuid4().hex[:12]}")

def create_api_endpoint_id() -> APIEndpointId:
    """Create a new API endpoint identifier."""
    return APIEndpointId(f"api_{uuid.uuid4().hex[:12]}")

def create_code_review_id() -> CodeReviewId:
    """Create a new code review identifier."""
    return CodeReviewId(f"review_{uuid.uuid4().hex[:12]}")


class GitOperation(Enum):
    """Git operations for version control automation."""
    CLONE = "clone"
    COMMIT = "commit"
    PUSH = "push"
    PULL = "pull"
    BRANCH = "branch"
    MERGE = "merge"
    STATUS = "status"
    LOG = "log"
    DIFF = "diff"
    TAG = "tag"
    RESET = "reset"
    STASH = "stash"


class PipelineAction(Enum):
    """CI/CD pipeline actions."""
    CREATE = "create"
    EXECUTE = "execute"
    MONITOR = "monitor"
    CONFIGURE = "configure"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    RETRY = "retry"


class DeploymentStrategy(Enum):
    """Deployment strategies for automated deployments."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class APIOperation(Enum):
    """API management operations."""
    DISCOVER = "discover"
    DOCUMENT = "document"
    TEST = "test"
    GOVERN = "govern"
    MONITOR = "monitor"
    VERSION = "version"
    DEPRECATE = "deprecate"


class CodeQualityCheck(Enum):
    """Code quality analysis types."""
    LINTING = "linting"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    DEPENDENCIES = "dependencies"


class IaCProvider(Enum):
    """Infrastructure as Code providers."""
    TERRAFORM = "terraform"
    CLOUDFORMATION = "cloudformation"
    PULUMI = "pulumi"
    ANSIBLE = "ansible"
    KUBERNETES = "kubernetes"
    HELM = "helm"


@dataclass
class GitRepository:
    """Git repository configuration and metadata."""
    repository_id: GitRepositoryId
    name: str
    url: str
    local_path: Optional[str]
    default_branch: str
    authentication: Dict[str, str]
    collaborators: List[str]
    last_sync: Optional[datetime]
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: self.url.startswith(('https://', 'git@', 'ssh://')))
    def __post_init__(self):
        pass


@dataclass
class GitCommit:
    """Git commit information and metadata."""
    commit_id: str
    message: str
    author: str
    timestamp: datetime
    branch: str
    files_changed: List[str]
    additions: int
    deletions: int
    
    @require(lambda self: len(self.commit_id) >= 7)
    @require(lambda self: len(self.message.strip()) > 0)
    def __post_init__(self):
        pass


@dataclass
class CICDPipeline:
    """CI/CD pipeline configuration and state."""
    pipeline_id: PipelineId
    name: str
    repository: GitRepositoryId
    configuration: Dict[str, Any]
    stages: List[Dict[str, Any]]
    triggers: List[str]
    environment_variables: Dict[str, str]
    status: str
    last_execution: Optional[datetime]
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.stages) > 0)
    def __post_init__(self):
        pass


@dataclass
class DeploymentConfiguration:
    """Deployment configuration and environment settings."""
    deployment_id: DeploymentId
    name: str
    environment: str
    strategy: DeploymentStrategy
    target_instances: int
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: self.target_instances > 0)
    def __post_init__(self):
        pass


@dataclass
class APIEndpoint:
    """API endpoint specification and metadata."""
    endpoint_id: APIEndpointId
    path: str
    method: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    authentication_required: bool
    rate_limiting: Optional[Dict[str, int]]
    version: str
    
    @require(lambda self: self.path.startswith('/'))
    @require(lambda self: self.method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
    def __post_init__(self):
        pass


@dataclass
class CodeQualityReport:
    """Code quality analysis results."""
    analysis_id: str
    timestamp: datetime
    scope: str
    quality_score: float
    issues_found: List[Dict[str, Any]]
    security_vulnerabilities: List[Dict[str, Any]]
    performance_recommendations: List[str]
    complexity_metrics: Dict[str, float]
    coverage_percentage: float
    
    @require(lambda self: 0.0 <= self.quality_score <= 1.0)
    @require(lambda self: 0.0 <= self.coverage_percentage <= 100.0)
    def __post_init__(self):
        pass


@dataclass
class InfrastructureResource:
    """Infrastructure resource definition and state."""
    resource_id: str
    resource_type: str
    provider: IaCProvider
    configuration: Dict[str, Any]
    state: str
    dependencies: List[str]
    cost_estimate: Optional[float]
    last_updated: datetime
    
    @require(lambda self: len(self.resource_id) > 0)
    @require(lambda self: len(self.resource_type) > 0)
    def __post_init__(self):
        pass


class DeveloperToolkitError(Exception):
    """Base exception for developer toolkit operations."""
    
    def __init__(self, message: str, error_code: str = "DEVOPS_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
    
    @classmethod
    def git_operation_failed(cls, operation: str, details: str) -> 'DeveloperToolkitError':
        return cls(
            f"Git operation '{operation}' failed: {details}",
            "GIT_OPERATION_FAILED",
            {"operation": operation, "details": details}
        )
    
    @classmethod
    def pipeline_execution_failed(cls, pipeline_id: str, stage: str) -> 'DeveloperToolkitError':
        return cls(
            f"Pipeline '{pipeline_id}' failed at stage '{stage}'",
            "PIPELINE_EXECUTION_FAILED",
            {"pipeline_id": pipeline_id, "stage": stage}
        )
    
    @classmethod
    def deployment_failed(cls, deployment_id: str, environment: str) -> 'DeveloperToolkitError':
        return cls(
            f"Deployment '{deployment_id}' failed in environment '{environment}'",
            "DEPLOYMENT_FAILED",
            {"deployment_id": deployment_id, "environment": environment}
        )
    
    @classmethod
    def api_operation_failed(cls, operation: str, endpoint: str) -> 'DeveloperToolkitError':
        return cls(
            f"API operation '{operation}' failed for endpoint '{endpoint}'",
            "API_OPERATION_FAILED",
            {"operation": operation, "endpoint": endpoint}
        )
    
    @classmethod
    def code_quality_analysis_failed(cls, scope: str, reason: str) -> 'DeveloperToolkitError':
        return cls(
            f"Code quality analysis failed for scope '{scope}': {reason}",
            "CODE_QUALITY_ANALYSIS_FAILED",
            {"scope": scope, "reason": reason}
        )


# Utility functions for common operations
def validate_git_url(url: str) -> bool:
    """Validate Git repository URL format."""
    valid_prefixes = ['https://', 'git@', 'ssh://']
    return any(url.startswith(prefix) for prefix in valid_prefixes)

def calculate_deployment_risk_score(config: DeploymentConfiguration) -> float:
    """Calculate deployment risk score based on configuration."""
    base_risk = 0.1
    
    # Strategy risk
    strategy_risks = {
        DeploymentStrategy.RECREATE: 0.8,
        DeploymentStrategy.ROLLING: 0.3,
        DeploymentStrategy.BLUE_GREEN: 0.1,
        DeploymentStrategy.CANARY: 0.2,
        DeploymentStrategy.A_B_TESTING: 0.4
    }
    
    risk_score = base_risk + strategy_risks.get(config.strategy, 0.5)
    
    # Environment risk
    env_risks = {"production": 0.3, "staging": 0.1, "development": 0.05}
    risk_score += env_risks.get(config.environment.lower(), 0.2)
    
    # Health check availability
    if not config.health_check_config:
        risk_score += 0.2
    
    # Rollback capability
    if not config.rollback_config:
        risk_score += 0.3
    
    return min(1.0, risk_score)

def estimate_pipeline_duration(pipeline: CICDPipeline) -> int:
    """Estimate pipeline execution duration in minutes."""
    base_duration = 5  # Base overhead
    
    stage_durations = {
        "build": 10,
        "test": 15,
        "security_scan": 8,
        "deploy": 12,
        "integration_test": 20,
        "performance_test": 25
    }
    
    total_duration = base_duration
    for stage in pipeline.stages:
        stage_type = stage.get("type", "unknown").lower()
        total_duration += stage_durations.get(stage_type, 10)
    
    return total_duration

def calculate_api_complexity_score(endpoint: APIEndpoint) -> float:
    """Calculate API endpoint complexity score."""
    base_complexity = 0.1
    
    # Parameter complexity
    param_complexity = len(endpoint.parameters) * 0.1
    
    # Response complexity
    response_complexity = len(endpoint.responses) * 0.05
    
    # Authentication complexity
    auth_complexity = 0.2 if endpoint.authentication_required else 0.0
    
    # Rate limiting complexity
    rate_limit_complexity = 0.1 if endpoint.rate_limiting else 0.0
    
    total_complexity = base_complexity + param_complexity + response_complexity + auth_complexity + rate_limit_complexity
    return min(1.0, total_complexity)