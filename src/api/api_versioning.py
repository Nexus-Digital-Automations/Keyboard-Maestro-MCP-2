"""
API Versioning - TASK_64 Phase 4 Advanced Features

API version management and backward compatibility for API orchestration.
Provides intelligent version routing with deprecation management and compatibility layers.

Architecture: Version Management + Compatibility + Deprecation + Migration + Semantic Versioning
Performance: <25ms version resolution, <50ms compatibility check, <100ms migration mapping
Compatibility: Semantic versioning, backward compatibility, graceful deprecation, migration paths
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import re
import asyncio
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.api_orchestration_architecture import (
    ServiceId, APIOrchestrationError, create_service_id
)


class VersioningStrategy(Enum):
    """API versioning strategies."""
    SEMANTIC = "semantic"                  # Semantic versioning (1.2.3)
    CALENDAR = "calendar"                  # Calendar versioning (2024.01.15)
    SEQUENTIAL = "sequential"              # Sequential versioning (v1, v2, v3)
    CUSTOM = "custom"                      # Custom versioning scheme


class VersionStatus(Enum):
    """Version lifecycle status."""
    DEVELOPMENT = "development"            # In development
    PREVIEW = "preview"                    # Preview/beta release
    STABLE = "stable"                      # Stable release
    DEPRECATED = "deprecated"              # Deprecated but supported
    SUNSET = "sunset"                      # End of life, no longer supported


class CompatibilityLevel(Enum):
    """API compatibility levels."""
    MAJOR = "major"                        # Breaking changes
    MINOR = "minor"                        # New features, backward compatible
    PATCH = "patch"                        # Bug fixes, backward compatible
    INCOMPATIBLE = "incompatible"          # No compatibility


@dataclass
class APIVersion:
    """API version definition."""
    version_id: str
    version_string: str                    # e.g., "1.2.3", "2024.01.15"
    service_id: ServiceId
    
    # Version metadata
    major: int = 1
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None       # alpha, beta, rc
    build_metadata: Optional[str] = None
    
    # Lifecycle information
    status: VersionStatus = VersionStatus.DEVELOPMENT
    release_date: Optional[datetime] = None
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    
    # Compatibility
    compatible_versions: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    bug_fixes: List[str] = field(default_factory=list)
    
    # API specification
    openapi_spec_url: Optional[str] = None
    documentation_url: Optional[str] = None
    changelog_url: Optional[str] = None
    
    # Performance characteristics
    performance_profile: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with(self, other_version: str) -> bool:
        """Check if this version is compatible with another version."""
        return other_version in self.compatible_versions
    
    def is_newer_than(self, other: 'APIVersion') -> bool:
        """Check if this version is newer than another version."""
        if self.major != other.major:
            return self.major > other.major
        if self.minor != other.minor:
            return self.minor > other.minor
        return self.patch > other.patch
    
    def get_compatibility_level(self, other: 'APIVersion') -> CompatibilityLevel:
        """Get compatibility level with another version."""
        if self.major != other.major:
            return CompatibilityLevel.MAJOR
        elif self.minor != other.minor:
            return CompatibilityLevel.MINOR
        elif self.patch != other.patch:
            return CompatibilityLevel.PATCH
        else:
            return CompatibilityLevel.PATCH  # Same version


@dataclass
class VersionMigration:
    """Version migration definition."""
    migration_id: str
    from_version: str
    to_version: str
    service_id: ServiceId
    
    # Migration configuration
    migration_type: str = "automatic"      # automatic, manual, assisted
    migration_strategy: str = "gradual"    # gradual, immediate, parallel
    rollback_supported: bool = True
    
    # Migration rules
    field_mappings: Dict[str, str] = field(default_factory=dict)
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Migration scripts/functions
    pre_migration_hooks: List[str] = field(default_factory=list)
    post_migration_hooks: List[str] = field(default_factory=list)
    rollback_hooks: List[str] = field(default_factory=list)
    
    # Performance and safety
    max_batch_size: int = 1000
    migration_timeout_ms: int = 300000
    error_threshold: float = 0.05           # 5% error rate threshold
    
    # Status tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_rate: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionRoute:
    """Version-specific routing configuration."""
    route_id: str
    service_id: ServiceId
    version_pattern: str                   # e.g., "1.*", ">=2.0.0"
    target_version: str
    
    # Routing configuration
    weight: float = 1.0                    # Traffic percentage
    enabled: bool = True
    
    # Request transformation
    request_transformations: List[Dict[str, Any]] = field(default_factory=list)
    response_transformations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Fallback configuration
    fallback_version: Optional[str] = None
    fallback_enabled: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersioningDecision:
    """Record of a versioning decision."""
    decision_id: str
    timestamp: datetime
    requested_version: str
    resolved_version: str
    service_id: ServiceId
    decision_strategy: str
    compatibility_level: CompatibilityLevel
    migration_applied: bool = False
    migration_id: Optional[str] = None
    decision_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIVersionManager:
    """Advanced API version management system."""
    
    def __init__(self):
        self.versions: Dict[str, APIVersion] = {}                    # version_id -> APIVersion
        self.service_versions: Dict[ServiceId, List[str]] = {}       # service_id -> [version_ids]
        self.migrations: Dict[str, VersionMigration] = {}           # migration_id -> VersionMigration
        self.routes: Dict[str, VersionRoute] = {}                   # route_id -> VersionRoute
        self.decision_history: List[VersioningDecision] = []
        
        # Version resolution cache
        self.resolution_cache: Dict[str, str] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Performance metrics
        self.metrics = {
            "total_resolutions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "migrations_applied": 0,
            "compatibility_checks": 0,
            "average_resolution_time_ms": 0.0
        }
        
        # Default versioning strategy
        self.default_strategy = VersioningStrategy.SEMANTIC
    
    @require(lambda version: isinstance(version, APIVersion))
    def register_version(self, version: APIVersion) -> Either[APIOrchestrationError, bool]:
        """Register a new API version."""
        try:
            # Validate version format
            if not self._validate_version_format(version.version_string):
                return Either.error(APIOrchestrationError(f"Invalid version format: {version.version_string}"))
            
            # Register version
            self.versions[version.version_id] = version
            
            # Add to service versions
            if version.service_id not in self.service_versions:
                self.service_versions[version.service_id] = []
            self.service_versions[version.service_id].append(version.version_id)
            
            # Sort service versions by semantic order
            self._sort_service_versions(version.service_id)
            
            # Clear related cache entries
            self._invalidate_cache(version.service_id)
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Version registration failed: {str(e)}"))
    
    @require(lambda migration: isinstance(migration, VersionMigration))
    def register_migration(self, migration: VersionMigration) -> Either[APIOrchestrationError, bool]:
        """Register a version migration."""
        try:
            # Validate migration versions exist
            from_exists = any(v.version_string == migration.from_version for v in self.versions.values())
            to_exists = any(v.version_string == migration.to_version for v in self.versions.values())
            
            if not from_exists:
                return Either.error(APIOrchestrationError(f"Source version not found: {migration.from_version}"))
            if not to_exists:
                return Either.error(APIOrchestrationError(f"Target version not found: {migration.to_version}"))
            
            self.migrations[migration.migration_id] = migration
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Migration registration failed: {str(e)}"))
    
    @require(lambda route: isinstance(route, VersionRoute))
    def register_route(self, route: VersionRoute) -> Either[APIOrchestrationError, bool]:
        """Register a version route."""
        try:
            self.routes[route.route_id] = route
            return Either.success(True)
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Route registration failed: {str(e)}"))
    
    @require(lambda service_id: isinstance(service_id, ServiceId))
    @require(lambda requested_version: isinstance(requested_version, str))
    async def resolve_version(self, service_id: ServiceId, requested_version: str, request_context: Optional[Dict[str, Any]] = None) -> Either[APIOrchestrationError, VersioningDecision]:
        """
        Resolve requested version to actual version with compatibility checks.
        
        Args:
            service_id: Service identifier
            requested_version: Requested version string
            request_context: Additional request context
            
        Returns:
            Either API orchestration error or versioning decision
        """
        try:
            resolution_start = time.time()
            self.metrics["total_resolutions"] += 1
            
            # Check cache first
            cache_key = f"{service_id}:{requested_version}"
            cached_result = self._get_cached_resolution(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return Either.success(cached_result)
            
            self.metrics["cache_misses"] += 1
            
            # Get available versions for service
            if service_id not in self.service_versions:
                return Either.error(APIOrchestrationError(f"No versions found for service: {service_id}"))
            
            available_versions = [self.versions[vid] for vid in self.service_versions[service_id]]
            
            # Find best matching version
            resolved_version = await self._find_best_version_match(requested_version, available_versions, request_context)
            
            if not resolved_version:
                return Either.error(APIOrchestrationError(f"No compatible version found for: {requested_version}"))
            
            # Check if migration is needed
            migration_needed = resolved_version.version_string != requested_version
            migration_id = None
            
            if migration_needed:
                migration = self._find_migration(requested_version, resolved_version.version_string, service_id)
                if migration:
                    migration_id = migration.migration_id
                    self.metrics["migrations_applied"] += 1
            
            # Determine compatibility level
            compatibility_level = self._get_compatibility_level(requested_version, resolved_version.version_string)
            
            # Create decision record
            resolution_time = (time.time() - resolution_start) * 1000
            decision = VersioningDecision(
                decision_id=f"{service_id}_{requested_version}_{int(time.time() * 1000)}",
                timestamp=datetime.now(UTC),
                requested_version=requested_version,
                resolved_version=resolved_version.version_string,
                service_id=service_id,
                decision_strategy=self.default_strategy.value,
                compatibility_level=compatibility_level,
                migration_applied=migration_needed,
                migration_id=migration_id,
                decision_time_ms=resolution_time,
                metadata=request_context or {}
            )
            
            # Cache the decision
            self._cache_resolution(cache_key, decision)
            
            # Update metrics
            current_avg = self.metrics["average_resolution_time_ms"]
            total_resolutions = self.metrics["total_resolutions"]
            self.metrics["average_resolution_time_ms"] = (current_avg * (total_resolutions - 1) + resolution_time) / total_resolutions
            
            # Store in decision history
            self.decision_history.append(decision)
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
            
            return Either.success(decision)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Version resolution failed: {str(e)}"))
    
    async def _find_best_version_match(self, requested_version: str, available_versions: List[APIVersion], request_context: Optional[Dict[str, Any]]) -> Optional[APIVersion]:
        """Find the best matching version from available versions."""
        if not available_versions:
            return None
        
        # First try exact match
        for version in available_versions:
            if version.version_string == requested_version:
                return version
        
        # Try semantic version matching
        if self._is_semantic_version(requested_version):
            best_match = self._find_semantic_match(requested_version, available_versions)
            if best_match:
                return best_match
        
        # Try latest stable version
        stable_versions = [v for v in available_versions if v.status == VersionStatus.STABLE]
        if stable_versions:
            return max(stable_versions, key=lambda v: (v.major, v.minor, v.patch))
        
        # Fallback to latest version
        return max(available_versions, key=lambda v: (v.major, v.minor, v.patch))
    
    def _find_semantic_match(self, requested_version: str, available_versions: List[APIVersion]) -> Optional[APIVersion]:
        """Find semantic version match using version constraints."""
        # Parse requested version constraint (e.g., ">=1.2.0", "~1.2.0", "^1.2.0")
        constraint_match = re.match(r'^([~^>=<]*)(\d+)\.(\d+)\.(\d+)', requested_version)
        if not constraint_match:
            return None
        
        operator = constraint_match.group(1) or "="
        req_major = int(constraint_match.group(2))
        req_minor = int(constraint_match.group(3))
        req_patch = int(constraint_match.group(4))
        
        compatible_versions = []
        
        for version in available_versions:
            if self._version_satisfies_constraint(version, operator, req_major, req_minor, req_patch):
                compatible_versions.append(version)
        
        if not compatible_versions:
            return None
        
        # Return latest compatible version
        return max(compatible_versions, key=lambda v: (v.major, v.minor, v.patch))
    
    def _version_satisfies_constraint(self, version: APIVersion, operator: str, req_major: int, req_minor: int, req_patch: int) -> bool:
        """Check if version satisfies semantic version constraint."""
        if operator == "=" or operator == "":
            return version.major == req_major and version.minor == req_minor and version.patch == req_patch
        elif operator == ">=":
            return (version.major, version.minor, version.patch) >= (req_major, req_minor, req_patch)
        elif operator == ">":
            return (version.major, version.minor, version.patch) > (req_major, req_minor, req_patch)
        elif operator == "<=":
            return (version.major, version.minor, version.patch) <= (req_major, req_minor, req_patch)
        elif operator == "<":
            return (version.major, version.minor, version.patch) < (req_major, req_minor, req_patch)
        elif operator == "~":  # Compatible within minor version
            return version.major == req_major and version.minor == req_minor and version.patch >= req_patch
        elif operator == "^":  # Compatible within major version
            return version.major == req_major and (version.minor, version.patch) >= (req_minor, req_patch)
        
        return False
    
    def _find_migration(self, from_version: str, to_version: str, service_id: ServiceId) -> Optional[VersionMigration]:
        """Find migration path between versions."""
        for migration in self.migrations.values():
            if (migration.from_version == from_version and 
                migration.to_version == to_version and 
                migration.service_id == service_id):
                return migration
        return None
    
    def _get_compatibility_level(self, requested_version: str, resolved_version: str) -> CompatibilityLevel:
        """Determine compatibility level between versions."""
        if requested_version == resolved_version:
            return CompatibilityLevel.PATCH
        
        if self._is_semantic_version(requested_version) and self._is_semantic_version(resolved_version):
            req_parts = self._parse_semantic_version(requested_version)
            res_parts = self._parse_semantic_version(resolved_version)
            
            if req_parts and res_parts:
                if req_parts[0] != res_parts[0]:  # Major version difference
                    return CompatibilityLevel.MAJOR
                elif req_parts[1] != res_parts[1]:  # Minor version difference
                    return CompatibilityLevel.MINOR
                else:  # Patch version difference
                    return CompatibilityLevel.PATCH
        
        return CompatibilityLevel.INCOMPATIBLE
    
    def _validate_version_format(self, version_string: str) -> bool:
        """Validate version string format."""
        # Semantic version pattern
        semantic_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?(\+[a-zA-Z0-9]+)?$'
        if re.match(semantic_pattern, version_string):
            return True
        
        # Calendar version pattern
        calendar_pattern = r'^\d{4}\.\d{2}\.\d{2}$'
        if re.match(calendar_pattern, version_string):
            return True
        
        # Sequential version pattern
        sequential_pattern = r'^v?\d+$'
        if re.match(sequential_pattern, version_string):
            return True
        
        return False
    
    def _is_semantic_version(self, version_string: str) -> bool:
        """Check if version string follows semantic versioning."""
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?(\+[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version_string))
    
    def _parse_semantic_version(self, version_string: str) -> Optional[Tuple[int, int, int]]:
        """Parse semantic version into major.minor.patch tuple."""
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version_string)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return None
    
    def _sort_service_versions(self, service_id: ServiceId):
        """Sort service versions by semantic order."""
        if service_id not in self.service_versions:
            return
        
        version_ids = self.service_versions[service_id]
        versions = [self.versions[vid] for vid in version_ids]
        
        # Sort by major.minor.patch
        versions.sort(key=lambda v: (v.major, v.minor, v.patch))
        
        self.service_versions[service_id] = [v.version_id for v in versions]
    
    def _get_cached_resolution(self, cache_key: str) -> Optional[VersioningDecision]:
        """Get cached version resolution."""
        if cache_key not in self.resolution_cache:
            return None
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time or (datetime.now(UTC) - cache_time).total_seconds() > self.cache_ttl_seconds:
            # Cache expired
            del self.resolution_cache[cache_key]
            if cache_key in self.cache_timestamps:
                del self.cache_timestamps[cache_key]
            return None
        
        # Return cached decision (would need to reconstruct VersioningDecision)
        return None  # Simplified for now
    
    def _cache_resolution(self, cache_key: str, decision: VersioningDecision):
        """Cache version resolution decision."""
        self.resolution_cache[cache_key] = decision.resolved_version
        self.cache_timestamps[cache_key] = datetime.now(UTC)
        
        # Limit cache size
        if len(self.resolution_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:100]
            for key, _ in oldest_keys:
                self.resolution_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
    
    def _invalidate_cache(self, service_id: ServiceId):
        """Invalidate cache entries for a service."""
        keys_to_remove = [key for key in self.resolution_cache.keys() if key.startswith(f"{service_id}:")]
        for key in keys_to_remove:
            self.resolution_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def get_service_versions(self, service_id: ServiceId, status_filter: Optional[VersionStatus] = None) -> List[APIVersion]:
        """Get all versions for a service, optionally filtered by status."""
        if service_id not in self.service_versions:
            return []
        
        versions = [self.versions[vid] for vid in self.service_versions[service_id]]
        
        if status_filter:
            versions = [v for v in versions if v.status == status_filter]
        
        return versions
    
    def get_deprecated_versions(self, days_until_sunset: int = 30) -> List[APIVersion]:
        """Get versions that are deprecated or approaching sunset."""
        now = datetime.now(UTC)
        sunset_threshold = now + timedelta(days=days_until_sunset)
        
        deprecated_versions = []
        for version in self.versions.values():
            if version.status == VersionStatus.DEPRECATED:
                deprecated_versions.append(version)
            elif version.sunset_date and version.sunset_date <= sunset_threshold:
                deprecated_versions.append(version)
        
        return deprecated_versions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get version manager metrics."""
        total_versions = len(self.versions)
        active_services = len(self.service_versions)
        total_migrations = len(self.migrations)
        
        # Version status distribution
        status_counts = {}
        for status in VersionStatus:
            status_counts[status.value] = len([v for v in self.versions.values() if v.status == status])
        
        return {
            **self.metrics,
            "total_versions": total_versions,
            "active_services": active_services,
            "total_migrations": total_migrations,
            "total_routes": len(self.routes),
            "cache_size": len(self.resolution_cache),
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_resolutions"]),
            "version_status_distribution": status_counts,
            "decisions_made": len(self.decision_history)
        }


# Export the versioning classes
__all__ = [
    "APIVersionManager", "APIVersion", "VersionMigration", "VersionRoute", 
    "VersioningDecision", "VersioningStrategy", "VersionStatus", "CompatibilityLevel"
]