"""
Plugin marketplace system for discovery, installation, and management.

This module provides a comprehensive marketplace for plugins with search,
ratings, updates, and secure distribution capabilities.
"""

import asyncio
import json
import logging
import hashlib
import tempfile
import zipfile
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..core.plugin_architecture import (
    PluginMetadata, PluginId, PluginError, SecurityProfile
)
from ..core.either import Either
from ..core.errors import create_error_context
from .security_sandbox import PluginSecurityManager

logger = logging.getLogger(__name__)


class PluginCategory(Enum):
    """Plugin categories for marketplace organization."""
    PRODUCTIVITY = "productivity"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    UTILITY = "utility"
    ENTERTAINMENT = "entertainment"
    DEVELOPMENT = "development"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    FILE_MANAGEMENT = "file_management"
    TEXT_PROCESSING = "text_processing"


class PluginStatus(Enum):
    """Plugin status in marketplace."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass(frozen=True)
class PluginRating:
    """Plugin rating and review information."""
    rating: float  # 1.0 to 5.0
    review_count: int
    user_id: str
    review_text: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    helpful_votes: int = 0


@dataclass(frozen=True)
class PluginDownloadInfo:
    """Plugin download and distribution information."""
    download_url: str
    file_size: int
    checksum: str
    signature: Optional[str] = None
    mirrors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class MarketplaceEntry:
    """Complete marketplace entry for a plugin."""
    metadata: PluginMetadata
    category: PluginCategory
    status: PluginStatus
    download_info: PluginDownloadInfo
    rating: Optional[PluginRating] = None
    tags: Set[str] = field(default_factory=set)
    screenshots: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    support_url: Optional[str] = None
    license: str = "MIT"
    price: float = 0.0  # 0.0 for free plugins
    featured: bool = False
    verified_developer: bool = False
    
    def get_average_rating(self) -> float:
        """Get average rating for the plugin."""
        return self.rating.rating if self.rating else 0.0
    
    def is_free(self) -> bool:
        """Check if plugin is free."""
        return self.price == 0.0
    
    def get_security_level(self) -> SecurityProfile:
        """Get security level based on verification and review."""
        if self.verified_developer and self.status == PluginStatus.APPROVED:
            return SecurityProfile.STANDARD
        elif self.status == PluginStatus.APPROVED:
            return SecurityProfile.STRICT
        else:
            return SecurityProfile.SANDBOX


@dataclass
class SearchQuery:
    """Search query for marketplace plugins."""
    query: Optional[str] = None
    category: Optional[PluginCategory] = None
    tags: Set[str] = field(default_factory=set)
    min_rating: float = 0.0
    max_price: Optional[float] = None
    free_only: bool = False
    verified_only: bool = False
    sort_by: str = "relevance"  # relevance, rating, downloads, name, date
    limit: int = 20
    offset: int = 0


@dataclass
class InstallationProgress:
    """Progress tracking for plugin installation."""
    plugin_id: PluginId
    stage: str  # downloading, verifying, installing, configuring, completing
    progress_percent: float
    message: str
    started_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    error: Optional[str] = None


class PluginMarketplace:
    """Comprehensive plugin marketplace with discovery and management."""
    
    def __init__(self, marketplace_url: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.marketplace_url = marketplace_url or "https://api.km-mcp-plugins.com"
        self.cache_dir = cache_dir or Path.home() / ".km-mcp" / "marketplace_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.security_manager = PluginSecurityManager()
        self.cached_entries: Dict[PluginId, MarketplaceEntry] = {}
        self.download_cache: Dict[str, Path] = {}
        self.installation_progress: Dict[PluginId, InstallationProgress] = {}
        
        # Marketplace configuration
        self.config = {
            "cache_ttl_hours": 24,
            "max_download_size_mb": 50,
            "require_signature": True,
            "auto_update_check": True,
            "featured_limit": 10,
            "search_result_limit": 100
        }
    
    async def initialize(self) -> Either[PluginError, None]:
        """Initialize marketplace and load cached data."""
        try:
            await self._load_cache()
            logger.info("Plugin marketplace initialized")
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError.initialization_failed(f"Marketplace init failed: {str(e)}"))
    
    async def search_plugins(self, query: SearchQuery) -> Either[PluginError, List[MarketplaceEntry]]:
        """Search for plugins in the marketplace."""
        try:
            # Check cache first
            cached_results = await self._search_cache(query)
            if cached_results:
                return Either.right(cached_results)
            
            # Fetch from remote marketplace
            remote_results = await self._search_remote(query)
            if remote_results.is_left():
                return remote_results
            
            results = remote_results.get_right()
            
            # Cache results
            await self._cache_search_results(query, results)
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(PluginError(f"Search failed: {str(e)}", "SEARCH_ERROR"))
    
    async def get_plugin_details(self, plugin_id: PluginId) -> Either[PluginError, MarketplaceEntry]:
        """Get detailed information about a specific plugin."""
        try:
            # Check cache first
            if plugin_id in self.cached_entries:
                entry = self.cached_entries[plugin_id]
                if self._is_cache_valid(entry):
                    return Either.right(entry)
            
            # Fetch from remote
            remote_result = await self._fetch_plugin_details(plugin_id)
            if remote_result.is_left():
                return remote_result
            
            entry = remote_result.get_right()
            self.cached_entries[plugin_id] = entry
            
            return Either.right(entry)
            
        except Exception as e:
            return Either.left(PluginError(f"Failed to get plugin details: {str(e)}", "DETAILS_ERROR"))
    
    async def install_plugin(self, plugin_id: PluginId, target_dir: Path) -> Either[PluginError, Path]:
        """Install plugin from marketplace with progress tracking."""
        try:
            # Initialize progress tracking
            progress = InstallationProgress(
                plugin_id=plugin_id,
                stage="initializing",
                progress_percent=0.0,
                message="Preparing installation"
            )
            self.installation_progress[plugin_id] = progress
            
            # Get plugin details
            details_result = await self.get_plugin_details(plugin_id)
            if details_result.is_left():
                progress.error = details_result.get_left().message
                return details_result
            
            entry = details_result.get_right()
            
            # Security validation
            security_result = await self._validate_installation_security(entry)
            if security_result.is_left():
                progress.error = security_result.get_left().message
                return security_result
            
            # Download plugin
            progress.stage = "downloading"
            progress.message = f"Downloading {entry.metadata.name}"
            progress.progress_percent = 10.0
            
            download_result = await self._download_plugin(entry)
            if download_result.is_left():
                progress.error = download_result.get_left().message
                return download_result
            
            download_path = download_result.get_right()
            
            # Verify integrity
            progress.stage = "verifying"
            progress.message = "Verifying plugin integrity"
            progress.progress_percent = 40.0
            
            verify_result = await self._verify_plugin_integrity(entry, download_path)
            if verify_result.is_left():
                progress.error = verify_result.get_left().message
                return verify_result
            
            # Security scan
            progress.stage = "scanning"
            progress.message = "Performing security scan"
            progress.progress_percent = 60.0
            
            scan_result = await self._security_scan_plugin(download_path)
            if scan_result.is_left():
                progress.error = scan_result.get_left().message
                return scan_result
            
            # Install plugin
            progress.stage = "installing"
            progress.message = "Installing plugin files"
            progress.progress_percent = 80.0
            
            install_result = await self._install_plugin_files(entry, download_path, target_dir)
            if install_result.is_left():
                progress.error = install_result.get_left().message
                return install_result
            
            plugin_path = install_result.get_right()
            
            # Complete installation
            progress.stage = "completing"
            progress.message = "Finalizing installation"
            progress.progress_percent = 100.0
            
            # Record installation
            await self._record_installation(entry, plugin_path)
            
            logger.info(f"Plugin installed successfully: {plugin_id}")
            return Either.right(plugin_path)
            
        except Exception as e:
            error_msg = f"Installation failed: {str(e)}"
            if plugin_id in self.installation_progress:
                self.installation_progress[plugin_id].error = error_msg
            return Either.left(PluginError.installation_failed(error_msg))
    
    async def get_installation_progress(self, plugin_id: PluginId) -> Optional[InstallationProgress]:
        """Get installation progress for a plugin."""
        return self.installation_progress.get(plugin_id)
    
    async def get_featured_plugins(self) -> Either[PluginError, List[MarketplaceEntry]]:
        """Get featured plugins from marketplace."""
        query = SearchQuery(
            sort_by="featured",
            limit=self.config["featured_limit"]
        )
        
        search_result = await self.search_plugins(query)
        if search_result.is_left():
            return search_result
        
        featured = [entry for entry in search_result.get_right() if entry.featured]
        return Either.right(featured)
    
    async def get_plugin_updates(self, installed_plugins: Dict[PluginId, str]) -> Either[PluginError, List[MarketplaceEntry]]:
        """Check for updates to installed plugins."""
        try:
            updates = []
            
            for plugin_id, current_version in installed_plugins.items():
                details_result = await self.get_plugin_details(plugin_id)
                if details_result.is_left():
                    continue
                
                entry = details_result.get_right()
                if self._is_newer_version(entry.metadata.version, current_version):
                    updates.append(entry)
            
            return Either.right(updates)
            
        except Exception as e:
            return Either.left(PluginError(f"Update check failed: {str(e)}", "UPDATE_CHECK_ERROR"))
    
    async def _search_cache(self, query: SearchQuery) -> Optional[List[MarketplaceEntry]]:
        """Search cached entries."""
        if not self.cached_entries:
            return None
        
        results = []
        for entry in self.cached_entries.values():
            if self._matches_query(entry, query):
                results.append(entry)
        
        # Apply sorting and limits
        results = self._sort_results(results, query.sort_by)
        return results[query.offset:query.offset + query.limit]
    
    async def _search_remote(self, query: SearchQuery) -> Either[PluginError, List[MarketplaceEntry]]:
        """Search remote marketplace (mock implementation)."""
        # In a real implementation, this would make HTTP requests to the marketplace API
        # For now, return a mock result
        
        mock_plugins = [
            MarketplaceEntry(
                metadata=self._create_mock_metadata("text-processor", "Text Processor", "1.0.0"),
                category=PluginCategory.TEXT_PROCESSING,
                status=PluginStatus.APPROVED,
                download_info=PluginDownloadInfo(
                    download_url="https://example.com/plugins/text-processor.zip",
                    file_size=1024 * 100,  # 100KB
                    checksum="sha256:abcd1234..."
                ),
                rating=PluginRating(rating=4.5, review_count=42, user_id="system"),
                tags={"text", "processing", "utility"},
                featured=True,
                verified_developer=True
            ),
            MarketplaceEntry(
                metadata=self._create_mock_metadata("email-integration", "Email Integration", "2.1.0"),
                category=PluginCategory.COMMUNICATION,
                status=PluginStatus.APPROVED,
                download_info=PluginDownloadInfo(
                    download_url="https://example.com/plugins/email-integration.zip",
                    file_size=1024 * 250,  # 250KB
                    checksum="sha256:efgh5678..."
                ),
                rating=PluginRating(rating=4.2, review_count=28, user_id="system"),
                tags={"email", "communication", "automation"}
            )
        ]
        
        # Filter based on query
        filtered = [p for p in mock_plugins if self._matches_query(p, query)]
        return Either.right(filtered)
    
    def _create_mock_metadata(self, identifier: str, name: str, version: str) -> PluginMetadata:
        """Create mock plugin metadata for testing."""
        from ..core.plugin_architecture import PluginType, ApiVersion, PluginPermissions
        
        return PluginMetadata(
            identifier=PluginId(identifier),
            name=name,
            version=version,
            description=f"Mock plugin: {name}",
            author="Mock Developer",
            plugin_type=PluginType.UTILITY,
            api_version=ApiVersion.V1_0,
            permissions=PluginPermissions.standard()
        )
    
    def _matches_query(self, entry: MarketplaceEntry, query: SearchQuery) -> bool:
        """Check if plugin entry matches search query."""
        # Text search
        if query.query:
            search_text = query.query.lower()
            if not any(search_text in text.lower() for text in [
                entry.metadata.name,
                entry.metadata.description,
                entry.metadata.author
            ]):
                return False
        
        # Category filter
        if query.category and entry.category != query.category:
            return False
        
        # Tags filter
        if query.tags and not query.tags.intersection(entry.tags):
            return False
        
        # Rating filter
        if entry.get_average_rating() < query.min_rating:
            return False
        
        # Price filter
        if query.max_price is not None and entry.price > query.max_price:
            return False
        
        if query.free_only and not entry.is_free():
            return False
        
        # Verification filter
        if query.verified_only and not entry.verified_developer:
            return False
        
        return True
    
    def _sort_results(self, results: List[MarketplaceEntry], sort_by: str) -> List[MarketplaceEntry]:
        """Sort search results by specified criteria."""
        if sort_by == "rating":
            return sorted(results, key=lambda x: x.get_average_rating(), reverse=True)
        elif sort_by == "name":
            return sorted(results, key=lambda x: x.metadata.name)
        elif sort_by == "date":
            return sorted(results, key=lambda x: x.metadata.created_at, reverse=True)
        else:  # relevance (default)
            return results
    
    async def _fetch_plugin_details(self, plugin_id: PluginId) -> Either[PluginError, MarketplaceEntry]:
        """Fetch plugin details from remote marketplace."""
        # Mock implementation - would make HTTP request in real version
        if plugin_id == "text-processor":
            return Either.right(MarketplaceEntry(
                metadata=self._create_mock_metadata("text-processor", "Text Processor", "1.0.0"),
                category=PluginCategory.TEXT_PROCESSING,
                status=PluginStatus.APPROVED,
                download_info=PluginDownloadInfo(
                    download_url="https://example.com/plugins/text-processor.zip",
                    file_size=1024 * 100,
                    checksum="sha256:abcd1234..."
                )
            ))
        
        return Either.left(PluginError.plugin_not_found(plugin_id))
    
    async def _validate_installation_security(self, entry: MarketplaceEntry) -> Either[PluginError, None]:
        """Validate security requirements for installation."""
        # Check plugin status
        if entry.status != PluginStatus.APPROVED:
            return Either.left(PluginError.security_violation(f"Plugin not approved: {entry.status.value}"))
        
        # Check file size limits
        max_size = self.config["max_download_size_mb"] * 1024 * 1024
        if entry.download_info.file_size > max_size:
            return Either.left(PluginError.security_violation(f"Plugin too large: {entry.download_info.file_size} bytes"))
        
        # Check signature requirement
        if self.config["require_signature"] and not entry.download_info.signature:
            return Either.left(PluginError.security_violation("Plugin signature required but not provided"))
        
        return Either.right(None)
    
    async def _download_plugin(self, entry: MarketplaceEntry) -> Either[PluginError, Path]:
        """Download plugin file from marketplace."""
        try:
            # Check download cache
            cache_key = entry.download_info.checksum
            if cache_key in self.download_cache:
                cached_path = self.download_cache[cache_key]
                if cached_path.exists():
                    return Either.right(cached_path)
            
            # Create temporary download location
            download_dir = self.cache_dir / "downloads"
            download_dir.mkdir(exist_ok=True)
            
            file_name = f"{entry.metadata.identifier}-{entry.metadata.version}.zip"
            download_path = download_dir / file_name
            
            # Mock download - in real implementation would use HTTP client
            with open(download_path, 'wb') as f:
                # Write mock plugin content
                mock_content = b"PK\x03\x04"  # ZIP file signature
                f.write(mock_content)
            
            # Cache download
            self.download_cache[cache_key] = download_path
            
            return Either.right(download_path)
            
        except Exception as e:
            return Either.left(PluginError(f"Download failed: {str(e)}", "DOWNLOAD_ERROR"))
    
    async def _verify_plugin_integrity(self, entry: MarketplaceEntry, file_path: Path) -> Either[PluginError, None]:
        """Verify plugin file integrity."""
        try:
            # Calculate file checksum
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            calculated_checksum = f"sha256:{hasher.hexdigest()}"
            
            # Compare with expected checksum
            if calculated_checksum != entry.download_info.checksum:
                return Either.left(PluginError.security_violation(
                    f"Checksum mismatch: expected {entry.download_info.checksum}, got {calculated_checksum}"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError(f"Integrity verification failed: {str(e)}", "INTEGRITY_ERROR"))
    
    async def _security_scan_plugin(self, file_path: Path) -> Either[PluginError, None]:
        """Perform security scan on plugin file."""
        try:
            # Extract to temporary directory for scanning
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(temp_path)
                
                # Use security manager to scan
                scan_result = self.security_manager.get_security_report(temp_path)
                if scan_result.is_left():
                    return scan_result
                
                scan_data = scan_result.get_right()
                
                # Check security rating
                if scan_data['security_rating'] == 'DANGEROUS':
                    return Either.left(PluginError.security_violation(
                        f"Plugin failed security scan: {', '.join(scan_data['recommendations'])}"
                    ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(PluginError(f"Security scan failed: {str(e)}", "SECURITY_SCAN_ERROR"))
    
    async def _install_plugin_files(self, entry: MarketplaceEntry, source_path: Path, target_dir: Path) -> Either[PluginError, Path]:
        """Install plugin files to target directory."""
        try:
            plugin_dir = target_dir / entry.metadata.identifier
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract plugin files
            with zipfile.ZipFile(source_path, 'r') as zip_file:
                zip_file.extractall(plugin_dir)
            
            return Either.right(plugin_dir)
            
        except Exception as e:
            return Either.left(PluginError(f"File installation failed: {str(e)}", "INSTALL_ERROR"))
    
    async def _record_installation(self, entry: MarketplaceEntry, plugin_path: Path):
        """Record successful plugin installation."""
        installation_record = {
            "plugin_id": entry.metadata.identifier,
            "version": entry.metadata.version,
            "installed_at": datetime.now().isoformat(),
            "install_path": str(plugin_path),
            "marketplace_entry": entry.__dict__
        }
        
        # Save installation record
        records_file = self.cache_dir / "installations.json"
        records = {}
        
        if records_file.exists():
            with open(records_file, 'r') as f:
                records = json.load(f)
        
        records[entry.metadata.identifier] = installation_record
        
        with open(records_file, 'w') as f:
            json.dump(records, f, indent=2, default=str)
    
    async def _load_cache(self):
        """Load cached marketplace data."""
        cache_file = self.cache_dir / "marketplace_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Load cached entries (simplified - would need proper deserialization)
                logger.debug(f"Loaded {len(cache_data)} cached entries")
                
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
    
    async def _cache_search_results(self, query: SearchQuery, results: List[MarketplaceEntry]):
        """Cache search results for faster subsequent searches."""
        # Implementation would cache results with TTL
        pass
    
    def _is_cache_valid(self, entry: MarketplaceEntry) -> bool:
        """Check if cached entry is still valid."""
        ttl_hours = self.config["cache_ttl_hours"]
        expiry = entry.metadata.created_at + timedelta(hours=ttl_hours)
        return datetime.now() < expiry
    
    def _is_newer_version(self, available_version: str, current_version: str) -> bool:
        """Compare version strings to determine if update is available."""
        # Simplified version comparison - would use proper semver in production
        try:
            available_parts = [int(x) for x in available_version.split('.')]
            current_parts = [int(x) for x in current_version.split('.')]
            
            # Pad to same length
            max_length = max(len(available_parts), len(current_parts))
            available_parts.extend([0] * (max_length - len(available_parts)))
            current_parts.extend([0] * (max_length - len(current_parts)))
            
            return available_parts > current_parts
            
        except ValueError:
            return False