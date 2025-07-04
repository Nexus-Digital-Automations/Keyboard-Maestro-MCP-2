"""
Named Clipboard System for Persistent Storage

This module implements named clipboards for workflow data persistence,
providing organizational capabilities with security validation and
efficient management of clipboard collections.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import re
import time
import json
import asyncio
from pathlib import Path

from .clipboard_manager import ClipboardContent, ClipboardFormat
from ..core.contracts import require, ensure
from ..integration.km_client import Either, KMError


@dataclass(frozen=True)
class NamedClipboard:
    """Named clipboard for persistent storage with metadata tracking."""
    name: str
    content: ClipboardContent
    created_at: float
    accessed_at: float
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)
    description: Optional[str] = None
    
    @require(lambda self: len(self.name) > 0 and len(self.name) <= 100)
    @require(lambda self: re.match(r'^[a-zA-Z0-9_\-\s]+$', self.name))
    def __post_init__(self):
        """Validate named clipboard constraints."""
        pass
    
    def with_access(self) -> NamedClipboard:
        """Create new instance with updated access information."""
        return NamedClipboard(
            name=self.name,
            content=self.content,
            created_at=self.created_at,
            accessed_at=time.time(),
            access_count=self.access_count + 1,
            tags=self.tags,
            description=self.description
        )
    
    def with_tags(self, tags: Set[str]) -> NamedClipboard:
        """Create new instance with updated tags."""
        return NamedClipboard(
            name=self.name,
            content=self.content,
            created_at=self.created_at,
            accessed_at=self.accessed_at,
            access_count=self.access_count,
            tags=tags,
            description=self.description
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "content": {
                "data": self.content.content,
                "format": self.content.format.value,
                "size_bytes": self.content.size_bytes,
                "timestamp": self.content.timestamp,
                "is_sensitive": self.content.is_sensitive,
                "preview_safe": self.content.preview_safe
            },
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "tags": list(self.tags),
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NamedClipboard:
        """Create from dictionary data."""
        content_data = data["content"]
        content = ClipboardContent(
            content=content_data["data"],
            format=ClipboardFormat(content_data["format"]),
            size_bytes=content_data["size_bytes"],
            timestamp=content_data["timestamp"],
            is_sensitive=content_data.get("is_sensitive", False),
            preview_safe=content_data.get("preview_safe", True)
        )
        
        return cls(
            name=data["name"],
            content=content,
            created_at=data["created_at"],
            accessed_at=data["accessed_at"],
            access_count=data.get("access_count", 0),
            tags=set(data.get("tags", [])),
            description=data.get("description")
        )


class NamedClipboardManager:
    """
    Manage named clipboards with persistence and organization.
    
    Provides efficient storage and retrieval of named clipboard content
    with search capabilities, tagging, and security validation.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize named clipboard manager.
        
        Args:
            storage_path: Custom path for persistent storage
        """
        self._clipboards: Dict[str, NamedClipboard] = {}
        self._storage_path = storage_path or Path.home() / ".km_mcp" / "named_clipboards.json"
        self._max_clipboards = 1000
        self._max_name_length = 100
        
        # Ensure storage directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing clipboards
        asyncio.create_task(self._load_clipboards())
    
    @require(lambda name: len(name) > 0 and len(name) <= 100)
    @require(lambda name: re.match(r'^[a-zA-Z0-9_\-\s]+$', name))
    @ensure(lambda result: result.is_right() or result.get_left().code in ["NAME_CONFLICT", "STORAGE_FULL", "VALIDATION_ERROR"])
    async def create_named_clipboard(
        self, 
        name: str, 
        content: ClipboardContent,
        tags: Optional[Set[str]] = None,
        description: Optional[str] = None,
        overwrite: bool = False
    ) -> Either[KMError, bool]:
        """
        Create named clipboard with conflict detection and validation.
        
        Args:
            name: Unique clipboard name
            content: Clipboard content to store
            tags: Optional tags for organization
            description: Optional description
            overwrite: Whether to overwrite existing clipboard
            
        Returns:
            Either success status or error details
        """
        try:
            # Validate name format
            if not re.match(r'^[a-zA-Z0-9_\-\s]+$', name.strip()):
                return Either.left(KMError.validation_error(
                    "Clipboard name can only contain letters, numbers, spaces, underscores, and hyphens"
                ))
            
            name = name.strip()
            
            # Check for existing clipboard
            if name in self._clipboards and not overwrite:
                return Either.left(KMError.validation_error(
                    f"Named clipboard '{name}' already exists. Use overwrite=True to replace."
                ))
            
            # Check storage limits
            if len(self._clipboards) >= self._max_clipboards and name not in self._clipboards:
                return Either.left(KMError.validation_error(
                    f"Maximum named clipboards ({self._max_clipboards}) reached"
                ))
            
            # Validate tags
            validated_tags = set()
            if tags:
                for tag in tags:
                    if isinstance(tag, str) and re.match(r'^[a-zA-Z0-9_\-]+$', tag):
                        validated_tags.add(tag.strip().lower())
            
            # Create named clipboard
            current_time = time.time()
            named_clipboard = NamedClipboard(
                name=name,
                content=content,
                created_at=current_time,
                accessed_at=current_time,
                access_count=0,
                tags=validated_tags,
                description=description
            )
            
            # Store clipboard
            self._clipboards[name] = named_clipboard
            
            # Persist to storage
            await self._save_clipboards()
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to create named clipboard: {str(e)}"))
    
    async def get_named_clipboard(self, name: str) -> Either[KMError, NamedClipboard]:
        """
        Get named clipboard by name with access tracking.
        
        Args:
            name: Clipboard name to retrieve
            
        Returns:
            Either named clipboard or error details
        """
        try:
            if name not in self._clipboards:
                return Either.left(KMError.not_found_error(f"Named clipboard '{name}' not found"))
            
            # Update access information
            clipboard = self._clipboards[name].with_access()
            self._clipboards[name] = clipboard
            
            # Persist access update (async, don't wait)
            asyncio.create_task(self._save_clipboards())
            
            return Either.right(clipboard)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get named clipboard: {str(e)}"))
    
    async def list_named_clipboards(
        self, 
        tag_filter: Optional[str] = None,
        sort_by: str = "name"
    ) -> Either[KMError, List[NamedClipboard]]:
        """
        List all named clipboards with optional filtering and sorting.
        
        Args:
            tag_filter: Optional tag to filter by
            sort_by: Sort field ("name", "created_at", "accessed_at", "access_count")
            
        Returns:
            Either list of named clipboards or error details
        """
        try:
            clipboards = list(self._clipboards.values())
            
            # Apply tag filter
            if tag_filter:
                tag_filter = tag_filter.lower().strip()
                clipboards = [cb for cb in clipboards if tag_filter in cb.tags]
            
            # Apply sorting
            if sort_by == "name":
                clipboards.sort(key=lambda cb: cb.name.lower())
            elif sort_by == "created_at":
                clipboards.sort(key=lambda cb: cb.created_at, reverse=True)
            elif sort_by == "accessed_at":
                clipboards.sort(key=lambda cb: cb.accessed_at, reverse=True)
            elif sort_by == "access_count":
                clipboards.sort(key=lambda cb: cb.access_count, reverse=True)
            
            return Either.right(clipboards)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to list named clipboards: {str(e)}"))
    
    async def delete_named_clipboard(self, name: str) -> Either[KMError, bool]:
        """
        Delete named clipboard with validation.
        
        Args:
            name: Clipboard name to delete
            
        Returns:
            Either success status or error details
        """
        try:
            if name not in self._clipboards:
                return Either.left(KMError.not_found_error(f"Named clipboard '{name}' not found"))
            
            # Remove from memory
            del self._clipboards[name]
            
            # Persist changes
            await self._save_clipboards()
            
            return Either.right(True)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to delete named clipboard: {str(e)}"))
    
    async def search_named_clipboards(
        self, 
        query: str,
        search_content: bool = False,
        max_results: int = 50
    ) -> Either[KMError, List[NamedClipboard]]:
        """
        Search named clipboards by name, tags, or content.
        
        Args:
            query: Search query string
            search_content: Whether to search clipboard content
            max_results: Maximum number of results
            
        Returns:
            Either list of matching clipboards or error details
        """
        try:
            query = query.lower().strip()
            if not query:
                return Either.right([])
            
            matching_clipboards = []
            
            for clipboard in self._clipboards.values():
                # Check name match
                if query in clipboard.name.lower():
                    matching_clipboards.append(clipboard)
                    continue
                
                # Check tags match
                if any(query in tag for tag in clipboard.tags):
                    matching_clipboards.append(clipboard)
                    continue
                
                # Check description match
                if clipboard.description and query in clipboard.description.lower():
                    matching_clipboards.append(clipboard)
                    continue
                
                # Check content match (if enabled and content is text)
                if (search_content and 
                    clipboard.content.format == ClipboardFormat.TEXT and
                    not clipboard.content.is_sensitive and
                    isinstance(clipboard.content.content, str)):
                    if query in clipboard.content.content.lower():
                        matching_clipboards.append(clipboard)
            
            # Limit results and sort by relevance (access count as proxy)
            matching_clipboards.sort(key=lambda cb: cb.access_count, reverse=True)
            matching_clipboards = matching_clipboards[:max_results]
            
            return Either.right(matching_clipboards)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to search named clipboards: {str(e)}"))
    
    async def get_clipboard_stats(self) -> Either[KMError, Dict[str, Any]]:
        """
        Get statistics about named clipboards.
        
        Returns:
            Either statistics dictionary or error details
        """
        try:
            stats = {
                "total_clipboards": len(self._clipboards),
                "total_size_bytes": sum(cb.content.size_bytes for cb in self._clipboards.values()),
                "format_distribution": {},
                "most_accessed": None,
                "most_recent": None,
                "oldest": None,
                "average_access_count": 0,
                "tags": set()
            }
            
            if not self._clipboards:
                return Either.right(stats)
            
            # Calculate format distribution
            format_counts = {}
            for clipboard in self._clipboards.values():
                format_name = clipboard.content.format.value
                format_counts[format_name] = format_counts.get(format_name, 0) + 1
            stats["format_distribution"] = format_counts
            
            # Find most accessed
            most_accessed = max(self._clipboards.values(), key=lambda cb: cb.access_count)
            stats["most_accessed"] = {
                "name": most_accessed.name,
                "access_count": most_accessed.access_count
            }
            
            # Find most recent and oldest
            most_recent = max(self._clipboards.values(), key=lambda cb: cb.created_at)
            oldest = min(self._clipboards.values(), key=lambda cb: cb.created_at)
            stats["most_recent"] = {"name": most_recent.name, "created_at": most_recent.created_at}
            stats["oldest"] = {"name": oldest.name, "created_at": oldest.created_at}
            
            # Calculate average access count
            total_accesses = sum(cb.access_count for cb in self._clipboards.values())
            stats["average_access_count"] = total_accesses / len(self._clipboards)
            
            # Collect all tags
            all_tags = set()
            for clipboard in self._clipboards.values():
                all_tags.update(clipboard.tags)
            stats["tags"] = sorted(all_tags)
            
            return Either.right(stats)
            
        except Exception as e:
            return Either.left(KMError.execution_error(f"Failed to get clipboard stats: {str(e)}"))
    
    async def _load_clipboards(self) -> None:
        """Load named clipboards from persistent storage."""
        try:
            if not self._storage_path.exists():
                return
            
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for clipboard_data in data.get('clipboards', []):
                try:
                    clipboard = NamedClipboard.from_dict(clipboard_data)
                    self._clipboards[clipboard.name] = clipboard
                except Exception:
                    # Skip invalid clipboard data
                    continue
                    
        except Exception:
            # If loading fails, start with empty clipboards
            self._clipboards = {}
    
    async def _save_clipboards(self) -> None:
        """Save named clipboards to persistent storage."""
        try:
            data = {
                'version': '1.0',
                'clipboards': [cb.to_dict() for cb in self._clipboards.values()]
            }
            
            # Write to temporary file first for atomic update
            temp_path = self._storage_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(self._storage_path)
            
        except Exception:
            # Silent failure for storage operations
            pass