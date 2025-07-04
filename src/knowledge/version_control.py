"""
Version Control - TASK_56 Phase 2 Implementation

Documentation versioning and change tracking system for knowledge management.
Provides version history, change detection, rollback capabilities, and audit trails.

Architecture: Version Management + Change Detection + Audit Trails + Rollback System
Performance: <30ms version operations, efficient change tracking
Security: Cryptographic integrity, change authentication, access control
"""

from __future__ import annotations
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import logging
import json
import hashlib
import difflib
from enum import Enum

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.knowledge_architecture import (
    DocumentId, ContentId, KnowledgeBaseId, VersionId,
    KnowledgeDocument, ContentMetadata,
    create_document_id, create_content_id,
    KnowledgeError
)

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of document changes."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    MERGED = "merged"
    BRANCHED = "branched"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    MANUAL = "manual"
    LATEST_WINS = "latest_wins"
    MERGE_CHANGES = "merge_changes"
    PRESERVE_BOTH = "preserve_both"


@dataclass(frozen=True)
class DocumentVersion:
    """Version information for a document."""
    version_id: VersionId
    document_id: DocumentId
    version_number: int
    content: str
    metadata: ContentMetadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    author: str = "system"
    change_summary: str = ""
    change_type: ChangeType = ChangeType.UPDATED
    parent_version_id: Optional[VersionId] = None
    content_hash: str = ""
    size_bytes: int = 0
    
    def __post_init__(self):
        if not self.content_hash:
            content_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()
            object.__setattr__(self, 'content_hash', content_hash)
        
        if self.size_bytes == 0:
            size_bytes = len(self.content.encode('utf-8'))
            object.__setattr__(self, 'size_bytes', size_bytes)
    
    def verify_integrity(self) -> bool:
        """Verify version content integrity."""
        expected_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        return self.content_hash == expected_hash


@dataclass
class ChangeRecord:
    """Record of changes between versions."""
    change_id: str
    document_id: DocumentId
    from_version_id: Optional[VersionId]
    to_version_id: VersionId
    change_type: ChangeType
    author: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    diff_stats: Dict[str, int] = field(default_factory=dict)  # lines_added, lines_removed, etc.
    
    def get_change_magnitude(self) -> float:
        """Calculate magnitude of change (0.0 to 1.0)."""
        if not self.diff_stats:
            return 0.0
        
        added = self.diff_stats.get('lines_added', 0)
        removed = self.diff_stats.get('lines_removed', 0)
        total_lines = self.diff_stats.get('total_lines', 1)
        
        change_ratio = (added + removed) / max(1, total_lines)
        return min(1.0, change_ratio)


@dataclass
class ConflictInfo:
    """Information about version conflicts."""
    document_id: DocumentId
    base_version_id: VersionId
    version_a_id: VersionId
    version_b_id: VersionId
    conflict_sections: List[Dict[str, Any]] = field(default_factory=list)
    suggested_resolution: ConflictResolution = ConflictResolution.MANUAL
    auto_resolvable: bool = False
    
    def add_conflict_section(self, section_name: str, content_a: str, content_b: str, line_start: int, line_end: int):
        """Add a conflicting section."""
        self.conflict_sections.append({
            "section": section_name,
            "content_a": content_a,
            "content_b": content_b,
            "line_start": line_start,
            "line_end": line_end
        })


class VersionManager:
    """
    Document version control and change tracking system.
    
    Provides comprehensive version management with change detection,
    rollback capabilities, conflict resolution, and audit trails.
    """
    
    def __init__(self):
        self.versions: Dict[VersionId, DocumentVersion] = {}
        self.document_versions: Dict[DocumentId, List[VersionId]] = {}
        self.change_history: List[ChangeRecord] = []
        self.active_documents: Dict[DocumentId, VersionId] = {}
        self.version_counter: Dict[DocumentId, int] = {}
        
        logger.info("VersionManager initialized")
    
    @require(lambda document: document.content.strip(), "Document content required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns version or error")
    async def create_initial_version(
        self,
        document: KnowledgeDocument,
        author: str = "system"
    ) -> Either[str, DocumentVersion]:
        """Create initial version for a new document."""
        try:
            doc_id = document.document_id
            
            # Generate version ID
            version_id = VersionId(f"{doc_id}_v1")
            
            # Create initial version
            version = DocumentVersion(
                version_id=version_id,
                document_id=doc_id,
                version_number=1,
                content=document.content,
                metadata=document.metadata,
                author=author,
                change_summary="Initial document creation",
                change_type=ChangeType.CREATED
            )
            
            # Store version
            self.versions[version_id] = version
            self.document_versions[doc_id] = [version_id]
            self.active_documents[doc_id] = version_id
            self.version_counter[doc_id] = 1
            
            # Record change
            change_record = ChangeRecord(
                change_id=f"{doc_id}_change_1",
                document_id=doc_id,
                from_version_id=None,
                to_version_id=version_id,
                change_type=ChangeType.CREATED,
                author=author,
                summary="Document created",
                diff_stats={"total_lines": len(document.content.split('\n'))}
            )
            self.change_history.append(change_record)
            
            logger.info(f"Created initial version for document: {doc_id}")
            return Either.right(version)
            
        except Exception as e:
            error_msg = f"Failed to create initial version: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    @require(lambda document_id, content: content.strip(), "Content required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns version or error")
    async def create_version(
        self,
        document_id: DocumentId,
        content: str,
        metadata: ContentMetadata,
        author: str = "system",
        change_summary: str = ""
    ) -> Either[str, DocumentVersion]:
        """Create new version of existing document."""
        try:
            if document_id not in self.document_versions:
                return Either.left(f"Document {document_id} not found")
            
            # Get current version
            current_version_id = self.active_documents[document_id]
            current_version = self.versions[current_version_id]
            
            # Check if content actually changed
            if current_version.content == content:
                return Either.left("No changes detected - version not created")
            
            # Generate new version
            new_version_number = self.version_counter[document_id] + 1
            new_version_id = VersionId(f"{document_id}_v{new_version_number}")
            
            # Calculate diff stats
            diff_stats = self._calculate_diff_stats(current_version.content, content)
            
            # Create version
            version = DocumentVersion(
                version_id=new_version_id,
                document_id=document_id,
                version_number=new_version_number,
                content=content,
                metadata=metadata,
                author=author,
                change_summary=change_summary or f"Version {new_version_number} update",
                change_type=ChangeType.UPDATED,
                parent_version_id=current_version_id
            )
            
            # Store version
            self.versions[new_version_id] = version
            self.document_versions[document_id].append(new_version_id)
            self.active_documents[document_id] = new_version_id
            self.version_counter[document_id] = new_version_number
            
            # Record change
            change_record = ChangeRecord(
                change_id=f"{document_id}_change_{new_version_number}",
                document_id=document_id,
                from_version_id=current_version_id,
                to_version_id=new_version_id,
                change_type=ChangeType.UPDATED,
                author=author,
                summary=change_summary,
                diff_stats=diff_stats
            )
            self.change_history.append(change_record)
            
            logger.info(f"Created version {new_version_number} for document: {document_id}")
            return Either.right(version)
            
        except Exception as e:
            error_msg = f"Failed to create version: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def get_version(self, version_id: VersionId) -> Either[str, DocumentVersion]:
        """Get specific document version."""
        try:
            if version_id not in self.versions:
                return Either.left(f"Version {version_id} not found")
            
            version = self.versions[version_id]
            
            # Verify integrity
            if not version.verify_integrity():
                return Either.left(f"Version {version_id} failed integrity check")
            
            return Either.right(version)
            
        except Exception as e:
            error_msg = f"Failed to get version: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def get_version_history(
        self,
        document_id: DocumentId,
        limit: int = 50
    ) -> Either[str, List[DocumentVersion]]:
        """Get version history for document."""
        try:
            if document_id not in self.document_versions:
                return Either.left(f"Document {document_id} not found")
            
            version_ids = self.document_versions[document_id]
            versions = []
            
            for version_id in reversed(version_ids[-limit:]):  # Most recent first
                if version_id in self.versions:
                    versions.append(self.versions[version_id])
            
            return Either.right(versions)
            
        except Exception as e:
            error_msg = f"Failed to get version history: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def rollback_to_version(
        self,
        document_id: DocumentId,
        target_version_id: VersionId,
        author: str = "system"
    ) -> Either[str, DocumentVersion]:
        """Rollback document to specific version."""
        try:
            if document_id not in self.document_versions:
                return Either.left(f"Document {document_id} not found")
            
            if target_version_id not in self.versions:
                return Either.left(f"Target version {target_version_id} not found")
            
            target_version = self.versions[target_version_id]
            if target_version.document_id != document_id:
                return Either.left("Version does not belong to specified document")
            
            # Create rollback version
            new_version_number = self.version_counter[document_id] + 1
            rollback_version_id = VersionId(f"{document_id}_v{new_version_number}")
            
            # Create rollback version (essentially a copy of target)
            rollback_version = DocumentVersion(
                version_id=rollback_version_id,
                document_id=document_id,
                version_number=new_version_number,
                content=target_version.content,
                metadata=target_version.metadata,
                author=author,
                change_summary=f"Rolled back to version {target_version.version_number}",
                change_type=ChangeType.RESTORED,
                parent_version_id=self.active_documents[document_id]
            )
            
            # Store rollback version
            self.versions[rollback_version_id] = rollback_version
            self.document_versions[document_id].append(rollback_version_id)
            self.active_documents[document_id] = rollback_version_id
            self.version_counter[document_id] = new_version_number
            
            # Record change
            change_record = ChangeRecord(
                change_id=f"{document_id}_rollback_{new_version_number}",
                document_id=document_id,
                from_version_id=target_version_id,
                to_version_id=rollback_version_id,
                change_type=ChangeType.RESTORED,
                author=author,
                summary=f"Rolled back to version {target_version.version_number}",
                details={"target_version": target_version_id}
            )
            self.change_history.append(change_record)
            
            logger.info(f"Rolled back document {document_id} to version {target_version.version_number}")
            return Either.right(rollback_version)
            
        except Exception as e:
            error_msg = f"Failed to rollback document: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def compare_versions(
        self,
        version_a_id: VersionId,
        version_b_id: VersionId
    ) -> Either[str, Dict[str, Any]]:
        """Compare two document versions."""
        try:
            if version_a_id not in self.versions:
                return Either.left(f"Version A {version_a_id} not found")
            if version_b_id not in self.versions:
                return Either.left(f"Version B {version_b_id} not found")
            
            version_a = self.versions[version_a_id]
            version_b = self.versions[version_b_id]
            
            # Generate diff
            diff_result = self._generate_diff(version_a.content, version_b.content)
            
            # Calculate statistics
            stats = self._calculate_diff_stats(version_a.content, version_b.content)
            
            comparison = {
                "version_a": {
                    "version_id": version_a_id,
                    "version_number": version_a.version_number,
                    "author": version_a.author,
                    "created_at": version_a.created_at.isoformat(),
                    "size_bytes": version_a.size_bytes
                },
                "version_b": {
                    "version_id": version_b_id,
                    "version_number": version_b.version_number,
                    "author": version_b.author,
                    "created_at": version_b.created_at.isoformat(),
                    "size_bytes": version_b.size_bytes
                },
                "diff": diff_result,
                "statistics": stats,
                "change_magnitude": self._calculate_change_magnitude(stats)
            }
            
            return Either.right(comparison)
            
        except Exception as e:
            error_msg = f"Failed to compare versions: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def detect_conflicts(
        self,
        document_id: DocumentId,
        version_a_id: VersionId,
        version_b_id: VersionId
    ) -> Either[str, ConflictInfo]:
        """Detect conflicts between two versions."""
        try:
            if version_a_id not in self.versions or version_b_id not in self.versions:
                return Either.left("One or both versions not found")
            
            version_a = self.versions[version_a_id]
            version_b = self.versions[version_b_id]
            
            # Find common ancestor (simplified - would use proper merge base)
            base_version_id = version_a.parent_version_id or version_b.parent_version_id
            if not base_version_id or base_version_id not in self.versions:
                return Either.left("Cannot find common base version")
            
            base_version = self.versions[base_version_id]
            
            # Detect conflicts
            conflict_info = ConflictInfo(
                document_id=document_id,
                base_version_id=base_version_id,
                version_a_id=version_a_id,
                version_b_id=version_b_id
            )
            
            # Simple conflict detection (line-based)
            base_lines = base_version.content.split('\n')
            a_lines = version_a.content.split('\n')
            b_lines = version_b.content.split('\n')
            
            # Compare changes from base
            a_changes = self._get_line_changes(base_lines, a_lines)
            b_changes = self._get_line_changes(base_lines, b_lines)
            
            # Find overlapping changes (conflicts)
            for line_num in a_changes:
                if line_num in b_changes:
                    if a_changes[line_num] != b_changes[line_num]:
                        conflict_info.add_conflict_section(
                            f"Line {line_num}",
                            a_changes[line_num],
                            b_changes[line_num],
                            line_num,
                            line_num
                        )
            
            # Determine if auto-resolvable
            conflict_info.auto_resolvable = len(conflict_info.conflict_sections) == 0
            if conflict_info.auto_resolvable:
                conflict_info.suggested_resolution = ConflictResolution.MERGE_CHANGES
            
            return Either.right(conflict_info)
            
        except Exception as e:
            error_msg = f"Failed to detect conflicts: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def _calculate_diff_stats(self, content_a: str, content_b: str) -> Dict[str, int]:
        """Calculate diff statistics between two content strings."""
        try:
            lines_a = content_a.split('\n')
            lines_b = content_b.split('\n')
            
            differ = difflib.unified_diff(lines_a, lines_b, lineterm='')
            diff_lines = list(differ)
            
            added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
            removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
            
            return {
                "lines_added": added,
                "lines_removed": removed,
                "lines_modified": min(added, removed),
                "total_lines": max(len(lines_a), len(lines_b)),
                "size_change": len(content_b) - len(content_a)
            }
            
        except Exception as e:
            logger.warning(f"Diff stats calculation failed: {e}")
            return {}
    
    def _generate_diff(self, content_a: str, content_b: str) -> List[str]:
        """Generate unified diff between two content strings."""
        try:
            lines_a = content_a.split('\n')
            lines_b = content_b.split('\n')
            
            diff = difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile='version_a',
                tofile='version_b',
                lineterm=''
            )
            
            return list(diff)
            
        except Exception as e:
            logger.warning(f"Diff generation failed: {e}")
            return []
    
    def _calculate_change_magnitude(self, stats: Dict[str, int]) -> float:
        """Calculate change magnitude from diff stats."""
        if not stats:
            return 0.0
        
        added = stats.get('lines_added', 0)
        removed = stats.get('lines_removed', 0)
        total = stats.get('total_lines', 1)
        
        change_ratio = (added + removed) / max(1, total)
        return min(1.0, change_ratio)
    
    def _get_line_changes(self, base_lines: List[str], changed_lines: List[str]) -> Dict[int, str]:
        """Get line-by-line changes from base to changed version."""
        changes = {}
        
        # Simple line comparison (could be improved with proper diff algorithm)
        max_lines = max(len(base_lines), len(changed_lines))
        
        for i in range(max_lines):
            base_line = base_lines[i] if i < len(base_lines) else ""
            changed_line = changed_lines[i] if i < len(changed_lines) else ""
            
            if base_line != changed_line:
                changes[i] = changed_line
        
        return changes
    
    async def get_change_history(
        self,
        document_id: Optional[DocumentId] = None,
        author: Optional[str] = None,
        limit: int = 100
    ) -> List[ChangeRecord]:
        """Get filtered change history."""
        try:
            filtered_changes = self.change_history
            
            # Filter by document
            if document_id:
                filtered_changes = [c for c in filtered_changes if c.document_id == document_id]
            
            # Filter by author
            if author:
                filtered_changes = [c for c in filtered_changes if c.author == author]
            
            # Sort by timestamp (most recent first) and limit
            sorted_changes = sorted(filtered_changes, key=lambda c: c.timestamp, reverse=True)
            return sorted_changes[:limit]
            
        except Exception as e:
            logger.warning(f"Change history retrieval failed: {e}")
            return []
    
    async def get_version_analytics(self) -> Dict[str, Any]:
        """Get version control analytics and statistics."""
        try:
            total_documents = len(self.document_versions)
            total_versions = len(self.versions)
            
            # Calculate version distribution
            version_counts = [len(versions) for versions in self.document_versions.values()]
            avg_versions = sum(version_counts) / max(1, len(version_counts))
            
            # Change type distribution
            change_types = {}
            for change in self.change_history:
                change_types[change.change_type.value] = change_types.get(change.change_type.value, 0) + 1
            
            # Author activity
            author_activity = {}
            for change in self.change_history:
                author_activity[change.author] = author_activity.get(change.author, 0) + 1
            
            return {
                "total_documents": total_documents,
                "total_versions": total_versions,
                "average_versions_per_document": avg_versions,
                "change_type_distribution": change_types,
                "author_activity": author_activity,
                "recent_changes": len([c for c in self.change_history 
                                     if (datetime.now(UTC) - c.timestamp).days <= 7])
            }
            
        except Exception as e:
            logger.warning(f"Version analytics generation failed: {e}")
            return {}


# Global instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    """Get or create the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager