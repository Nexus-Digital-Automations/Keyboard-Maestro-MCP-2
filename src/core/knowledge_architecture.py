"""
Knowledge management architecture for documentation automation and knowledge base management.

This module provides the foundational types and contracts for automated documentation
generation, intelligent content organization, and comprehensive knowledge management.
"""

from __future__ import annotations
from typing import NewType, Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import uuid
import hashlib
import json


# Branded Types for Knowledge Management
DocumentId = NewType('DocumentId', str)
KnowledgeBaseId = NewType('KnowledgeBaseId', str)
ContentId = NewType('ContentId', str)
TemplateId = NewType('TemplateId', str)
SearchQueryId = NewType('SearchQueryId', str)
VersionId = NewType('VersionId', str)


class DocumentType(Enum):
    """Types of documentation that can be generated."""
    OVERVIEW = "overview"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    USER_GUIDE = "user_guide"
    API_REFERENCE = "api_reference"
    TROUBLESHOOTING = "troubleshooting"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


class ContentFormat(Enum):
    """Content output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    CONFLUENCE = "confluence"
    DOCX = "docx"
    JSON = "json"
    PLAIN_TEXT = "plain_text"


class SearchType(Enum):
    """Types of knowledge search."""
    TEXT = "text"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    EXACT = "exact"
    BOOLEAN = "boolean"


class KnowledgeCategory(Enum):
    """Knowledge content categories."""
    AUTOMATION = "automation"
    DOCUMENTATION = "documentation"
    PROCEDURES = "procedures"
    TEMPLATES = "templates"
    EXAMPLES = "examples"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"
    REFERENCE = "reference"


class QualityMetric(Enum):
    """Content quality assessment metrics."""
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    FRESHNESS = "freshness"
    ACCESSIBILITY = "accessibility"


@dataclass(frozen=True)
class ContentMetadata:
    """Metadata for knowledge content with comprehensive tracking."""
    content_id: ContentId
    title: str
    description: str
    category: KnowledgeCategory
    tags: Set[str] = field(default_factory=set)
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    language: str = "en"
    word_count: int = 0
    reading_time_minutes: int = 0
    difficulty_level: str = "intermediate"
    
    def __post_init__(self):
        if not self.title.strip():
            raise ValueError("Content title cannot be empty")
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")
        if self.reading_time_minutes < 0:
            raise ValueError("Reading time cannot be negative")


@dataclass(frozen=True)
class DocumentationSource:
    """Source information for documentation generation."""
    source_type: str  # macro|workflow|group|system
    source_id: str
    source_name: str
    source_data: Dict[str, Any] = field(default_factory=dict)
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.source_id.strip():
            raise ValueError("Source ID cannot be empty")
        if self.source_type not in ["macro", "workflow", "group", "system"]:
            raise ValueError(f"Invalid source type: {self.source_type}")


@dataclass(frozen=True)
class KnowledgeDocument:
    """Complete knowledge document with content and metadata."""
    document_id: DocumentId
    metadata: ContentMetadata
    content: str
    source: Optional[DocumentationSource] = None
    related_documents: Set[DocumentId] = field(default_factory=set)
    quality_score: float = 0.0
    checksum: str = ""
    
    def __post_init__(self):
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        if not (0.0 <= self.quality_score <= 100.0):
            raise ValueError("Quality score must be between 0.0 and 100.0")
        if not self.checksum:
            # Calculate content checksum for integrity
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()
            object.__setattr__(self, 'checksum', content_hash)
    
    def calculate_quality_score(self, metrics: Dict[QualityMetric, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not metrics:
            return 0.0
        
        # Weight different quality metrics
        weights = {
            QualityMetric.CLARITY: 0.25,
            QualityMetric.COMPLETENESS: 0.20,
            QualityMetric.ACCURACY: 0.20,
            QualityMetric.CONSISTENCY: 0.15,
            QualityMetric.RELEVANCE: 0.10,
            QualityMetric.FRESHNESS: 0.05,
            QualityMetric.ACCESSIBILITY: 0.05
        }
        
        weighted_score = sum(
            metrics.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    def verify_integrity(self) -> bool:
        """Verify document content integrity using checksum."""
        current_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return current_hash == self.checksum


@dataclass(frozen=True)
class KnowledgeBase:
    """Knowledge base container with organization and metadata."""
    knowledge_base_id: KnowledgeBaseId
    name: str
    description: str
    categories: Set[KnowledgeCategory] = field(default_factory=set)
    documents: Set[DocumentId] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    auto_categorize: bool = True
    enable_search: bool = True
    access_permissions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Knowledge base name cannot be empty")
        if len(self.name) > 100:
            raise ValueError("Knowledge base name too long (max 100 characters)")


# Helper Functions
def create_document_id() -> DocumentId:
    """Create a new unique document ID."""
    return DocumentId(f"doc_{uuid.uuid4().hex[:12]}")


def create_knowledge_base_id() -> KnowledgeBaseId:
    """Create a new unique knowledge base ID."""
    return KnowledgeBaseId(f"kb_{uuid.uuid4().hex[:12]}")


def create_content_id() -> ContentId:
    """Create a new unique content ID."""
    return ContentId(f"content_{uuid.uuid4().hex[:12]}")


# Knowledge Management Error Types
class KnowledgeError(Exception):
    """Base class for knowledge management errors."""
    pass


class DocumentGenerationError(KnowledgeError):
    """Error during document generation."""
    pass


class SearchError(KnowledgeError):
    """Error during knowledge search."""
    pass