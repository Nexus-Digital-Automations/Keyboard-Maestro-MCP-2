"""
Content Organizer - TASK_56 Phase 2 Implementation

Intelligent content organization and categorization system for knowledge management.
Provides automated tagging, category detection, and content relationship mapping.

Architecture: Content Analysis + Intelligent Categorization + Relationship Detection
Performance: <100ms content analysis, efficient categorization algorithms
Security: Content validation, safe categorization rules
"""

from __future__ import annotations
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging
import re
import json
from collections import defaultdict

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.knowledge_architecture import (
    DocumentId, ContentId, KnowledgeBaseId,
    KnowledgeCategory, QualityMetric,
    KnowledgeDocument, ContentMetadata,
    create_content_id,
    KnowledgeError
)

logger = logging.getLogger(__name__)


@dataclass
class OrganizationConfig:
    """Configuration for content organization."""
    auto_categorize: bool = True
    auto_tag: bool = True
    detect_relationships: bool = True
    similarity_threshold: float = 0.7
    max_tags_per_document: int = 10
    enable_keyword_extraction: bool = True
    enable_topic_modeling: bool = False
    language: str = "en"


@dataclass
class ContentAnalysis:
    """Analysis results for content organization."""
    content_id: ContentId
    suggested_category: KnowledgeCategory
    confidence_score: float
    extracted_keywords: Set[str] = field(default_factory=set)
    suggested_tags: Set[str] = field(default_factory=set)
    topics: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    quality_metrics: Dict[QualityMetric, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not (0.0 <= self.complexity_score <= 1.0):
            raise ValueError("Complexity score must be between 0.0 and 1.0")


@dataclass
class ContentRelationship:
    """Relationship between content items."""
    source_id: ContentId
    target_id: ContentId
    relationship_type: str  # similar|references|depends_on|supersedes
    strength: float  # 0.0 to 1.0
    explanation: str = ""
    
    def __post_init__(self):
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError("Relationship strength must be between 0.0 and 1.0")


class ContentOrganizer:
    """
    Intelligent content organization and categorization system.
    
    Provides automated content analysis, categorization, tagging, and
    relationship detection for comprehensive knowledge organization.
    """
    
    def __init__(self, config: OrganizationConfig = None):
        self.config = config or OrganizationConfig()
        self.category_rules: Dict[KnowledgeCategory, List[str]] = {}
        self.content_cache: Dict[ContentId, ContentAnalysis] = {}
        self.relationships: List[ContentRelationship] = []
        self._initialize_category_rules()
        
        logger.info("ContentOrganizer initialized")
    
    def _initialize_category_rules(self) -> None:
        """Initialize categorization rules for different content types."""
        self.category_rules = {
            KnowledgeCategory.AUTOMATION: [
                "macro", "automation", "trigger", "action", "workflow", "script",
                "keyboard", "mouse", "application", "execute", "run", "launch"
            ],
            KnowledgeCategory.DOCUMENTATION: [
                "documentation", "guide", "manual", "reference", "help", "readme",
                "instruction", "tutorial", "overview", "specification"
            ],
            KnowledgeCategory.PROCEDURES: [
                "procedure", "process", "step", "workflow", "method", "protocol",
                "checklist", "routine", "standard", "operation"
            ],
            KnowledgeCategory.TEMPLATES: [
                "template", "pattern", "format", "structure", "layout", "skeleton",
                "boilerplate", "framework", "blueprint", "model"
            ],
            KnowledgeCategory.EXAMPLES: [
                "example", "sample", "demo", "illustration", "case", "instance",
                "specimen", "prototype", "showcase", "demonstration"
            ],
            KnowledgeCategory.BEST_PRACTICES: [
                "best", "practice", "recommendation", "guideline", "standard",
                "convention", "principle", "rule", "tip", "advice"
            ],
            KnowledgeCategory.TROUBLESHOOTING: [
                "troubleshoot", "debug", "error", "problem", "issue", "fix",
                "solution", "resolve", "diagnose", "repair"
            ],
            KnowledgeCategory.REFERENCE: [
                "reference", "lookup", "dictionary", "glossary", "index",
                "catalog", "directory", "registry", "database", "table"
            ]
        }
    
    @require(lambda document: document.content.strip(), "Document content required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns analysis or error")
    async def analyze_content(
        self,
        document: KnowledgeDocument
    ) -> Either[str, ContentAnalysis]:
        """Analyze content for intelligent organization."""
        try:
            content_text = document.content.lower()
            
            # Extract keywords
            keywords = self._extract_keywords(content_text) if self.config.enable_keyword_extraction else set()
            
            # Categorize content
            suggested_category, confidence = self._categorize_content(content_text, keywords)
            
            # Generate tags
            suggested_tags = self._generate_tags(content_text, keywords)
            
            # Detect topics
            topics = self._detect_topics(content_text) if self.config.enable_topic_modeling else []
            
            # Calculate complexity
            complexity_score = self._calculate_complexity(document)
            
            # Assess quality
            quality_metrics = self._assess_quality(document)
            
            analysis = ContentAnalysis(
                content_id=document.metadata.content_id,
                suggested_category=suggested_category,
                confidence_score=confidence,
                extracted_keywords=keywords,
                suggested_tags=suggested_tags,
                topics=topics,
                complexity_score=complexity_score,
                quality_metrics=quality_metrics
            )
            
            # Cache analysis
            self.content_cache[document.metadata.content_id] = analysis
            
            logger.info(f"Analyzed content: {document.metadata.title}")
            return Either.right(analysis)
            
        except Exception as e:
            error_msg = f"Content analysis failed: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def organize_documents(
        self,
        documents: List[KnowledgeDocument]
    ) -> Either[str, Dict[str, Any]]:
        """Organize multiple documents with relationship detection."""
        try:
            organization_results = {
                "analyses": [],
                "relationships": [],
                "categories": defaultdict(list),
                "tags": defaultdict(list),
                "quality_summary": {}
            }
            
            # Analyze each document
            for document in documents:
                analysis_result = await self.analyze_content(document)
                if analysis_result.is_right():
                    analysis = analysis_result.right()
                    organization_results["analyses"].append(analysis)
                    
                    # Group by category
                    organization_results["categories"][analysis.suggested_category.value].append({
                        "document_id": document.document_id,
                        "title": document.metadata.title,
                        "confidence": analysis.confidence_score
                    })
                    
                    # Group by tags
                    for tag in analysis.suggested_tags:
                        organization_results["tags"][tag].append({
                            "document_id": document.document_id,
                            "title": document.metadata.title
                        })
            
            # Detect relationships if enabled
            if self.config.detect_relationships and len(documents) > 1:
                relationships = await self._detect_relationships(documents)
                organization_results["relationships"] = relationships
                self.relationships.extend(relationships)
            
            # Generate quality summary
            organization_results["quality_summary"] = self._generate_quality_summary(
                organization_results["analyses"]
            )
            
            logger.info(f"Organized {len(documents)} documents")
            return Either.right(organization_results)
            
        except Exception as e:
            error_msg = f"Document organization failed: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract relevant keywords from content."""
        try:
            # Remove common stop words
            stop_words = {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", 
                "with", "by", "from", "up", "about", "into", "through", "during",
                "before", "after", "above", "below", "between", "among", "this",
                "that", "these", "those", "i", "you", "he", "she", "it", "we",
                "they", "am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "could", "should", "may", "might", "must", "can", "shall"
            }
            
            # Extract words (simple tokenization)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Filter and count
            word_freq = defaultdict(int)
            for word in words:
                if word not in stop_words and len(word) > 2:
                    word_freq[word] += 1
            
            # Get most frequent words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = {word for word, freq in sorted_words[:20] if freq > 1}
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return set()
    
    def _categorize_content(self, content: str, keywords: Set[str]) -> Tuple[KnowledgeCategory, float]:
        """Categorize content based on keywords and rules."""
        try:
            category_scores = {}
            
            for category, rule_keywords in self.category_rules.items():
                score = 0.0
                
                # Check direct keyword matches
                for rule_keyword in rule_keywords:
                    if rule_keyword in content:
                        score += 1.0
                    
                    # Check if any extracted keywords match
                    if keywords and rule_keyword in keywords:
                        score += 0.5
                
                # Normalize score
                if rule_keywords:
                    category_scores[category] = score / len(rule_keywords)
                else:
                    category_scores[category] = 0.0
            
            # Find best category
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                return best_category[0], min(1.0, best_category[1])
            else:
                return KnowledgeCategory.DOCUMENTATION, 0.1
                
        except Exception as e:
            logger.warning(f"Content categorization failed: {e}")
            return KnowledgeCategory.DOCUMENTATION, 0.1
    
    def _generate_tags(self, content: str, keywords: Set[str]) -> Set[str]:
        """Generate relevant tags for content."""
        try:
            tags = set()
            
            # Add top keywords as tags
            if keywords:
                # Take top keywords based on content relevance
                content_words = set(content.split())
                relevant_keywords = {kw for kw in keywords if kw in content_words}
                tags.update(list(relevant_keywords)[:self.config.max_tags_per_document // 2])
            
            # Add domain-specific tags
            domain_tags = {
                "automation": ["keyboard", "mouse", "macro", "trigger", "action"],
                "development": ["code", "script", "function", "variable", "syntax"],
                "system": ["file", "folder", "application", "window", "process"],
                "documentation": ["guide", "tutorial", "reference", "help", "manual"]
            }
            
            for domain, domain_keywords in domain_tags.items():
                matches = sum(1 for kw in domain_keywords if kw in content)
                if matches >= 2:  # Threshold for domain relevance
                    tags.add(domain)
            
            # Limit total tags
            return set(list(tags)[:self.config.max_tags_per_document])
            
        except Exception as e:
            logger.warning(f"Tag generation failed: {e}")
            return set()
    
    def _detect_topics(self, content: str) -> List[str]:
        """Detect main topics in content (simplified implementation)."""
        try:
            # Simplified topic detection based on common patterns
            topics = []
            
            topic_patterns = {
                "automation_workflow": r"automation|workflow|macro|trigger",
                "user_interface": r"interface|ui|button|menu|window",
                "file_management": r"file|folder|directory|path|save",
                "text_processing": r"text|string|format|replace|search",
                "system_control": r"system|process|service|command|execute"
            }
            
            for topic, pattern in topic_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    topics.append(topic)
            
            return topics[:5]  # Limit to top 5 topics
            
        except Exception as e:
            logger.warning(f"Topic detection failed: {e}")
            return []
    
    def _calculate_complexity(self, document: KnowledgeDocument) -> float:
        """Calculate content complexity score."""
        try:
            content = document.content
            
            # Factors contributing to complexity
            factors = {
                "length": min(1.0, len(content) / 5000),  # Longer = more complex
                "technical_terms": len(re.findall(r'\b[A-Z]{2,}\b', content)) / 100,
                "code_blocks": content.count('```') / 10,
                "nested_lists": content.count('  -') / 20,
                "external_references": content.count('http') / 10
            }
            
            # Weighted average
            weights = {
                "length": 0.2,
                "technical_terms": 0.3,
                "code_blocks": 0.2,
                "nested_lists": 0.15,
                "external_references": 0.15
            }
            
            complexity = sum(factors[factor] * weights[factor] for factor in factors)
            return min(1.0, complexity)
            
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return 0.5
    
    def _assess_quality(self, document: KnowledgeDocument) -> Dict[QualityMetric, float]:
        """Assess content quality across multiple metrics."""
        try:
            content = document.content
            metadata = document.metadata
            
            metrics = {}
            
            # Clarity - based on readability
            avg_sentence_length = len(content.split()) / max(1, content.count('.'))
            clarity_score = max(0.0, 1.0 - (avg_sentence_length - 15) / 20)
            metrics[QualityMetric.CLARITY] = max(0.0, min(1.0, clarity_score)) * 100
            
            # Completeness - based on content structure
            has_title = bool(metadata.title.strip())
            has_description = bool(metadata.description.strip())
            has_sections = content.count('#') > 0
            completeness = (has_title + has_description + has_sections) / 3
            metrics[QualityMetric.COMPLETENESS] = completeness * 100
            
            # Consistency - based on formatting
            consistent_headers = len(re.findall(r'^#+\s', content, re.MULTILINE)) > 0
            consistent_lists = content.count('-') > 0 or content.count('*') > 0
            consistency = (consistent_headers + consistent_lists) / 2
            metrics[QualityMetric.CONSISTENCY] = consistency * 100
            
            # Freshness - based on modification date
            days_old = (datetime.now(UTC) - metadata.modified_at).days
            freshness = max(0.0, 1.0 - days_old / 365)  # Decay over a year
            metrics[QualityMetric.FRESHNESS] = freshness * 100
            
            # Accessibility - based on structure
            has_headers = content.count('#') > 0
            reasonable_length = 100 <= len(content) <= 10000
            accessibility = (has_headers + reasonable_length) / 2
            metrics[QualityMetric.ACCESSIBILITY] = accessibility * 100
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {metric: 50.0 for metric in QualityMetric}
    
    async def _detect_relationships(
        self,
        documents: List[KnowledgeDocument]
    ) -> List[ContentRelationship]:
        """Detect relationships between documents."""
        try:
            relationships = []
            
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i+1:], i+1):
                    similarity = self._calculate_similarity(doc1, doc2)
                    
                    if similarity > self.config.similarity_threshold:
                        relationship = ContentRelationship(
                            source_id=doc1.metadata.content_id,
                            target_id=doc2.metadata.content_id,
                            relationship_type="similar",
                            strength=similarity,
                            explanation=f"Content similarity: {similarity:.2f}"
                        )
                        relationships.append(relationship)
                    
                    # Check for references
                    if doc2.metadata.title.lower() in doc1.content.lower():
                        relationship = ContentRelationship(
                            source_id=doc1.metadata.content_id,
                            target_id=doc2.metadata.content_id,
                            relationship_type="references",
                            strength=0.8,
                            explanation=f"Document references '{doc2.metadata.title}'"
                        )
                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Relationship detection failed: {e}")
            return []
    
    def _calculate_similarity(
        self,
        doc1: KnowledgeDocument,
        doc2: KnowledgeDocument
    ) -> float:
        """Calculate similarity between two documents."""
        try:
            # Simple similarity based on common keywords
            content1_words = set(doc1.content.lower().split())
            content2_words = set(doc2.content.lower().split())
            
            # Jaccard similarity
            intersection = len(content1_words & content2_words)
            union = len(content1_words | content2_words)
            
            if union == 0:
                return 0.0
            
            jaccard_sim = intersection / union
            
            # Category similarity boost
            analysis1 = self.content_cache.get(doc1.metadata.content_id)
            analysis2 = self.content_cache.get(doc2.metadata.content_id)
            
            category_boost = 0.0
            if analysis1 and analysis2:
                if analysis1.suggested_category == analysis2.suggested_category:
                    category_boost = 0.2
            
            return min(1.0, jaccard_sim + category_boost)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _generate_quality_summary(
        self,
        analyses: List[ContentAnalysis]
    ) -> Dict[str, Any]:
        """Generate quality summary from content analyses."""
        try:
            if not analyses:
                return {}
            
            summary = {
                "total_documents": len(analyses),
                "average_confidence": sum(a.confidence_score for a in analyses) / len(analyses),
                "average_complexity": sum(a.complexity_score for a in analyses) / len(analyses),
                "category_distribution": {},
                "quality_scores": {}
            }
            
            # Category distribution
            category_counts = defaultdict(int)
            for analysis in analyses:
                category_counts[analysis.suggested_category.value] += 1
            summary["category_distribution"] = dict(category_counts)
            
            # Quality metrics average
            all_metrics = defaultdict(list)
            for analysis in analyses:
                for metric, score in analysis.quality_metrics.items():
                    all_metrics[metric.value].append(score)
            
            for metric, scores in all_metrics.items():
                summary["quality_scores"][metric] = sum(scores) / len(scores)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Quality summary generation failed: {e}")
            return {}
    
    async def get_content_recommendations(
        self,
        content_id: ContentId
    ) -> Either[str, Dict[str, Any]]:
        """Get improvement recommendations for content."""
        try:
            analysis = self.content_cache.get(content_id)
            if not analysis:
                return Either.left(f"No analysis found for content {content_id}")
            
            recommendations = {
                "category_confidence": analysis.confidence_score,
                "suggestions": [],
                "quality_improvements": [],
                "related_content": []
            }
            
            # Category confidence recommendations
            if analysis.confidence_score < 0.7:
                recommendations["suggestions"].append(
                    "Consider reviewing content categorization - low confidence in current category"
                )
            
            # Quality improvements
            for metric, score in analysis.quality_metrics.items():
                if score < 70.0:
                    recommendations["quality_improvements"].append({
                        "metric": metric.value,
                        "current_score": score,
                        "suggestion": self._get_quality_suggestion(metric, score)
                    })
            
            # Related content
            related = [r for r in self.relationships if r.source_id == content_id or r.target_id == content_id]
            recommendations["related_content"] = [
                {
                    "content_id": r.target_id if r.source_id == content_id else r.source_id,
                    "relationship": r.relationship_type,
                    "strength": r.strength
                }
                for r in related
            ]
            
            return Either.right(recommendations)
            
        except Exception as e:
            error_msg = f"Failed to get recommendations: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def _get_quality_suggestion(self, metric: QualityMetric, score: float) -> str:
        """Get specific suggestion for quality improvement."""
        suggestions = {
            QualityMetric.CLARITY: "Improve readability by using shorter sentences and clearer language",
            QualityMetric.COMPLETENESS: "Add missing sections like overview, examples, or conclusion",
            QualityMetric.CONSISTENCY: "Use consistent formatting for headers, lists, and code blocks",
            QualityMetric.FRESHNESS: "Update content with recent information and current best practices",
            QualityMetric.ACCESSIBILITY: "Add clear headers and ensure reasonable content length"
        }
        return suggestions.get(metric, "Review and improve content quality")


# Global instance
_content_organizer: Optional[ContentOrganizer] = None


def get_content_organizer(config: OrganizationConfig = None) -> ContentOrganizer:
    """Get or create the global content organizer instance."""
    global _content_organizer
    if _content_organizer is None:
        _content_organizer = ContentOrganizer(config)
    return _content_organizer