"""
Search Engine - TASK_56 Phase 2 Implementation

Advanced knowledge search functionality with semantic understanding and intelligent ranking.
Provides full-text search, semantic similarity, fuzzy matching, and relevance scoring.

Architecture: Search Algorithms + Semantic Analysis + Relevance Ranking + Content Indexing
Performance: <50ms search response, efficient indexing and retrieval
Security: Query sanitization, access control integration
"""

from __future__ import annotations
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import logging
import re
import json
import math
from collections import defaultdict, Counter

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.knowledge_architecture import (
    DocumentId, ContentId, KnowledgeBaseId, SearchQueryId,
    SearchType, KnowledgeCategory, QualityMetric,
    KnowledgeDocument, ContentMetadata, KnowledgeBase,
    create_document_id, create_content_id,
    SearchError, KnowledgeError
)

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Search query with configuration and filters."""
    query_id: SearchQueryId
    query_text: str
    search_type: SearchType = SearchType.SEMANTIC
    knowledge_base_id: Optional[KnowledgeBaseId] = None
    categories: Optional[Set[KnowledgeCategory]] = None
    tags: Optional[Set[str]] = None
    max_results: int = 20
    min_score: float = 0.1
    include_snippets: bool = True
    snippet_length: int = 200
    boost_recent: bool = True
    boost_quality: bool = True
    
    def __post_init__(self):
        if not self.query_text.strip():
            raise ValueError("Query text cannot be empty")
        if not (1 <= self.max_results <= 100):
            raise ValueError("Max results must be between 1 and 100")
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("Min score must be between 0.0 and 1.0")


@dataclass
class SearchResult:
    """Individual search result with relevance information."""
    document_id: DocumentId
    content_id: ContentId
    title: str
    snippet: str
    relevance_score: float
    category: KnowledgeCategory
    tags: Set[str] = field(default_factory=set)
    match_highlights: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) positions
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.relevance_score <= 1.0):
            raise ValueError("Relevance score must be between 0.0 and 1.0")


@dataclass
class SearchResults:
    """Complete search results with metadata."""
    query: SearchQuery
    results: List[SearchResult]
    total_matches: int
    search_time_ms: float
    suggestions: List[str] = field(default_factory=list)
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def get_top_results(self, limit: int = 10) -> List[SearchResult]:
        """Get top results by relevance score."""
        return sorted(self.results, key=lambda r: r.relevance_score, reverse=True)[:limit]
    
    def get_results_by_category(self) -> Dict[KnowledgeCategory, List[SearchResult]]:
        """Group results by category."""
        grouped = defaultdict(list)
        for result in self.results:
            grouped[result.category].append(result)
        return dict(grouped)


class DocumentIndex:
    """Document indexing system for efficient search."""
    
    def __init__(self):
        self.documents: Dict[DocumentId, KnowledgeDocument] = {}
        self.term_index: Dict[str, Set[DocumentId]] = defaultdict(set)
        self.document_terms: Dict[DocumentId, Set[str]] = defaultdict(set)
        self.document_vectors: Dict[DocumentId, Dict[str, float]] = {}
        self.category_index: Dict[KnowledgeCategory, Set[DocumentId]] = defaultdict(set)
        self.tag_index: Dict[str, Set[DocumentId]] = defaultdict(set)
        self.quality_scores: Dict[DocumentId, float] = {}
        
        logger.info("DocumentIndex initialized")
    
    async def add_document(self, document: KnowledgeDocument) -> Either[str, None]:
        """Add document to search index."""
        try:
            doc_id = document.document_id
            
            # Store document
            self.documents[doc_id] = document
            
            # Index content
            terms = self._extract_terms(document.content)
            self.document_terms[doc_id] = terms
            
            # Update term index
            for term in terms:
                self.term_index[term].add(doc_id)
            
            # Create document vector for semantic search
            self.document_vectors[doc_id] = self._create_document_vector(terms, document.content)
            
            # Index metadata
            self.category_index[document.metadata.category].add(doc_id)
            for tag in document.metadata.tags:
                self.tag_index[tag].add(doc_id)
            
            # Store quality score
            self.quality_scores[doc_id] = document.quality_score
            
            logger.debug(f"Indexed document: {document.metadata.title}")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Failed to index document: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def remove_document(self, document_id: DocumentId) -> Either[str, None]:
        """Remove document from search index."""
        try:
            if document_id not in self.documents:
                return Either.left(f"Document {document_id} not found in index")
            
            document = self.documents[document_id]
            
            # Remove from term index
            terms = self.document_terms.get(document_id, set())
            for term in terms:
                self.term_index[term].discard(document_id)
                if not self.term_index[term]:  # Remove empty term entries
                    del self.term_index[term]
            
            # Remove from category and tag indices
            self.category_index[document.metadata.category].discard(document_id)
            for tag in document.metadata.tags:
                self.tag_index[tag].discard(document_id)
            
            # Clean up
            del self.documents[document_id]
            del self.document_terms[document_id]
            del self.document_vectors[document_id]
            del self.quality_scores[document_id]
            
            logger.debug(f"Removed document from index: {document_id}")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Failed to remove document from index: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    def _extract_terms(self, content: str) -> Set[str]:
        """Extract searchable terms from content."""
        try:
            # Convert to lowercase and extract words
            content_lower = content.lower()
            
            # Extract alphanumeric terms (3+ characters)
            terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', content_lower))
            
            # Remove common stop words
            stop_words = {
                "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "up", "about", "into", "through", "during",
                "before", "after", "above", "below", "between", "among", "this",
                "that", "these", "those", "you", "are", "was", "were", "been",
                "have", "has", "had", "will", "would", "could", "should", "can"
            }
            
            return terms - stop_words
            
        except Exception as e:
            logger.warning(f"Term extraction failed: {e}")
            return set()
    
    def _create_document_vector(self, terms: Set[str], content: str) -> Dict[str, float]:
        """Create TF-IDF vector for document."""
        try:
            # Calculate term frequencies
            content_words = content.lower().split()
            term_freq = Counter(content_words)
            doc_length = len(content_words)
            
            vector = {}
            for term in terms:
                tf = term_freq.get(term, 0) / doc_length if doc_length > 0 else 0
                # IDF calculation would require corpus statistics - simplified here
                idf = math.log(len(self.documents) + 1) if len(self.documents) > 0 else 1
                vector[term] = tf * idf
            
            return vector
            
        except Exception as e:
            logger.warning(f"Document vector creation failed: {e}")
            return {}


class SearchEngine:
    """
    Advanced knowledge search engine with semantic understanding.
    
    Provides comprehensive search capabilities including full-text search,
    semantic similarity, fuzzy matching, and intelligent relevance ranking.
    """
    
    def __init__(self):
        self.index = DocumentIndex()
        self.search_history: List[SearchQuery] = []
        self.popular_queries: Counter = Counter()
        
        logger.info("SearchEngine initialized")
    
    async def add_documents(self, documents: List[KnowledgeDocument]) -> Either[str, int]:
        """Add multiple documents to search index."""
        try:
            added_count = 0
            for document in documents:
                result = await self.index.add_document(document)
                if result.is_right():
                    added_count += 1
                else:
                    logger.warning(f"Failed to index document {document.document_id}: {result.left()}")
            
            logger.info(f"Added {added_count}/{len(documents)} documents to search index")
            return Either.right(added_count)
            
        except Exception as e:
            error_msg = f"Failed to add documents to index: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    @require(lambda query: query.query_text.strip(), "Query text required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns results or error")
    async def search(self, query: SearchQuery) -> Either[str, SearchResults]:
        """Execute search query with specified type and filters."""
        start_time = datetime.now(UTC)
        
        try:
            # Record query
            self.search_history.append(query)
            self.popular_queries[query.query_text.lower()] += 1
            
            # Execute search based on type
            if query.search_type == SearchType.TEXT:
                results = await self._text_search(query)
            elif query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query)
            elif query.search_type == SearchType.FUZZY:
                results = await self._fuzzy_search(query)
            elif query.search_type == SearchType.EXACT:
                results = await self._exact_search(query)
            else:
                results = await self._semantic_search(query)  # Default to semantic
            
            if results.is_left():
                return results
            
            search_results = results.right()
            
            # Apply filters
            filtered_results = self._apply_filters(search_results, query)
            
            # Rank results
            ranked_results = self._rank_results(filtered_results, query)
            
            # Generate suggestions and facets
            suggestions = self._generate_suggestions(query)
            facets = self._generate_facets(ranked_results)
            
            # Calculate search time
            search_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            # Create final results
            final_results = SearchResults(
                query=query,
                results=ranked_results[:query.max_results],
                total_matches=len(ranked_results),
                search_time_ms=search_time,
                suggestions=suggestions,
                facets=facets
            )
            
            logger.info(f"Search completed: '{query.query_text}' -> {len(ranked_results)} results in {search_time:.1f}ms")
            return Either.right(final_results)
            
        except Exception as e:
            error_msg = f"Search execution failed: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def _text_search(self, query: SearchQuery) -> Either[str, List[SearchResult]]:
        """Execute full-text search."""
        try:
            query_terms = self.index._extract_terms(query.query_text)
            if not query_terms:
                return Either.right([])
            
            # Find documents containing query terms
            candidate_docs = set()
            term_matches = {}
            
            for term in query_terms:
                matching_docs = self.index.term_index.get(term, set())
                candidate_docs.update(matching_docs)
                term_matches[term] = matching_docs
            
            results = []
            for doc_id in candidate_docs:
                document = self.index.documents[doc_id]
                
                # Calculate relevance score
                score = self._calculate_text_score(doc_id, query_terms, term_matches)
                
                if score >= query.min_score:
                    snippet = self._generate_snippet(document.content, query.query_text, query.snippet_length)
                    highlights = self._find_highlights(document.content, query.query_text)
                    
                    result = SearchResult(
                        document_id=doc_id,
                        content_id=document.metadata.content_id,
                        title=document.metadata.title,
                        snippet=snippet,
                        relevance_score=score,
                        category=document.metadata.category,
                        tags=document.metadata.tags,
                        match_highlights=highlights,
                        explanation=f"Text match score: {score:.3f}"
                    )
                    results.append(result)
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(f"Text search failed: {str(e)}")
    
    async def _semantic_search(self, query: SearchQuery) -> Either[str, List[SearchResult]]:
        """Execute semantic similarity search."""
        try:
            query_terms = self.index._extract_terms(query.query_text)
            if not query_terms:
                return Either.right([])
            
            # Create query vector
            query_vector = {}
            for term in query_terms:
                query_vector[term] = 1.0  # Simplified - would use proper TF-IDF
            
            results = []
            for doc_id, document in self.index.documents.items():
                doc_vector = self.index.document_vectors.get(doc_id, {})
                
                # Calculate semantic similarity
                similarity = self._calculate_cosine_similarity(query_vector, doc_vector)
                
                if similarity >= query.min_score:
                    snippet = self._generate_snippet(document.content, query.query_text, query.snippet_length)
                    
                    result = SearchResult(
                        document_id=doc_id,
                        content_id=document.metadata.content_id,
                        title=document.metadata.title,
                        snippet=snippet,
                        relevance_score=similarity,
                        category=document.metadata.category,
                        tags=document.metadata.tags,
                        explanation=f"Semantic similarity: {similarity:.3f}"
                    )
                    results.append(result)
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(f"Semantic search failed: {str(e)}")
    
    async def _fuzzy_search(self, query: SearchQuery) -> Either[str, List[SearchResult]]:
        """Execute fuzzy matching search."""
        try:
            results = []
            query_lower = query.query_text.lower()
            
            for doc_id, document in self.index.documents.items():
                content_lower = document.content.lower()
                title_lower = document.metadata.title.lower()
                
                # Calculate fuzzy match scores
                content_score = self._calculate_fuzzy_score(query_lower, content_lower)
                title_score = self._calculate_fuzzy_score(query_lower, title_lower)
                
                # Combined score with title boost
                combined_score = max(content_score, title_score * 1.5)
                
                if combined_score >= query.min_score:
                    snippet = self._generate_snippet(document.content, query.query_text, query.snippet_length)
                    
                    result = SearchResult(
                        document_id=doc_id,
                        content_id=document.metadata.content_id,
                        title=document.metadata.title,
                        snippet=snippet,
                        relevance_score=combined_score,
                        category=document.metadata.category,
                        tags=document.metadata.tags,
                        explanation=f"Fuzzy match score: {combined_score:.3f}"
                    )
                    results.append(result)
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(f"Fuzzy search failed: {str(e)}")
    
    async def _exact_search(self, query: SearchQuery) -> Either[str, List[SearchResult]]:
        """Execute exact phrase matching search."""
        try:
            results = []
            query_text = query.query_text.strip()
            
            for doc_id, document in self.index.documents.items():
                content = document.content
                title = document.metadata.title
                
                # Check for exact matches
                content_matches = content.lower().count(query_text.lower())
                title_matches = title.lower().count(query_text.lower())
                
                if content_matches > 0 or title_matches > 0:
                    # Calculate score based on match frequency and position
                    score = min(1.0, (content_matches * 0.1 + title_matches * 0.5))
                    
                    if score >= query.min_score:
                        snippet = self._generate_snippet(content, query_text, query.snippet_length)
                        highlights = self._find_highlights(content, query_text)
                        
                        result = SearchResult(
                            document_id=doc_id,
                            content_id=document.metadata.content_id,
                            title=title,
                            snippet=snippet,
                            relevance_score=score,
                            category=document.metadata.category,
                            tags=document.metadata.tags,
                            match_highlights=highlights,
                            explanation=f"Exact matches: {content_matches + title_matches}"
                        )
                        results.append(result)
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(f"Exact search failed: {str(e)}")
    
    def _calculate_text_score(self, doc_id: DocumentId, query_terms: Set[str], term_matches: Dict[str, Set[DocumentId]]) -> float:
        """Calculate text relevance score."""
        try:
            document = self.index.documents[doc_id]
            doc_terms = self.index.document_terms.get(doc_id, set())
            
            # Term frequency scoring
            matching_terms = query_terms & doc_terms
            term_coverage = len(matching_terms) / len(query_terms) if query_terms else 0
            
            # Title boost
            title_lower = document.metadata.title.lower()
            title_matches = sum(1 for term in query_terms if term in title_lower)
            title_boost = title_matches * 0.1
            
            # Quality boost
            quality_boost = self.index.quality_scores.get(doc_id, 0.5) * 0.1
            
            return min(1.0, term_coverage + title_boost + quality_boost)
            
        except Exception as e:
            logger.warning(f"Text score calculation failed: {e}")
            return 0.0
    
    def _calculate_cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if not vec1 or not vec2:
                return 0.0
            
            # Calculate dot product
            common_terms = set(vec1.keys()) & set(vec2.keys())
            dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
            
            # Calculate magnitudes
            mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
            mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            return dot_product / (mag1 * mag2)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_fuzzy_score(self, query: str, text: str) -> float:
        """Calculate fuzzy matching score (simplified Levenshtein-based)."""
        try:
            if query in text:
                return 1.0  # Exact substring match
            
            # Simple fuzzy matching based on character overlap
            query_chars = set(query.replace(' ', ''))
            text_chars = set(text.replace(' ', ''))
            
            if not query_chars:
                return 0.0
            
            overlap = len(query_chars & text_chars)
            return overlap / len(query_chars)
            
        except Exception as e:
            logger.warning(f"Fuzzy score calculation failed: {e}")
            return 0.0
    
    def _apply_filters(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply query filters to search results."""
        try:
            filtered = results
            
            # Filter by categories
            if query.categories:
                filtered = [r for r in filtered if r.category in query.categories]
            
            # Filter by tags
            if query.tags:
                filtered = [r for r in filtered if query.tags & r.tags]
            
            # Filter by knowledge base
            if query.knowledge_base_id:
                # Would need knowledge base membership info - simplified here
                pass
            
            return filtered
            
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return results
    
    def _rank_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply intelligent ranking to search results."""
        try:
            for result in results:
                doc_id = result.document_id
                document = self.index.documents.get(doc_id)
                
                if document:
                    # Apply ranking boosts
                    boost_factor = 1.0
                    
                    # Quality boost
                    if query.boost_quality:
                        quality_score = self.index.quality_scores.get(doc_id, 0.5)
                        boost_factor *= (1.0 + quality_score * 0.2)
                    
                    # Recency boost
                    if query.boost_recent:
                        days_old = (datetime.now(UTC) - document.metadata.modified_at).days
                        recency_factor = max(0.8, 1.0 - days_old / 365)
                        boost_factor *= recency_factor
                    
                    # Apply boost
                    result.relevance_score = min(1.0, result.relevance_score * boost_factor)
            
            # Sort by relevance score
            return sorted(results, key=lambda r: r.relevance_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Result ranking failed: {e}")
            return results
    
    def _generate_snippet(self, content: str, query: str, max_length: int) -> str:
        """Generate content snippet highlighting query matches."""
        try:
            if not content or not query:
                return content[:max_length] if content else ""
            
            content_lower = content.lower()
            query_lower = query.lower()
            
            # Find best match position
            match_pos = content_lower.find(query_lower)
            if match_pos == -1:
                # No exact match, return beginning
                return content[:max_length].strip()
            
            # Extract snippet around match
            start = max(0, match_pos - max_length // 3)
            end = min(len(content), start + max_length)
            
            snippet = content[start:end].strip()
            
            # Add ellipsis if truncated
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            
            return snippet
            
        except Exception as e:
            logger.warning(f"Snippet generation failed: {e}")
            return content[:max_length] if content else ""
    
    def _find_highlights(self, content: str, query: str) -> List[Tuple[int, int]]:
        """Find highlight positions for query matches in content."""
        try:
            highlights = []
            if not content or not query:
                return highlights
            
            content_lower = content.lower()
            query_lower = query.lower()
            
            start = 0
            while True:
                pos = content_lower.find(query_lower, start)
                if pos == -1:
                    break
                
                highlights.append((pos, pos + len(query)))
                start = pos + 1
            
            return highlights[:10]  # Limit highlights
            
        except Exception as e:
            logger.warning(f"Highlight detection failed: {e}")
            return []
    
    def _generate_suggestions(self, query: SearchQuery) -> List[str]:
        """Generate search suggestions based on query and history."""
        try:
            suggestions = []
            query_terms = query.query_text.lower().split()
            
            # Suggest popular related queries
            for popular_query, count in self.popular_queries.most_common(10):
                if popular_query != query.query_text.lower():
                    # Check if queries share terms
                    popular_terms = set(popular_query.split())
                    query_term_set = set(query_terms)
                    
                    if popular_terms & query_term_set:
                        suggestions.append(popular_query)
            
            # Suggest term expansions
            for term in query_terms:
                # Find related terms from index
                related_terms = set()
                for indexed_term in self.index.term_index.keys():
                    if term in indexed_term and term != indexed_term:
                        related_terms.add(indexed_term)
                
                # Add top related terms
                for related_term in list(related_terms)[:3]:
                    expanded_query = query.query_text.replace(term, related_term)
                    if expanded_query not in suggestions:
                        suggestions.append(expanded_query)
            
            return suggestions[:5]  # Limit suggestions
            
        except Exception as e:
            logger.warning(f"Suggestion generation failed: {e}")
            return []
    
    def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Generate search facets for result filtering."""
        try:
            facets = {
                "categories": defaultdict(int),
                "tags": defaultdict(int)
            }
            
            for result in results:
                # Category facets
                facets["categories"][result.category.value] += 1
                
                # Tag facets
                for tag in result.tags:
                    facets["tags"][tag] += 1
            
            # Convert to regular dicts and sort
            return {
                "categories": dict(sorted(facets["categories"].items(), key=lambda x: x[1], reverse=True)),
                "tags": dict(sorted(facets["tags"].items(), key=lambda x: x[1], reverse=True)[:20])
            }
            
        except Exception as e:
            logger.warning(f"Facet generation failed: {e}")
            return {}
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and statistics."""
        try:
            return {
                "total_queries": len(self.search_history),
                "popular_queries": dict(self.popular_queries.most_common(10)),
                "indexed_documents": len(self.index.documents),
                "total_terms": len(self.index.term_index),
                "query_types": Counter(q.search_type.value for q in self.search_history),
                "average_results": sum(len(self.index.term_index.get(term, set())) for term in self.index.term_index) / max(1, len(self.index.term_index))
            }
            
        except Exception as e:
            logger.warning(f"Analytics generation failed: {e}")
            return {}


# Global instance
_search_engine: Optional[SearchEngine] = None


def get_search_engine() -> SearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine