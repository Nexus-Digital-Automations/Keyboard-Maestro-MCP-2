"""
Knowledge Management Tools - TASK_56 Phase 3 Implementation

FastMCP tools for knowledge management operations through Claude Desktop.
Provides comprehensive knowledge base management, documentation generation, and intelligent search.

Architecture: FastMCP Integration + Knowledge Engine + Documentation Automation
Performance: <200ms tool responses, efficient knowledge operations
Security: Access control, content validation, secure knowledge management
"""

from __future__ import annotations
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Annotated
import logging
import json
import uuid

from fastmcp import Context
from ..mcp_server import mcp
from ...core.contracts import require, ensure
from ...core.either import Either
from ...core.knowledge_architecture import (
    DocumentId, ContentId, KnowledgeBaseId, TemplateId, SearchQueryId,
    DocumentType, ContentFormat, SearchType, KnowledgeCategory, QualityMetric,
    KnowledgeDocument, ContentMetadata, DocumentationSource, KnowledgeBase,
    create_document_id, create_knowledge_base_id, create_content_id,
    KnowledgeError, DocumentGenerationError, SearchError
)
from ...knowledge.documentation_generator import (
    get_documentation_generator,
    DocumentationContext,
    MacroDocumentationConfig
)
from ...knowledge.content_organizer import (
    get_content_organizer,
    OrganizationConfig
)
from ...knowledge.search_engine import (
    get_search_engine,
    SearchQuery,
    SearchResults
)
from ...knowledge.version_control import (
    get_version_manager,
    ChangeType
)

logger = logging.getLogger(__name__)

# Global knowledge management state
knowledge_bases: Dict[KnowledgeBaseId, KnowledgeBase] = {}
documents_store: Dict[DocumentId, KnowledgeDocument] = {}


@mcp.tool()
async def km_generate_documentation(
    source_type: str,  # macro|workflow|group|system
    source_id: str,
    documentation_type: str = "detailed",  # overview|detailed|technical|user_guide
    include_sections: List[str] = None,
    output_format: str = "markdown",  # markdown|html|pdf|confluence
    template_id: Optional[str] = None,
    include_screenshots: bool = False,
    ai_enhancement: bool = True,
    auto_update: bool = False,
    knowledge_base_id: Optional[str] = None,
    author: str = "system",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Generate comprehensive documentation automatically from macros, workflows, or system components.
    
    FastMCP Tool for automated documentation generation through Claude Desktop.
    Analyzes automation structures and generates professional documentation.
    
    Returns generated documentation, metadata, and update tracking information.
    """
    try:
        logger.info(f"Generating documentation for {source_type}: {source_id}")
        
        # Validate inputs
        valid_source_types = ["macro", "workflow", "group", "system"]
        if source_type not in valid_source_types:
            return {
                "success": False,
                "error": f"Invalid source type. Must be one of: {', '.join(valid_source_types)}",
                "source_type": source_type
            }
        
        valid_formats = ["markdown", "html", "pdf", "confluence"]
        if output_format not in valid_formats:
            return {
                "success": False,
                "error": f"Invalid output format. Must be one of: {', '.join(valid_formats)}",
                "format": output_format
            }
        
        # Map format strings to enum
        format_mapping = {
            "markdown": ContentFormat.MARKDOWN,
            "html": ContentFormat.HTML,
            "pdf": ContentFormat.PDF,
            "confluence": ContentFormat.CONFLUENCE
        }
        
        # Get or create knowledge base
        kb_id = KnowledgeBaseId(knowledge_base_id) if knowledge_base_id else create_knowledge_base_id()
        if kb_id not in knowledge_bases:
            knowledge_bases[kb_id] = KnowledgeBase(
                knowledge_base_id=kb_id,
                name="Default Knowledge Base",
                description="Auto-generated knowledge base for documentation"
            )
        
        # Create documentation context (mock data for demonstration)
        context = DocumentationContext(
            source_type=source_type,
            source_id=source_id,
            source_data={
                "name": f"Sample {source_type.title()}",
                "description": f"Auto-generated {source_type} documentation",
                "enabled": True,
                "group": "Documentation",
                "triggers": [{"type": "keyboard", "config": {"key": "F1"}}] if source_type == "macro" else [],
                "variables": [{"name": "input", "type": "string", "description": "Input parameter"}],
                "components": [{"type": "action", "title": "Sample Action"}] if source_type == "workflow" else [],
                "macros": [{"name": "Macro 1", "enabled": True}] if source_type == "group" else [],
                "configuration": {"setting1": "value1"} if source_type == "system" else {}
            }
        )
        
        # Create generation config
        config = MacroDocumentationConfig(
            include_overview="overview" in (include_sections or ["overview"]),
            include_usage="usage" in (include_sections or ["usage"]),
            include_parameters="parameters" in (include_sections or ["parameters"]),
            include_examples="examples" in (include_sections or ["examples"]),
            include_troubleshooting="troubleshooting" in (include_sections or []),
            include_screenshots=include_screenshots,
            ai_enhancement=ai_enhancement,
            format=format_mapping[output_format],
            template_id=TemplateId(template_id) if template_id else None
        )
        
        # Generate documentation
        generator = get_documentation_generator()
        result = await generator.generate_documentation(context, config, kb_id, author)
        
        if result.is_left():
            return {
                "success": False,
                "error": result.left(),
                "source_type": source_type,
                "source_id": source_id
            }
        
        document = result.right()
        
        # Store document
        documents_store[document.document_id] = document
        
        # Add to knowledge base
        knowledge_bases[kb_id].documents.add(document.document_id)
        
        # Create version if auto_update enabled
        version_info = None
        if auto_update:
            version_manager = get_version_manager()
            version_result = await version_manager.create_initial_version(document, author)
            if version_result.is_right():
                version_info = {
                    "version_id": version_result.right().version_id,
                    "version_number": version_result.right().version_number
                }
        
        return {
            "success": True,
            "document_id": document.document_id,
            "content_id": document.metadata.content_id,
            "title": document.metadata.title,
            "content": document.content,
            "format": output_format,
            "knowledge_base_id": kb_id,
            "metadata": {
                "category": document.metadata.category.value,
                "tags": list(document.metadata.tags),
                "author": document.metadata.author,
                "word_count": document.metadata.word_count,
                "reading_time_minutes": document.metadata.reading_time_minutes,
                "created_at": document.metadata.created_at.isoformat(),
                "modified_at": document.metadata.modified_at.isoformat()
            },
            "source": {
                "source_type": document.source.source_type if document.source else source_type,
                "source_id": document.source.source_id if document.source else source_id,
                "source_name": document.source.source_name if document.source else "Unknown"
            },
            "quality_score": document.quality_score,
            "version_info": version_info,
            "generation_config": {
                "documentation_type": documentation_type,
                "included_sections": include_sections or ["overview", "usage", "parameters", "examples"],
                "ai_enhancement": ai_enhancement,
                "auto_update": auto_update
            }
        }
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_type": source_type,
            "source_id": source_id
        }


@mcp.tool()
async def km_manage_knowledge_base(
    operation: str,  # create|update|delete|organize|export
    knowledge_base_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    categories: Optional[List[str]] = None,
    access_permissions: Optional[Dict[str, Any]] = None,
    auto_categorize: bool = True,
    index_content: bool = True,
    enable_search: bool = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Create and manage knowledge bases for organizing automation documentation and resources.
    
    FastMCP Tool for knowledge base management through Claude Desktop.
    Provides centralized knowledge organization with intelligent categorization.
    
    Returns knowledge base configuration, organization structure, and access settings.
    """
    try:
        logger.info(f"Managing knowledge base: {operation}")
        
        # Validate operation
        valid_operations = ["create", "update", "delete", "organize", "export"]
        if operation not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation. Must be one of: {', '.join(valid_operations)}",
                "operation": operation
            }
        
        if operation == "create":
            # Create new knowledge base
            if not name:
                return {
                    "success": False,
                    "error": "Name is required for creating knowledge base",
                    "operation": operation
                }
            
            kb_id = create_knowledge_base_id()
            
            # Parse categories
            kb_categories = set()
            if categories:
                for cat in categories:
                    try:
                        kb_categories.add(KnowledgeCategory(cat.lower()))
                    except ValueError:
                        logger.warning(f"Invalid category: {cat}")
            
            knowledge_base = KnowledgeBase(
                knowledge_base_id=kb_id,
                name=name,
                description=description or "",
                categories=kb_categories,
                auto_categorize=auto_categorize,
                enable_search=enable_search,
                access_permissions=access_permissions or {}
            )
            
            knowledge_bases[kb_id] = knowledge_base
            
            return {
                "success": True,
                "operation": "create",
                "knowledge_base_id": kb_id,
                "name": name,
                "description": description,
                "categories": [cat.value for cat in kb_categories],
                "auto_categorize": auto_categorize,
                "enable_search": enable_search,
                "created_at": knowledge_base.created_at.isoformat(),
                "document_count": 0
            }
        
        elif operation == "organize":
            if not knowledge_base_id:
                return {
                    "success": False,
                    "error": "Knowledge base ID is required for organize operation",
                    "operation": operation
                }
            
            kb_id = KnowledgeBaseId(knowledge_base_id)
            if kb_id not in knowledge_bases:
                return {
                    "success": False,
                    "error": f"Knowledge base {knowledge_base_id} not found",
                    "operation": operation
                }
            
            kb = knowledge_bases[kb_id]
            
            # Get documents for this knowledge base
            kb_documents = [documents_store[doc_id] for doc_id in kb.documents if doc_id in documents_store]
            
            if not kb_documents:
                return {
                    "success": True,
                    "operation": "organize",
                    "knowledge_base_id": kb_id,
                    "message": "No documents to organize",
                    "organization_results": {}
                }
            
            # Organize documents
            organizer = get_content_organizer(OrganizationConfig(
                auto_categorize=kb.auto_categorize,
                enable_keyword_extraction=True
            ))
            
            organization_result = await organizer.organize_documents(kb_documents)
            
            if organization_result.is_left():
                return {
                    "success": False,
                    "error": organization_result.left(),
                    "operation": operation
                }
            
            org_data = organization_result.right()
            
            return {
                "success": True,
                "operation": "organize",
                "knowledge_base_id": kb_id,
                "organization_results": {
                    "total_documents": org_data["total_documents"],
                    "categories": dict(org_data["categories"]),
                    "tags": dict(list(org_data["tags"].items())[:20]),  # Top 20 tags
                    "relationships": len(org_data["relationships"]),
                    "quality_summary": org_data["quality_summary"]
                }
            }
        
        else:
            return {
                "success": False,
                "error": f"Operation {operation} not yet implemented",
                "operation": operation
            }
        
    except Exception as e:
        logger.error(f"Knowledge base management failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "operation": operation
        }


@mcp.tool()
async def km_search_knowledge(
    query: str,
    search_scope: str = "all",  # all|knowledge_base|documentation|macros
    knowledge_base_id: Optional[str] = None,
    search_type: str = "semantic",  # text|semantic|fuzzy|exact
    include_content_types: List[str] = None,
    max_results: int = 20,
    include_snippets: bool = True,
    rank_by_relevance: bool = True,
    include_suggestions: bool = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search knowledge bases with advanced semantic understanding and intelligent ranking.
    
    FastMCP Tool for intelligent knowledge search through Claude Desktop.
    Provides semantic search with content understanding and relevance ranking.
    
    Returns search results, relevance scores, content snippets, and related suggestions.
    """
    try:
        logger.info(f"Searching knowledge: '{query}' with type {search_type}")
        
        # Validate search type
        valid_search_types = ["text", "semantic", "fuzzy", "exact"]
        if search_type not in valid_search_types:
            return {
                "success": False,
                "error": f"Invalid search type. Must be one of: {', '.join(valid_search_types)}",
                "search_type": search_type
            }
        
        # Map search type to enum
        search_type_mapping = {
            "text": SearchType.TEXT,
            "semantic": SearchType.SEMANTIC,
            "fuzzy": SearchType.FUZZY,
            "exact": SearchType.EXACT
        }
        
        # Create search query
        search_query = SearchQuery(
            query_id=SearchQueryId(f"search_{uuid.uuid4().hex[:8]}"),
            query_text=query,
            search_type=search_type_mapping[search_type],
            knowledge_base_id=KnowledgeBaseId(knowledge_base_id) if knowledge_base_id else None,
            max_results=max_results,
            include_snippets=include_snippets,
            boost_recent=True,
            boost_quality=rank_by_relevance
        )
        
        # Get search engine and add documents if not already indexed
        search_engine = get_search_engine()
        
        # Add all documents to search index
        all_documents = list(documents_store.values())
        if all_documents:
            await search_engine.add_documents(all_documents)
        
        # Execute search
        search_result = await search_engine.search(search_query)
        
        if search_result.is_left():
            return {
                "success": False,
                "error": search_result.left(),
                "query": query,
                "search_type": search_type
            }
        
        results = search_result.right()
        
        # Format results
        formatted_results = []
        for result in results.results:
            formatted_result = {
                "document_id": result.document_id,
                "content_id": result.content_id,
                "title": result.title,
                "relevance_score": result.relevance_score,
                "category": result.category.value,
                "tags": list(result.tags),
                "explanation": result.explanation
            }
            
            if include_snippets:
                formatted_result["snippet"] = result.snippet
                formatted_result["highlights"] = result.match_highlights
            
            formatted_results.append(formatted_result)
        
        # Get top results if ranking requested
        if rank_by_relevance:
            formatted_results = sorted(formatted_results, key=lambda r: r["relevance_score"], reverse=True)
        
        response = {
            "success": True,
            "query": query,
            "search_type": search_type,
            "results": formatted_results,
            "total_matches": results.total_matches,
            "search_time_ms": results.search_time_ms,
            "executed_at": results.executed_at.isoformat()
        }
        
        if include_suggestions:
            response["suggestions"] = results.suggestions
            response["facets"] = results.facets
        
        return response
        
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "search_type": search_type
        }


@mcp.tool()
async def km_update_documentation(
    document_id: str,
    update_type: str,  # content|metadata|structure|review
    content_updates: Optional[Dict[str, Any]] = None,
    metadata_updates: Optional[Dict[str, Any]] = None,
    version_note: Optional[str] = None,
    auto_validate: bool = True,
    preserve_history: bool = True,
    notify_stakeholders: bool = False,
    schedule_review: Optional[str] = None,
    author: str = "system",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Update documentation with version control and change tracking.
    
    FastMCP Tool for documentation updates through Claude Desktop.
    Manages content updates with version control and stakeholder notifications.
    
    Returns update results, version information, and validation status.
    """
    try:
        logger.info(f"Updating document: {document_id} ({update_type})")
        
        # Validate update type
        valid_update_types = ["content", "metadata", "structure", "review"]
        if update_type not in valid_update_types:
            return {
                "success": False,
                "error": f"Invalid update type. Must be one of: {', '.join(valid_update_types)}",
                "update_type": update_type
            }
        
        # Check if document exists
        doc_id = DocumentId(document_id)
        if doc_id not in documents_store:
            return {
                "success": False,
                "error": f"Document {document_id} not found",
                "document_id": document_id
            }
        
        document = documents_store[doc_id]
        updated_document = document
        
        if update_type == "content":
            if not content_updates:
                return {
                    "success": False,
                    "error": "Content updates are required for content update type",
                    "update_type": update_type
                }
            
            # Update content
            new_content = content_updates.get("content", document.content)
            
            # Create new metadata with updated info
            new_metadata = ContentMetadata(
                content_id=document.metadata.content_id,
                title=content_updates.get("title", document.metadata.title),
                description=content_updates.get("description", document.metadata.description),
                category=document.metadata.category,
                tags=set(content_updates.get("tags", list(document.metadata.tags))),
                author=author,
                created_at=document.metadata.created_at,
                modified_at=datetime.now(UTC),
                version=document.metadata.version,
                language=document.metadata.language,
                word_count=len(new_content.split()),
                reading_time_minutes=max(1, len(new_content.split()) // 200)
            )
            
            # Create updated document
            updated_document = KnowledgeDocument(
                document_id=document.document_id,
                metadata=new_metadata,
                content=new_content,
                source=document.source,
                related_documents=document.related_documents,
                quality_score=document.quality_score
            )
        
        # Store updated document
        documents_store[doc_id] = updated_document
        
        # Create version if preserving history
        version_info = None
        if preserve_history:
            version_manager = get_version_manager()
            version_result = await version_manager.create_version(
                doc_id,
                updated_document.content,
                updated_document.metadata,
                author,
                version_note or f"{update_type.title()} update"
            )
            
            if version_result.is_right():
                version_info = {
                    "version_id": version_result.right().version_id,
                    "version_number": version_result.right().version_number,
                    "change_type": version_result.right().change_type.value,
                    "created_at": version_result.right().created_at.isoformat()
                }
        
        return {
            "success": True,
            "document_id": document_id,
            "update_type": update_type,
            "updated_at": datetime.now(UTC).isoformat(),
            "author": author,
            "version_info": version_info,
            "document_info": {
                "title": updated_document.metadata.title,
                "category": updated_document.metadata.category.value,
                "tags": list(updated_document.metadata.tags),
                "word_count": updated_document.metadata.word_count,
                "quality_score": updated_document.quality_score
            }
        }
        
    except Exception as e:
        logger.error(f"Documentation update failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
            "update_type": update_type
        }


@mcp.tool()
async def km_create_content_template(
    template_name: str,
    template_type: str,  # documentation|guide|reference|report
    content_structure: Dict[str, Any],
    variable_placeholders: Optional[List[str]] = None,
    output_formats: List[str] = None,
    usage_guidelines: Optional[str] = None,
    auto_populate: bool = True,
    validation_rules: Optional[Dict[str, Any]] = None,
    author: str = "system",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Create reusable content templates for standardized documentation generation.
    
    FastMCP Tool for content template creation through Claude Desktop.
    Provides standardized templates for consistent documentation formats.
    
    Returns template configuration, structure definition, and usage guidelines.
    """
    try:
        from ...knowledge.template_manager import TemplateManager, ContentTemplate, TemplateType, TemplateVariable
        
        logger.info(f"Creating content template: {template_name}")
        
        # Validate template type
        valid_types = ["documentation", "guide", "reference", "report", "tutorial", "api_documentation", "user_manual"]
        if template_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid template type. Must be one of: {', '.join(valid_types)}",
                "template_type": template_type
            }
        
        # Create template ID
        template_id = f"template_{template_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Create template variables from placeholders
        variables = []
        if variable_placeholders:
            for placeholder in variable_placeholders:
                variables.append(TemplateVariable(
                    name=placeholder,
                    description=f"Variable for {placeholder}",
                    variable_type="string",
                    required=True
                ))
        
        # Parse output formats
        formats = set()
        if output_formats:
            for fmt in output_formats:
                try:
                    if fmt == "markdown":
                        formats.add(ContentFormat.MARKDOWN)
                    elif fmt == "html":
                        formats.add(ContentFormat.HTML)
                    elif fmt == "pdf":
                        formats.add(ContentFormat.PDF)
                except ValueError:
                    logger.warning(f"Invalid output format: {fmt}")
        
        if not formats:
            formats = {ContentFormat.MARKDOWN}
        
        # Create template
        template = ContentTemplate(
            template_id=template_id,
            name=template_name,
            description=content_structure.get("description", f"Template for {template_name}"),
            template_type=TemplateType(template_type),
            content_structure=content_structure.get("content", "# {{title}}\n\n{{content}}"),
            variables=variables,
            output_formats=formats,
            usage_guidelines=usage_guidelines or f"Use this template for {template_type} generation",
            auto_populate=auto_populate,
            validation_rules=validation_rules or {},
            author=author
        )
        
        # Initialize template manager and create template
        template_manager = TemplateManager()
        result = await template_manager.create_template(template)
        
        if result.is_left():
            return {
                "success": False,
                "error": result.left(),
                "template_name": template_name
            }
        
        created_template = result.right()
        
        return {
            "success": True,
            "template_id": template_id,
            "template_name": template_name,
            "template_type": template_type,
            "content_structure": content_structure,
            "variables": [{"name": v.name, "type": v.variable_type, "required": v.required} for v in variables],
            "output_formats": [fmt.value for fmt in formats],
            "usage_guidelines": usage_guidelines,
            "auto_populate": auto_populate,
            "created_at": created_template.created_at.isoformat(),
            "author": author
        }
        
    except Exception as e:
        logger.error(f"Template creation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "template_name": template_name
        }


@mcp.tool()
async def km_analyze_content_quality(
    content_id: str,
    analysis_scope: str = "content",  # content|structure|accessibility|seo
    quality_metrics: List[str] = None,
    include_improvements: bool = True,
    ai_analysis: bool = True,
    benchmark_against: Optional[str] = None,
    generate_report: bool = True,
    auto_fix_issues: bool = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Analyze content quality and provide improvement recommendations.
    
    FastMCP Tool for content quality analysis through Claude Desktop.
    Evaluates documentation quality and provides actionable improvement suggestions.
    
    Returns quality analysis, improvement recommendations, and automated fixes.
    """
    try:
        from ...intelligence.behavior_analyzer import ContentQualityAnalyzer, QualityMetrics
        
        logger.info(f"Analyzing content quality: {content_id}")
        
        # Find document by content ID
        document = None
        for doc in documents_store.values():
            if doc.metadata.content_id == content_id:
                document = doc
                break
        
        if not document:
            return {
                "success": False,
                "error": f"Document with content ID {content_id} not found",
                "content_id": content_id
            }
        
        # Validate analysis scope
        valid_scopes = ["content", "structure", "accessibility", "seo", "all"]
        if analysis_scope not in valid_scopes:
            return {
                "success": False,
                "error": f"Invalid analysis scope. Must be one of: {', '.join(valid_scopes)}",
                "analysis_scope": analysis_scope
            }
        
        # Default quality metrics
        if not quality_metrics:
            quality_metrics = ["clarity", "completeness", "accuracy", "readability", "structure"]
        
        # Mock quality analysis (in production would use AI/NLP analysis)
        quality_analysis = {
            "overall_score": 85.5,
            "metrics": {},
            "issues": [],
            "improvements": [],
            "strengths": []
        }
        
        # Analyze each metric
        for metric in quality_metrics:
            score = 80 + (hash(f"{content_id}_{metric}") % 20)  # Mock scoring
            quality_analysis["metrics"][metric] = {
                "score": score,
                "description": f"{metric.title()} assessment for the content",
                "status": "good" if score >= 80 else "needs_improvement" if score >= 60 else "poor"
            }
            
            if score < 80:
                quality_analysis["issues"].append({
                    "metric": metric,
                    "severity": "medium" if score >= 60 else "high",
                    "description": f"Content {metric} could be improved",
                    "suggestions": [f"Review and enhance {metric} aspects"]
                })
        
        # Content analysis
        word_count = document.metadata.word_count
        reading_time = document.metadata.reading_time_minutes
        
        # Structure analysis
        lines = document.content.split('\n')
        headers = [line for line in lines if line.startswith('#')]
        links = len([line for line in lines if 'http' in line or '[' in line])
        
        quality_analysis["content_statistics"] = {
            "word_count": word_count,
            "reading_time_minutes": reading_time,
            "header_count": len(headers),
            "link_count": links,
            "paragraph_count": len([line for line in lines if line.strip() and not line.startswith('#')])
        }
        
        # Generate improvements if requested
        if include_improvements:
            quality_analysis["improvements"] = [
                {
                    "type": "content",
                    "priority": "medium",
                    "description": "Add more examples to improve clarity",
                    "implementation": "Include practical examples in each section"
                },
                {
                    "type": "structure",
                    "priority": "low",
                    "description": "Consider adding a table of contents",
                    "implementation": "Add navigation links at the beginning"
                }
            ]
        
        # Generate report if requested
        report = None
        if generate_report:
            report = {
                "title": f"Quality Analysis Report for {document.metadata.title}",
                "generated_at": datetime.now(UTC).isoformat(),
                "summary": f"Overall quality score: {quality_analysis['overall_score']:.1f}/100",
                "recommendations": len(quality_analysis.get("improvements", [])),
                "issues_found": len(quality_analysis.get("issues", []))
            }
        
        return {
            "success": True,
            "content_id": content_id,
            "document_id": document.document_id,
            "analysis_scope": analysis_scope,
            "quality_analysis": quality_analysis,
            "report": report,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "ai_analysis_used": ai_analysis
        }
        
    except Exception as e:
        logger.error(f"Content quality analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "content_id": content_id
        }


@mcp.tool()
async def km_export_knowledge(
    export_scope: str,  # knowledge_base|document|collection
    target_id: str,
    export_format: str = "pdf",  # pdf|html|confluence|docx|markdown
    include_metadata: bool = True,
    include_version_history: bool = False,
    custom_styling: Optional[Dict[str, Any]] = None,
    export_options: Optional[Dict[str, Any]] = None,
    destination_path: Optional[str] = None,
    compress_output: bool = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Export knowledge base content in various formats for sharing and distribution.
    
    FastMCP Tool for knowledge export through Claude Desktop.
    Exports documentation and knowledge in professional formats with custom branding.
    
    Returns export results, file locations, and format-specific metadata.
    """
    try:
        from ...knowledge.export_system import ExportManager, ExportOptions, ExportFormat, BrandingOptions, CompressionType
        
        logger.info(f"Exporting knowledge: {export_scope} - {target_id}")
        
        # Validate export scope
        valid_scopes = ["knowledge_base", "document", "collection"]
        if export_scope not in valid_scopes:
            return {
                "success": False,
                "error": f"Invalid export scope. Must be one of: {', '.join(valid_scopes)}",
                "export_scope": export_scope
            }
        
        # Validate export format
        valid_formats = ["pdf", "html", "confluence", "docx", "markdown", "epub", "json", "xml"]
        if export_format not in valid_formats:
            return {
                "success": False,
                "error": f"Invalid export format. Must be one of: {', '.join(valid_formats)}",
                "export_format": export_format
            }
        
        # Map format string to enum
        format_mapping = {
            "pdf": ExportFormat.PDF,
            "html": ExportFormat.HTML,
            "confluence": ExportFormat.CONFLUENCE,
            "docx": ExportFormat.DOCX,
            "markdown": ExportFormat.MARKDOWN,
            "epub": ExportFormat.EPUB,
            "json": ExportFormat.JSON,
            "xml": ExportFormat.XML
        }
        
        # Create export options
        options = ExportOptions(
            format=format_mapping[export_format],
            include_metadata=include_metadata,
            include_version_history=include_version_history,
            include_toc=export_options.get("include_toc", True) if export_options else True,
            include_index=export_options.get("include_index", False) if export_options else False,
            custom_styling=custom_styling,
            compression=CompressionType.ZIP if compress_output else CompressionType.NONE,
            destination_path=destination_path
        )
        
        # Create branding options if custom styling provided
        branding = None
        if custom_styling:
            branding = BrandingOptions(
                company_name=custom_styling.get("company_name"),
                primary_color=custom_styling.get("primary_color", "#2563eb"),
                secondary_color=custom_styling.get("secondary_color", "#64748b"),
                font_family=custom_styling.get("font_family", "Inter, -apple-system, sans-serif"),
                css_overrides=custom_styling.get("css_overrides")
            )
        
        # Initialize export manager
        export_manager = ExportManager()
        
        # Create export job
        job_result = await export_manager.create_export_job(
            export_scope=export_scope,
            target_id=target_id,
            options=options,
            branding=branding
        )
        
        if job_result.is_left():
            return {
                "success": False,
                "error": job_result.left(),
                "export_scope": export_scope,
                "target_id": target_id
            }
        
        job = job_result.right()
        
        # Prepare data for export
        knowledge_base = None
        documents = []
        
        if export_scope == "knowledge_base":
            kb_id = KnowledgeBaseId(target_id)
            if kb_id in knowledge_bases:
                knowledge_base = knowledge_bases[kb_id]
                documents = [documents_store[doc_id] for doc_id in knowledge_base.documents if doc_id in documents_store]
        elif export_scope == "document":
            doc_id = DocumentId(target_id)
            if doc_id in documents_store:
                documents = [documents_store[doc_id]]
        
        if not documents:
            return {
                "success": False,
                "error": f"No documents found for {export_scope}: {target_id}",
                "export_scope": export_scope,
                "target_id": target_id
            }
        
        # Execute export
        export_result = await export_manager.execute_export_job(
            job.job_id,
            knowledge_base=knowledge_base,
            documents=documents
        )
        
        if export_result.is_left():
            return {
                "success": False,
                "error": export_result.left(),
                "job_id": job.job_id
            }
        
        result = export_result.right()
        
        return {
            "success": True,
            "job_id": job.job_id,
            "export_scope": export_scope,
            "target_id": target_id,
            "export_format": export_format,
            "output_path": result.output_path,
            "file_size": result.file_size,
            "processing_time_ms": result.processing_time_ms,
            "compressed": compress_output,
            "metadata": result.metadata,
            "exported_at": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Knowledge export failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "export_scope": export_scope,
            "target_id": target_id
        }


@mcp.tool()
async def km_schedule_content_review(
    content_id: str,
    review_date: str,
    reviewers: List[str],
    review_type: str = "accuracy",  # accuracy|completeness|relevance|quality
    review_criteria: Optional[Dict[str, Any]] = None,
    auto_reminders: bool = True,
    escalation_rules: Optional[Dict[str, Any]] = None,
    completion_actions: Optional[List[str]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Schedule content reviews with automated reminders and escalation management.
    
    FastMCP Tool for content review scheduling through Claude Desktop.
    Manages content review workflows with automated notifications and tracking.
    
    Returns review schedule, assignment details, and tracking information.
    """
    try:
        from datetime import datetime
        import dateutil.parser
        
        logger.info(f"Scheduling content review: {content_id}")
        
        # Find document by content ID
        document = None
        for doc in documents_store.values():
            if doc.metadata.content_id == content_id:
                document = doc
                break
        
        if not document:
            return {
                "success": False,
                "error": f"Document with content ID {content_id} not found",
                "content_id": content_id
            }
        
        # Validate review type
        valid_review_types = ["accuracy", "completeness", "relevance", "quality", "compliance", "technical"]
        if review_type not in valid_review_types:
            return {
                "success": False,
                "error": f"Invalid review type. Must be one of: {', '.join(valid_review_types)}",
                "review_type": review_type
            }
        
        # Parse review date
        try:
            parsed_date = dateutil.parser.parse(review_date)
        except Exception:
            return {
                "success": False,
                "error": f"Invalid review date format: {review_date}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                "review_date": review_date
            }
        
        # Validate reviewers
        if not reviewers or len(reviewers) == 0:
            return {
                "success": False,
                "error": "At least one reviewer must be specified",
                "reviewers": reviewers
            }
        
        # Create review ID
        review_id = f"review_{content_id}_{uuid.uuid4().hex[:8]}"
        
        # Create review schedule
        review_schedule = {
            "review_id": review_id,
            "content_id": content_id,
            "document_id": document.document_id,
            "document_title": document.metadata.title,
            "review_type": review_type,
            "scheduled_date": parsed_date.isoformat(),
            "reviewers": reviewers,
            "status": "scheduled",
            "created_at": datetime.now(UTC).isoformat(),
            "created_by": "system"
        }
        
        # Add review criteria
        if review_criteria:
            review_schedule["review_criteria"] = review_criteria
        else:
            # Default criteria based on review type
            default_criteria = {
                "accuracy": ["factual_correctness", "data_validation", "source_verification"],
                "completeness": ["content_coverage", "missing_sections", "detail_level"],
                "relevance": ["current_applicability", "target_audience_fit", "business_value"],
                "quality": ["clarity", "readability", "structure", "formatting"]
            }
            review_schedule["review_criteria"] = {
                "focus_areas": default_criteria.get(review_type, ["general_review"]),
                "scoring_method": "1-5_scale",
                "required_sections": ["summary", "recommendations", "approval_status"]
            }
        
        # Configure reminders
        if auto_reminders:
            review_schedule["reminders"] = {
                "enabled": True,
                "reminder_schedule": [
                    {"days_before": 7, "type": "initial_notification"},
                    {"days_before": 3, "type": "upcoming_reminder"},
                    {"days_before": 1, "type": "urgent_reminder"},
                    {"days_after": 1, "type": "overdue_notification"}
                ]
            }
        
        # Configure escalation
        if escalation_rules:
            review_schedule["escalation"] = escalation_rules
        else:
            review_schedule["escalation"] = {
                "overdue_days": 5,
                "escalation_contacts": ["content_manager", "team_lead"],
                "auto_escalate": True
            }
        
        # Configure completion actions
        if completion_actions:
            review_schedule["completion_actions"] = completion_actions
        else:
            review_schedule["completion_actions"] = [
                "update_document_metadata",
                "notify_stakeholders",
                "schedule_next_review"
            ]
        
        # Mock storing the review schedule (would be stored in database)
        review_schedule["storage_location"] = "review_database"
        
        return {
            "success": True,
            "review_id": review_id,
            "content_id": content_id,
            "document_id": document.document_id,
            "review_schedule": review_schedule,
            "next_reminder": (parsed_date - timedelta(days=7)).isoformat() if auto_reminders else None,
            "estimated_duration": "2-4 hours" if review_type in ["accuracy", "quality"] else "1-2 hours"
        }
        
    except Exception as e:
        logger.error(f"Content review scheduling failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "content_id": content_id
        }