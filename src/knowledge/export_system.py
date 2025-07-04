"""
Export System for Knowledge Management.

This module provides comprehensive export capabilities for knowledge base content,
supporting multiple formats with custom styling and branding options.
"""

from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio
import json
import logging
import base64
import zipfile
import io
import tempfile

from src.core.knowledge_architecture import (
    DocumentId, KnowledgeBaseId, ContentFormat, KnowledgeDocument, KnowledgeBase,
    KnowledgeError
)
from src.core.contracts import require, ensure
from ..core.either import Either

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format options."""
    PDF = "pdf"
    HTML = "html"
    CONFLUENCE = "confluence"
    DOCX = "docx"
    MARKDOWN = "markdown"
    EPUB = "epub"
    JSON = "json"
    XML = "xml"


class CompressionType(Enum):
    """Compression options."""
    NONE = "none"
    ZIP = "zip"
    GZIP = "gzip"
    TAR = "tar"


@dataclass
class ExportOptions:
    """Export configuration options."""
    format: ExportFormat
    include_metadata: bool = True
    include_version_history: bool = False
    include_toc: bool = True
    include_index: bool = False
    custom_styling: Optional[Dict[str, Any]] = None
    compression: CompressionType = CompressionType.NONE
    destination_path: Optional[str] = None
    filename_template: str = "{title}_{timestamp}"
    
    def get_filename(self, title: str, timestamp: Optional[datetime] = None) -> str:
        """Generate filename from template."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        return self.filename_template.format(
            title=title.replace(" ", "_"),
            timestamp=timestamp.strftime("%Y%m%d_%H%M%S"),
            format=self.format.value
        )


@dataclass
class BrandingOptions:
    """Branding and styling options."""
    logo_url: Optional[str] = None
    logo_base64: Optional[str] = None
    company_name: Optional[str] = None
    primary_color: str = "#2563eb"
    secondary_color: str = "#64748b"
    font_family: str = "Inter, -apple-system, sans-serif"
    header_template: Optional[str] = None
    footer_template: Optional[str] = None
    css_overrides: Optional[str] = None
    
    def get_css_variables(self) -> str:
        """Get CSS variables for styling."""
        return f"""
        :root {{
            --primary-color: {self.primary_color};
            --secondary-color: {self.secondary_color};
            --font-family: {self.font_family};
        }}
        """


@dataclass
class ExportJob:
    """Export job tracking."""
    job_id: str
    export_scope: str  # knowledge_base, document, collection
    target_id: str
    options: ExportOptions
    branding: Optional[BrandingOptions] = None
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0
    
    def update_progress(self, progress: float, status: Optional[str] = None) -> None:
        """Update job progress."""
        self.progress = min(100.0, max(0.0, progress))
        if status:
            self.status = status
        
        if status == "processing" and not self.started_at:
            self.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            self.completed_at = datetime.utcnow()


@dataclass
class ExportResult:
    """Export operation result."""
    job_id: str
    success: bool
    output_path: Optional[str] = None
    output_format: Optional[ExportFormat] = None
    file_size: int = 0
    processing_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class MarkdownExporter:
    """Markdown format exporter."""
    
    def __init__(self, branding: Optional[BrandingOptions] = None):
        self.branding = branding
    
    async def export_document(self, document: KnowledgeDocument, 
                            options: ExportOptions) -> str:
        """Export single document to markdown."""
        content_parts = []
        
        # Add title
        content_parts.append(f"# {document.metadata.title}")
        content_parts.append("")
        
        # Add metadata if requested
        if options.include_metadata:
            content_parts.append("## Document Information")
            content_parts.append(f"- **Created**: {document.metadata.created_at.isoformat()}")
            content_parts.append(f"- **Modified**: {document.metadata.modified_at.isoformat()}")
            content_parts.append(f"- **Author**: {document.metadata.author}")
            content_parts.append(f"- **Version**: {document.metadata.version}")
            if document.metadata.description:
                content_parts.append(f"- **Description**: {document.metadata.description}")
            if document.metadata.tags:
                content_parts.append(f"- **Tags**: {', '.join(document.metadata.tags)}")
            content_parts.append("")
        
        # Add main content
        content_parts.append(document.content)
        
        return "\n".join(content_parts)
    
    async def export_knowledge_base(self, knowledge_base: KnowledgeBase,
                                  documents: List[KnowledgeDocument],
                                  options: ExportOptions) -> str:
        """Export knowledge base to markdown."""
        content_parts = []
        
        # Add title page
        content_parts.append(f"# {knowledge_base.name}")
        content_parts.append("")
        if knowledge_base.description:
            content_parts.append(knowledge_base.description)
            content_parts.append("")
        
        # Add table of contents if requested
        if options.include_toc:
            content_parts.append("## Table of Contents")
            for i, doc in enumerate(documents, 1):
                content_parts.append(f"{i}. [{doc.metadata.title}](#{self._anchor_link(doc.metadata.title)})")
            content_parts.append("")
        
        # Add documents
        for doc in documents:
            doc_content = await self.export_document(doc, options)
            content_parts.append(doc_content)
            content_parts.append("\n---\n")
        
        return "\n".join(content_parts)
    
    def _anchor_link(self, title: str) -> str:
        """Generate anchor link from title."""
        return title.lower().replace(" ", "-").replace("'", "")


class HTMLExporter:
    """HTML format exporter."""
    
    def __init__(self, branding: Optional[BrandingOptions] = None):
        self.branding = branding
    
    async def export_document(self, document: KnowledgeDocument,
                            options: ExportOptions) -> str:
        """Export single document to HTML."""
        # Convert markdown to HTML (simplified)
        content = self._markdown_to_html(document.content)
        
        # Build complete HTML document
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{document.metadata.title}</title>",
            self._get_styles(),
            "</head>",
            "<body>",
            self._get_header(document.metadata.title),
            "<main class='content'>",
            f"<h1>{document.metadata.title}</h1>"
        ]
        
        # Add metadata if requested
        if options.include_metadata:
            html_parts.extend(self._generate_metadata_html(document))
        
        # Add content
        html_parts.append(content)
        
        # Close HTML
        html_parts.extend([
            "</main>",
            self._get_footer(),
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    async def export_knowledge_base(self, knowledge_base: KnowledgeBase,
                                  documents: List[KnowledgeDocument],
                                  options: ExportOptions) -> str:
        """Export knowledge base to HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{knowledge_base.name}</title>",
            self._get_styles(),
            "</head>",
            "<body>",
            self._get_header(knowledge_base.name)
        ]
        
        # Add navigation if TOC requested
        if options.include_toc:
            html_parts.extend(self._generate_navigation(documents))
        
        html_parts.append("<main class='content'>")
        html_parts.append(f"<h1>{knowledge_base.name}</h1>")
        
        if knowledge_base.description:
            html_parts.append(f"<p class='description'>{knowledge_base.description}</p>")
        
        # Add documents
        for doc in documents:
            html_parts.append(f"<section id='{self._anchor_link(doc.metadata.title)}'>")
            doc_content = self._markdown_to_html(doc.content)
            html_parts.append(doc_content)
            html_parts.append("</section>")
        
        html_parts.extend([
            "</main>",
            self._get_footer(),
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (simplified)."""
        html = markdown
        
        # Headers
        html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Code blocks
        html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
        
        # Lists
        html = re.sub(r'^- (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
        
        # Paragraphs
        paragraphs = html.split('\n\n')
        html_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                html_paragraphs.append(f'<p>{p}</p>')
            else:
                html_paragraphs.append(p)
        
        return '\n'.join(html_paragraphs)
    
    def _get_styles(self) -> str:
        """Get CSS styles."""
        base_styles = """
        <style>
        body {
            font-family: var(--font-family, 'Inter, -apple-system, sans-serif');
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            border-bottom: 2px solid var(--primary-color, #2563eb);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .content {
            margin: 20px 0;
        }
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .navigation {
            background: #f1f5f9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .footer {
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
            margin-top: 40px;
            text-align: center;
            color: #64748b;
        }
        h1, h2, h3 { color: var(--primary-color, #2563eb); }
        code { background: #f1f5f9; padding: 2px 4px; border-radius: 3px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
        """
        
        if self.branding:
            base_styles += f"<style>{self.branding.get_css_variables()}</style>"
            if self.branding.css_overrides:
                base_styles += f"<style>{self.branding.css_overrides}</style>"
        
        return base_styles
    
    def _get_header(self, title: str) -> str:
        """Get HTML header."""
        if self.branding and self.branding.header_template:
            return self.branding.header_template.format(title=title)
        
        header_parts = ["<header class='header'>"]
        
        if self.branding and (self.branding.logo_url or self.branding.logo_base64):
            if self.branding.logo_url:
                header_parts.append(f"<img src='{self.branding.logo_url}' alt='Logo' style='height: 40px;'>")
            elif self.branding.logo_base64:
                header_parts.append(f"<img src='data:image/png;base64,{self.branding.logo_base64}' alt='Logo' style='height: 40px;'>")
        
        if self.branding and self.branding.company_name:
            header_parts.append(f"<h1>{self.branding.company_name}</h1>")
        
        header_parts.append("</header>")
        return "\n".join(header_parts)
    
    def _get_footer(self) -> str:
        """Get HTML footer."""
        if self.branding and self.branding.footer_template:
            return self.branding.footer_template
        
        footer_content = f"<footer class='footer'><p>Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p></footer>"
        
        if self.branding and self.branding.company_name:
            footer_content = footer_content.replace("</p>", f" by {self.branding.company_name}</p>")
        
        return footer_content
    
    def _generate_metadata_html(self, document: KnowledgeDocument) -> List[str]:
        """Generate metadata HTML."""
        metadata_parts = ["<div class='metadata'>", "<h3>Document Information</h3>"]
        
        metadata_parts.append(f"<p><strong>Created:</strong> {document.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        metadata_parts.append(f"<p><strong>Modified:</strong> {document.metadata.modified_at.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        metadata_parts.append(f"<p><strong>Author:</strong> {document.metadata.author}</p>")
        metadata_parts.append(f"<p><strong>Version:</strong> {document.metadata.version}</p>")
        
        if document.metadata.description:
            metadata_parts.append(f"<p><strong>Description:</strong> {document.metadata.description}</p>")
        
        if document.metadata.tags:
            tags_html = ", ".join([f"<span class='tag'>{tag}</span>" for tag in document.metadata.tags])
            metadata_parts.append(f"<p><strong>Tags:</strong> {tags_html}</p>")
        
        metadata_parts.append("</div>")
        return metadata_parts
    
    def _generate_navigation(self, documents: List[KnowledgeDocument]) -> List[str]:
        """Generate navigation HTML."""
        nav_parts = ["<nav class='navigation'>", "<h3>Table of Contents</h3>", "<ul>"]
        
        for doc in documents:
            anchor = self._anchor_link(doc.metadata.title)
            nav_parts.append(f"<li><a href='#{anchor}'>{doc.metadata.title}</a></li>")
        
        nav_parts.extend(["</ul>", "</nav>"])
        return nav_parts
    
    def _anchor_link(self, title: str) -> str:
        """Generate anchor link from title."""
        return title.lower().replace(" ", "-").replace("'", "")


class ExportManager:
    """Export management system."""
    
    def __init__(self):
        self.jobs: Dict[str, ExportJob] = {}
        self.exporters = {
            ExportFormat.MARKDOWN: MarkdownExporter,
            ExportFormat.HTML: HTMLExporter
        }
        self.job_counter = 0
        
        logger.info("ExportManager initialized")
    
    @require(lambda export_scope, target_id: export_scope and target_id, "Export scope and target required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns job or error")
    async def create_export_job(self, 
                              export_scope: str,
                              target_id: str,
                              options: ExportOptions,
                              branding: Optional[BrandingOptions] = None) -> Either[str, ExportJob]:
        """Create new export job."""
        try:
            self.job_counter += 1
            job_id = f"export_job_{self.job_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            job = ExportJob(
                job_id=job_id,
                export_scope=export_scope,
                target_id=target_id,
                options=options,
                branding=branding
            )
            
            self.jobs[job_id] = job
            
            logger.info(f"Created export job: {job_id}")
            return Either.right(job)
            
        except Exception as e:
            error_msg = f"Failed to create export job: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def execute_export_job(self, job_id: str,
                               knowledge_base: Optional[KnowledgeBase] = None,
                               documents: Optional[List[KnowledgeDocument]] = None) -> Either[str, ExportResult]:
        """Execute export job."""
        try:
            if job_id not in self.jobs:
                return Either.left(f"Export job {job_id} not found")
            
            job = self.jobs[job_id]
            start_time = datetime.utcnow()
            
            job.update_progress(0, "processing")
            
            # Get exporter for format
            exporter_class = self.exporters.get(job.options.format)
            if not exporter_class:
                job.update_progress(0, "failed")
                job.error_message = f"Unsupported export format: {job.options.format.value}"
                return Either.left(job.error_message)
            
            exporter = exporter_class(job.branding)
            job.update_progress(10, "processing")
            
            # Export content based on scope
            content = ""
            if job.export_scope == "knowledge_base" and knowledge_base and documents:
                content = await exporter.export_knowledge_base(knowledge_base, documents, job.options)
                job.update_progress(80, "processing")
            elif job.export_scope == "document" and documents and len(documents) > 0:
                content = await exporter.export_document(documents[0], job.options)
                job.update_progress(80, "processing")
            else:
                error_msg = f"Invalid export scope or missing data: {job.export_scope}"
                job.update_progress(0, "failed")
                job.error_message = error_msg
                return Either.left(error_msg)
            
            # Save content to file
            output_path = await self._save_export_content(job, content)
            job.output_path = output_path
            job.file_size = len(content.encode('utf-8'))
            job.update_progress(100, "completed")
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = ExportResult(
                job_id=job_id,
                success=True,
                output_path=output_path,
                output_format=job.options.format,
                file_size=job.file_size,
                processing_time_ms=int(processing_time),
                metadata={
                    "export_scope": job.export_scope,
                    "target_id": job.target_id,
                    "created_at": job.created_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None
                }
            )
            
            logger.info(f"Export job completed: {job_id}")
            return Either.right(result)
            
        except Exception as e:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.update_progress(job.progress, "failed")
                job.error_message = str(e)
            
            error_msg = f"Export job execution failed: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def get_job_status(self, job_id: str) -> Either[str, ExportJob]:
        """Get export job status."""
        try:
            if job_id not in self.jobs:
                return Either.left(f"Export job {job_id} not found")
            
            return Either.right(self.jobs[job_id])
            
        except Exception as e:
            error_msg = f"Failed to get job status: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def cancel_job(self, job_id: str) -> Either[str, bool]:
        """Cancel export job."""
        try:
            if job_id not in self.jobs:
                return Either.left(f"Export job {job_id} not found")
            
            job = self.jobs[job_id]
            if job.status in ["completed", "failed"]:
                return Either.left(f"Cannot cancel job in status: {job.status}")
            
            job.update_progress(job.progress, "cancelled")
            
            logger.info(f"Cancelled export job: {job_id}")
            return Either.right(True)
            
        except Exception as e:
            error_msg = f"Failed to cancel job: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def _save_export_content(self, job: ExportJob, content: str) -> str:
        """Save export content to file."""
        try:
            # Generate filename
            title = job.target_id  # Simplified - would get actual title
            filename = job.options.get_filename(title)
            
            # Add file extension
            filename += f".{job.options.format.value}"
            
            # Determine output path
            if job.options.destination_path:
                output_dir = Path(job.options.destination_path)
            else:
                output_dir = Path(tempfile.gettempdir()) / "km_exports"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            
            # Save content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Apply compression if requested
            if job.options.compression != CompressionType.NONE:
                compressed_path = await self._compress_file(output_path, job.options.compression)
                output_path.unlink()  # Remove uncompressed file
                return str(compressed_path)
            
            return str(output_path)
            
        except Exception as e:
            raise KnowledgeError(f"Failed to save export content: {str(e)}")
    
    async def _compress_file(self, file_path: Path, compression: CompressionType) -> Path:
        """Compress exported file."""
        try:
            if compression == CompressionType.ZIP:
                zip_path = file_path.with_suffix(file_path.suffix + '.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, file_path.name)
                return zip_path
            
            # Other compression types could be added here
            return file_path
            
        except Exception as e:
            raise KnowledgeError(f"Failed to compress file: {str(e)}")
    
    async def list_jobs(self, status_filter: Optional[str] = None) -> List[ExportJob]:
        """List export jobs with optional status filter."""
        try:
            jobs = list(self.jobs.values())
            
            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]
            
            # Sort by creation time (most recent first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            return []
    
    async def cleanup_completed_jobs(self, max_age_days: int = 7) -> int:
        """Clean up old completed jobs."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            jobs_to_remove = []
            
            for job_id, job in self.jobs.items():
                if job.status in ["completed", "failed", "cancelled"]:
                    if job.completed_at and job.completed_at < cutoff_date:
                        jobs_to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
            
            logger.info(f"Cleaned up {len(jobs_to_remove)} old export jobs")
            return len(jobs_to_remove)
            
        except Exception as e:
            logger.error(f"Failed to cleanup jobs: {str(e)}")
            return 0


# Add missing imports
import re
from datetime import timedelta