"""
Clipboard Management MCP Tools

Provides comprehensive clipboard operations for Keyboard Maestro MCP including
current content access, history management, named clipboards, and security validation.
"""

from typing import Dict, Any, Optional, List, Set
from pydantic import Field
from typing_extensions import Annotated
import asyncio

from fastmcp import Context
from ...clipboard import ClipboardManager, NamedClipboardManager, ClipboardFormat
from ...integration.km_client import KMError


# Global instances for clipboard management
_clipboard_manager: Optional[ClipboardManager] = None
_named_clipboard_manager: Optional[NamedClipboardManager] = None


def get_clipboard_manager() -> ClipboardManager:
    """Get or create clipboard manager instance."""
    global _clipboard_manager
    if _clipboard_manager is None:
        _clipboard_manager = ClipboardManager()
    return _clipboard_manager


def get_named_clipboard_manager() -> NamedClipboardManager:
    """Get or create named clipboard manager instance."""
    global _named_clipboard_manager
    if _named_clipboard_manager is None:
        _named_clipboard_manager = NamedClipboardManager()
    return _named_clipboard_manager


async def km_clipboard_manager(
    operation: Annotated[str, Field(
        description="Clipboard operation type",
        pattern=r"^(get|set|get_history|list_history|manage_named|search_named|stats)$"
    )],
    clipboard_name: Annotated[Optional[str], Field(
        default=None,
        description="For named clipboards - clipboard name",
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-\s]*$"
    )] = None,
    history_index: Annotated[Optional[int], Field(
        default=None,
        description="0-based history position for get_history",
        ge=0,
        le=199
    )] = None,
    history_count: Annotated[Optional[int], Field(
        default=10,
        description="Number of history items to retrieve",
        ge=1,
        le=50
    )] = 10,
    content: Annotated[Optional[str], Field(
        default=None,
        description="Content to set (for set operation)",
        max_length=1_000_000
    )] = None,
    format_filter: Annotated[str, Field(
        default="text",
        description="Content format filter",
        pattern=r"^(text|image|file|url|all)$"
    )] = "text",
    include_sensitive: Annotated[bool, Field(
        default=False,
        description="Include potentially sensitive content"
    )] = False,
    tags: Annotated[Optional[List[str]], Field(
        default=None,
        description="Tags for named clipboard organization"
    )] = None,
    description: Annotated[Optional[str], Field(
        default=None,
        description="Description for named clipboard",
        max_length=500
    )] = None,
    overwrite: Annotated[bool, Field(
        default=False,
        description="Overwrite existing named clipboard"
    )] = False,
    search_query: Annotated[Optional[str], Field(
        default=None,
        description="Search query for named clipboards",
        max_length=100
    )] = None,
    search_content: Annotated[bool, Field(
        default=False,
        description="Search within clipboard content"
    )] = False,
    sort_by: Annotated[str, Field(
        default="name",
        description="Sort field for listings",
        pattern=r"^(name|created_at|accessed_at|access_count)$"
    )] = "name",
    tag_filter: Annotated[Optional[str], Field(
        default=None,
        description="Filter by tag for listings",
        max_length=50
    )] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Comprehensive clipboard management with security and privacy protection.
    
    Operations:
    - get: Retrieve current clipboard content with format detection
    - set: Set clipboard content with security validation
    - get_history: Access specific clipboard history item by index
    - list_history: Get list of clipboard history items
    - manage_named: Create, get, list, or delete named clipboards
    - search_named: Search named clipboards by name, tags, or content
    - stats: Get statistics about named clipboards
    
    Security Features:
    - Automatic sensitive content detection and filtering
    - Size limits to prevent memory issues
    - Content preview for large items
    - Access logging for audit trails
    
    Returns clipboard operations results with metadata and security status.
    """
    if ctx:
        await ctx.info(f"Performing clipboard operation: {operation}")
    
    try:
        clipboard_manager = get_clipboard_manager()
        named_manager = get_named_clipboard_manager()
        
        # Get current clipboard content
        if operation == "get":
            result = await clipboard_manager.get_clipboard(include_sensitive)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            content_obj = result.get_right()
            return {
                "success": True,
                "data": {
                    "content": content_obj.content if not content_obj.is_sensitive or include_sensitive else "[SENSITIVE CONTENT HIDDEN]",
                    "format": content_obj.format.value,
                    "size_bytes": content_obj.size_bytes,
                    "timestamp": content_obj.timestamp,
                    "is_sensitive": content_obj.is_sensitive,
                    "preview": content_obj.preview(),
                    "is_empty": content_obj.is_empty()
                },
                "metadata": {
                    "operation": "get",
                    "security_filtered": content_obj.is_sensitive and not include_sensitive
                }
            }
        
        # Set clipboard content
        elif operation == "set":
            if not content:
                return {
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Content is required for set operation"
                    }
                }
            
            result = await clipboard_manager.set_clipboard(content)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            return {
                "success": True,
                "data": {
                    "content_size": len(content.encode('utf-8')),
                    "content_preview": content[:50] + "..." if len(content) > 50 else content
                },
                "metadata": {
                    "operation": "set"
                }
            }
        
        # Get specific history item
        elif operation == "get_history":
            if history_index is None:
                return {
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "history_index is required for get_history operation"
                    }
                }
            
            result = await clipboard_manager.get_history_item(history_index, include_sensitive)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            content_obj = result.get_right()
            return {
                "success": True,
                "data": {
                    "index": history_index,
                    "content": content_obj.content if not content_obj.is_sensitive or include_sensitive else "[SENSITIVE CONTENT HIDDEN]",
                    "format": content_obj.format.value,
                    "size_bytes": content_obj.size_bytes,
                    "timestamp": content_obj.timestamp,
                    "is_sensitive": content_obj.is_sensitive,
                    "preview": content_obj.preview()
                },
                "metadata": {
                    "operation": "get_history",
                    "security_filtered": content_obj.is_sensitive and not include_sensitive
                }
            }
        
        # List clipboard history
        elif operation == "list_history":
            result = await clipboard_manager.get_history_list(history_count, include_sensitive)
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            history_items = result.get_right()
            return {
                "success": True,
                "data": {
                    "history": [
                        {
                            "index": i,
                            "content": item.content if not item.is_sensitive or include_sensitive else "[SENSITIVE CONTENT HIDDEN]",
                            "format": item.format.value,
                            "size_bytes": item.size_bytes,
                            "timestamp": item.timestamp,
                            "is_sensitive": item.is_sensitive,
                            "preview": item.preview()
                        }
                        for i, item in enumerate(history_items)
                    ],
                    "total_items": len(history_items)
                },
                "metadata": {
                    "operation": "list_history",
                    "requested_count": history_count,
                    "security_filtering": not include_sensitive
                }
            }
        
        # Manage named clipboards
        elif operation == "manage_named":
            if not clipboard_name:
                return {
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "clipboard_name is required for manage_named operation"
                    }
                }
            
            # Create or update named clipboard
            if content is not None:
                # Get current clipboard content to create ClipboardContent object
                current_result = await clipboard_manager.get_clipboard(True)  # Include sensitive for internal operations
                if current_result.is_left():
                    # Use provided content instead
                    from ...clipboard.clipboard_manager import ClipboardContent
                    import time
                    
                    content_obj = ClipboardContent(
                        content=content,
                        format=ClipboardFormat.TEXT,
                        size_bytes=len(content.encode('utf-8')),
                        timestamp=time.time()
                    )
                else:
                    content_obj = current_result.get_right()
                
                # Convert tags list to set
                tag_set = set(tags) if tags else set()
                
                result = await named_manager.create_named_clipboard(
                    clipboard_name, content_obj, tag_set, description, overwrite
                )
                
                if result.is_left():
                    error = result.get_left()
                    return {
                        "success": False,
                        "error": {
                            "code": error.code,
                            "message": error.message,
                            "details": error.details
                        }
                    }
                
                return {
                    "success": True,
                    "data": {
                        "name": clipboard_name,
                        "created": True,
                        "overwritten": overwrite
                    },
                    "metadata": {
                        "operation": "create_named"
                    }
                }
            
            # Get named clipboard
            else:
                result = await named_manager.get_named_clipboard(clipboard_name)
                if result.is_left():
                    error = result.get_left()
                    return {
                        "success": False,
                        "error": {
                            "code": error.code,
                            "message": error.message,
                            "details": error.details
                        }
                    }
                
                named_cb = result.get_right()
                return {
                    "success": True,
                    "data": {
                        "name": named_cb.name,
                        "content": named_cb.content.content if not named_cb.content.is_sensitive or include_sensitive else "[SENSITIVE CONTENT HIDDEN]",
                        "format": named_cb.content.format.value,
                        "size_bytes": named_cb.content.size_bytes,
                        "created_at": named_cb.created_at,
                        "accessed_at": named_cb.accessed_at,
                        "access_count": named_cb.access_count,
                        "tags": list(named_cb.tags),
                        "description": named_cb.description,
                        "preview": named_cb.content.preview()
                    },
                    "metadata": {
                        "operation": "get_named",
                        "security_filtered": named_cb.content.is_sensitive and not include_sensitive
                    }
                }
        
        # Search named clipboards
        elif operation == "search_named":
            if not search_query:
                # List all named clipboards
                result = await named_manager.list_named_clipboards(tag_filter, sort_by)
            else:
                result = await named_manager.search_named_clipboards(search_query, search_content, 50)
            
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            clipboards = result.get_right()
            return {
                "success": True,
                "data": {
                    "clipboards": [
                        {
                            "name": cb.name,
                            "format": cb.content.format.value,
                            "size_bytes": cb.content.size_bytes,
                            "created_at": cb.created_at,
                            "accessed_at": cb.accessed_at,
                            "access_count": cb.access_count,
                            "tags": list(cb.tags),
                            "description": cb.description,
                            "preview": cb.content.preview(),
                            "is_sensitive": cb.content.is_sensitive
                        }
                        for cb in clipboards
                    ],
                    "total_found": len(clipboards)
                },
                "metadata": {
                    "operation": "search_named",
                    "query": search_query,
                    "searched_content": search_content,
                    "tag_filter": tag_filter,
                    "sort_by": sort_by
                }
            }
        
        # Get clipboard statistics
        elif operation == "stats":
            result = await named_manager.get_clipboard_stats()
            if result.is_left():
                error = result.get_left()
                return {
                    "success": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details
                    }
                }
            
            stats = result.get_right()
            return {
                "success": True,
                "data": stats,
                "metadata": {
                    "operation": "stats"
                }
            }
        
        else:
            return {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": f"Unsupported operation: {operation}"
                }
            }
    
    except Exception as e:
        if ctx:
            await ctx.error(f"Clipboard operation failed: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "EXECUTION_ERROR",
                "message": f"Clipboard operation failed: {str(e)}",
                "details": {"exception_type": type(e).__name__}
            }
        }