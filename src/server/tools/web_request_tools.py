"""
Web request tools for HTTP/REST API integration.

This module implements the km_web_request MCP tool, enabling AI to make HTTP requests
to APIs, webhooks, and web services. Includes comprehensive security validation,
authentication support, and response processing.

Security: URL validation, SSRF protection, credential sanitization
Performance: Connection pooling, timeout management, response streaming
Integration: Token processor integration for dynamic URL construction
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from urllib.parse import urljoin

from fastmcp import Context
from fastmcp.exceptions import ToolError

from src.core.http_client import HTTPClient, HTTPRequest, HTTPMethod, HTTPResponse
from src.web.authentication import (
    AuthenticationManager, AuthenticationCredentials, 
    create_api_key_auth, create_bearer_token_auth, create_basic_auth
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError, MCPError
from src.core.contracts import require, ensure
from src.tokens.token_processor import TokenProcessor

# Setup module logger
logger = logging.getLogger(__name__)


class WebRequestProcessor:
    """Process web requests with comprehensive validation and security."""
    
    def __init__(self, allow_localhost: bool = False):
        self.allow_localhost = allow_localhost
        self.token_processor = TokenProcessor()
    
    @require(lambda self, url: isinstance(url, str) and len(url.strip()) > 0)
    @require(lambda self, method: isinstance(method, str))
    async def process_web_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, Dict[str, Any]]] = None,
        params: Optional[Dict[str, str]] = None,
        auth_type: str = "none",
        auth_credentials: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 30,
        follow_redirects: bool = True,
        verify_ssl: bool = True,
        max_response_size: int = 10485760,
        response_format: str = "auto",
        save_response_to: Optional[str] = None,
        ctx: Optional[Context] = None
    ) -> Either[MCPError, Dict[str, Any]]:
        """
        Process web request with comprehensive validation and security.
        
        Architecture: HTTP client with security-first design
        Security: URL validation, SSRF protection, credential sanitization
        Performance: Efficient request processing with streaming support
        """
        try:
            if ctx:
                await ctx.info(f"Processing {method.upper()} request to {url}")
            
            # Process URL with token substitution
            processed_url_result = await self._process_url_with_tokens(url, ctx)
            if processed_url_result.is_left():
                return Either.left(processed_url_result.get_left())
            
            processed_url = processed_url_result.get_right()
            
            # Validate and convert HTTP method
            method_result = self._validate_http_method(method)
            if method_result.is_left():
                return Either.left(method_result.get_left())
            
            http_method = method_result.get_right()
            
            # Process headers with token substitution
            processed_headers_result = await self._process_headers_with_tokens(
                headers or {}, ctx
            )
            if processed_headers_result.is_left():
                return Either.left(processed_headers_result.get_left())
            
            processed_headers = processed_headers_result.get_right()
            
            # Process parameters with token substitution
            processed_params_result = await self._process_params_with_tokens(
                params or {}, ctx
            )
            if processed_params_result.is_left():
                return Either.left(processed_params_result.get_left())
            
            processed_params = processed_params_result.get_right()
            
            # Process request data
            processed_data_result = await self._process_request_data(data, ctx)
            if processed_data_result.is_left():
                return Either.left(processed_data_result.get_left())
            
            processed_data = processed_data_result.get_right()
            
            # Create HTTP request
            http_request = HTTPRequest(
                url=processed_url,
                method=http_method,
                headers=processed_headers,
                data=processed_data,
                params=processed_params,
                timeout_seconds=timeout_seconds,
                verify_ssl=verify_ssl,
                follow_redirects=follow_redirects,
                max_response_size=max_response_size
            )
            
            # Process authentication
            auth_headers_result = await self._process_authentication(
                auth_type, auth_credentials, ctx
            )
            if auth_headers_result.is_left():
                return Either.left(auth_headers_result.get_left())
            
            auth_headers = auth_headers_result.get_right()
            
            # Execute request
            if ctx:
                await ctx.report_progress(50, 100, "Executing HTTP request")
            
            async with HTTPClient(allow_localhost=self.allow_localhost) as client:
                response_result = await client.execute_request(http_request, auth_headers)
                
                if response_result.is_left():
                    return Either.left(response_result.get_left())
                
                response = response_result.get_right()
            
            # Process response
            if ctx:
                await ctx.report_progress(75, 100, "Processing response")
            
            processed_response_result = await self._process_response(
                response, response_format, save_response_to, ctx
            )
            
            if processed_response_result.is_left():
                return Either.left(processed_response_result.get_left())
            
            final_response = processed_response_result.get_right()
            
            if ctx:
                await ctx.report_progress(100, 100, "Request completed successfully")
                await ctx.info(f"Request completed: {response.status_code} in {response.duration_ms:.1f}ms")
            
            return Either.right(final_response)
            
        except Exception as e:
            error_msg = f"Failed to process web request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if ctx:
                await ctx.error(error_msg)
            return Either.left(MCPError("WEB_REQUEST_ERROR", error_msg))
    
    async def _process_url_with_tokens(
        self, 
        url: str, 
        ctx: Optional[Context]
    ) -> Either[MCPError, str]:
        """Process URL with token substitution."""
        try:
            # Process tokens in URL
            processed_url = await self.token_processor.process_tokens_in_text(url)
            
            if ctx:
                if processed_url != url:
                    await ctx.info("Applied token substitution to URL")
            
            return Either.right(processed_url)
            
        except Exception as e:
            return Either.left(MCPError(
                "URL_PROCESSING_ERROR",
                f"Failed to process URL tokens: {str(e)}"
            ))
    
    def _validate_http_method(self, method: str) -> Either[MCPError, HTTPMethod]:
        """Validate and convert HTTP method."""
        try:
            method_upper = method.strip().upper()
            try:
                return Either.right(HTTPMethod(method_upper))
            except ValueError:
                valid_methods = [m.value for m in HTTPMethod]
                return Either.left(MCPError(
                    "INVALID_HTTP_METHOD",
                    f"Invalid HTTP method '{method}'. Valid methods: {', '.join(valid_methods)}"
                ))
                
        except Exception as e:
            return Either.left(MCPError(
                "METHOD_VALIDATION_ERROR",
                f"Failed to validate HTTP method: {str(e)}"
            ))
    
    async def _process_headers_with_tokens(
        self,
        headers: Dict[str, str],
        ctx: Optional[Context]
    ) -> Either[MCPError, Dict[str, str]]:
        """Process headers with token substitution."""
        try:
            processed_headers = {}
            
            for name, value in headers.items():
                # Process tokens in header value
                processed_value = await self.token_processor.process_tokens_in_text(str(value))
                processed_headers[name] = processed_value
            
            if ctx and headers and any(
                processed_headers[k] != headers[k] for k in headers
            ):
                await ctx.info("Applied token substitution to headers")
            
            return Either.right(processed_headers)
            
        except Exception as e:
            return Either.left(MCPError(
                "HEADER_PROCESSING_ERROR",
                f"Failed to process header tokens: {str(e)}"
            ))
    
    async def _process_params_with_tokens(
        self,
        params: Dict[str, str],
        ctx: Optional[Context]
    ) -> Either[MCPError, Dict[str, str]]:
        """Process query parameters with token substitution."""
        try:
            processed_params = {}
            
            for name, value in params.items():
                # Process tokens in parameter value
                processed_value = await self.token_processor.process_tokens_in_text(str(value))
                processed_params[name] = processed_value
            
            if ctx and params and any(
                processed_params[k] != params[k] for k in params
            ):
                await ctx.info("Applied token substitution to parameters")
            
            return Either.right(processed_params)
            
        except Exception as e:
            return Either.left(MCPError(
                "PARAM_PROCESSING_ERROR",
                f"Failed to process parameter tokens: {str(e)}"
            ))
    
    async def _process_request_data(
        self,
        data: Optional[Union[str, Dict[str, Any]]],
        ctx: Optional[Context]
    ) -> Either[MCPError, Optional[Union[str, Dict[str, Any]]]]:
        """Process request data with token substitution."""
        try:
            if data is None:
                return Either.right(None)
            
            if isinstance(data, str):
                # Process tokens in string data
                processed_data = await self.token_processor.process_tokens_in_text(data)
                
                if ctx and processed_data != data:
                    await ctx.info("Applied token substitution to request data")
                
                return Either.right(processed_data)
            
            elif isinstance(data, dict):
                # Process tokens in dictionary values
                processed_data = {}
                
                for key, value in data.items():
                    if isinstance(value, str):
                        processed_value = await self.token_processor.process_tokens_in_text(value)
                        processed_data[key] = processed_value
                    else:
                        processed_data[key] = value
                
                if ctx and any(
                    isinstance(data[k], str) and processed_data[k] != data[k] 
                    for k in data
                ):
                    await ctx.info("Applied token substitution to request data")
                
                return Either.right(processed_data)
            
            else:
                # Return data as-is for other types
                return Either.right(data)
                
        except Exception as e:
            return Either.left(MCPError(
                "DATA_PROCESSING_ERROR",
                f"Failed to process request data: {str(e)}"
            ))
    
    async def _process_authentication(
        self,
        auth_type: str,
        auth_credentials: Optional[Dict[str, str]],
        ctx: Optional[Context]
    ) -> Either[MCPError, Optional[Dict[str, str]]]:
        """Process authentication credentials."""
        try:
            if auth_type == "none" or not auth_credentials:
                return Either.right(None)
            
            # Create authentication
            auth_result = AuthenticationManager.create_authentication(
                auth_type, auth_credentials
            )
            
            if auth_result.is_left():
                validation_error = auth_result.get_left()
                return Either.left(MCPError(
                    "AUTHENTICATION_ERROR",
                    f"Authentication validation failed: {validation_error.constraint}"
                ))
            
            auth = auth_result.get_right()
            
            # Apply authentication
            headers_result = AuthenticationManager.apply_authentication(auth)
            
            if headers_result.is_left():
                validation_error = headers_result.get_left()
                return Either.left(MCPError(
                    "AUTHENTICATION_APPLICATION_ERROR",
                    f"Failed to apply authentication: {validation_error.constraint}"
                ))
            
            auth_data = headers_result.get_right()
            
            if ctx:
                await ctx.info(f"Applied {auth_type} authentication")
            
            return Either.right(auth_data.get("headers"))
            
        except Exception as e:
            return Either.left(MCPError(
                "AUTHENTICATION_PROCESSING_ERROR",
                f"Failed to process authentication: {str(e)}"
            ))
    
    async def _process_response(
        self,
        response: HTTPResponse,
        response_format: str,
        save_response_to: Optional[str],
        ctx: Optional[Context]
    ) -> Either[MCPError, Dict[str, Any]]:
        """Process HTTP response with format conversion and saving."""
        try:
            # Determine content format
            if response_format == "auto":
                content_type = response.content_type.lower()
                if 'application/json' in content_type:
                    format_type = "json"
                elif content_type.startswith('text/'):
                    format_type = "text"
                elif content_type.startswith('image/'):
                    format_type = "binary"
                else:
                    format_type = "text"
            else:
                format_type = response_format
            
            # Process content based on format
            if format_type == "json":
                json_result = response.get_json()
                if json_result.is_left():
                    content = str(response.content)
                else:
                    content = json_result.get_right()
            else:
                content = response.content
            
            # Save response if requested
            if save_response_to:
                await self._save_response(response, save_response_to, ctx)
            
            # Build response data
            response_data = {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content,
                "url": response.url,
                "method": response.method,
                "duration_ms": response.duration_ms,
                "content_type": response.content_type,
                "content_length": response.content_length,
                "is_success": response.is_success(),
                "is_client_error": response.is_client_error(),
                "is_server_error": response.is_server_error(),
                "format": format_type
            }
            
            return Either.right(response_data)
            
        except Exception as e:
            return Either.left(MCPError(
                "RESPONSE_PROCESSING_ERROR",
                f"Failed to process response: {str(e)}"
            ))
    
    async def _save_response(
        self,
        response: HTTPResponse,
        save_to: str,
        ctx: Optional[Context]
    ) -> None:
        """Save response content to file or variable."""
        try:
            if save_to.startswith("var:"):
                # Save to KM variable
                variable_name = save_to[4:]
                # This would integrate with KM variable system
                # For now, just log
                if ctx:
                    await ctx.info(f"Would save response to variable: {variable_name}")
            
            elif save_to.startswith("file:"):
                # Save to file
                file_path = save_to[5:]
                # This would save to file system with security validation
                # For now, just log
                if ctx:
                    await ctx.info(f"Would save response to file: {file_path}")
            
            else:
                if ctx:
                    await ctx.warn(f"Unknown save target format: {save_to}")
                    
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to save response: {str(e)}")


async def km_web_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Union[str, Dict[str, Any]]] = None,
    params: Optional[Dict[str, str]] = None,
    auth_type: str = "none",
    auth_credentials: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 30,
    follow_redirects: bool = True,
    verify_ssl: bool = True,
    max_response_size: int = 10485760,
    response_format: str = "auto",
    save_response_to: Optional[str] = None,
    ctx: Optional[Context] = None
) -> str:
    """
    Make HTTP requests to APIs, webhooks, and web services.
    
    Enables AI to integrate with modern web services, REST APIs, and cloud platforms.
    Supports all major HTTP methods, authentication types, and response formats.
    Includes comprehensive security validation and token substitution.
    
    Args:
        url: Target URL (supports token substitution)
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)
        headers: Request headers (supports token substitution in values)
        data: Request body data (string or JSON object, supports token substitution)
        params: URL query parameters (supports token substitution in values)
        auth_type: Authentication type (none, api_key, bearer_token, basic_auth, oauth2, custom_header)
        auth_credentials: Authentication credentials (depends on auth_type)
        timeout_seconds: Request timeout (1-300 seconds)
        follow_redirects: Whether to follow HTTP redirects
        verify_ssl: Whether to verify SSL certificates
        max_response_size: Maximum response size in bytes (1KB-100MB)
        response_format: Response parsing format (auto, json, text, xml, binary)
        save_response_to: Save response to file or variable (file:path or var:name)
        
    Returns:
        JSON response with status, headers, content, and metadata
        
    Raises:
        ToolError: For validation failures, security violations, or request errors
        
    Examples:
        # GET request to API
        km_web_request(
            "https://api.github.com/user/repos",
            method="GET",
            auth_type="bearer_token",
            auth_credentials={"token": "github_token"}
        )
        
        # POST JSON data
        km_web_request(
            "https://api.example.com/data",
            method="POST",
            headers={"Content-Type": "application/json"},
            data={"name": "test", "value": 42},
            auth_type="api_key",
            auth_credentials={"api_key": "secret_key"}
        )
        
        # Webhook with token substitution
        km_web_request(
            "https://hooks.slack.com/services/%SlackWebhook%",
            method="POST",
            data={"text": "Automation completed at %ShortTime%"}
        )
    """
    try:
        # Initialize processor
        processor = WebRequestProcessor(allow_localhost=False)
        
        # Process web request
        result = await processor.process_web_request(
            url=url,
            method=method,
            headers=headers,
            data=data,
            params=params,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            timeout_seconds=timeout_seconds,
            follow_redirects=follow_redirects,
            verify_ssl=verify_ssl,
            max_response_size=max_response_size,
            response_format=response_format,
            save_response_to=save_response_to,
            ctx=ctx
        )
        
        if result.is_left():
            error = result.get_left()
            raise ToolError(f"[{error.code}] {error.message}")
        
        return str(result.get_right())
        
    except ToolError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in km_web_request: {str(e)}", exc_info=True)
        raise ToolError(f"Unexpected error processing web request: {str(e)}")


# Tool registration function for MCP server
def register_web_request_tools(mcp):
    """Register web request tools with the MCP server."""
    mcp.tool()(km_web_request)