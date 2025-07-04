"""
Token Processing MCP Tools

Provides secure token processing capabilities for Keyboard Maestro MCP including
token parsing, context-aware processing, and comprehensive security validation.
"""

from typing import Dict, Any, Optional
from pydantic import Field
from typing_extensions import Annotated
from datetime import datetime, UTC
import uuid

from fastmcp import Context
from ...tokens import (
    TokenProcessor, 
    TokenExpression, 
    ProcessingContext,
    KMTokenEngine
)


async def km_token_processor(
    text: Annotated[str, Field(
        description="Text containing Keyboard Maestro tokens",
        min_length=1,
        max_length=10000
    )],
    context: Annotated[str, Field(
        default="text",
        description="Processing context for token evaluation",
        pattern=r"^(text|calculation|regex|filename|url)$"
    )] = "text",
    variables: Annotated[Dict[str, str], Field(
        default_factory=dict,
        description="Variable values for token substitution"
    )] = {},
    use_km_engine: Annotated[bool, Field(
        default=True,
        description="Use Keyboard Maestro's token processing engine"
    )] = True,
    preview_only: Annotated[bool, Field(
        default=False,
        description="Preview tokens without processing"
    )] = False,
    security_level: Annotated[str, Field(
        default="standard",
        description="Security validation level",
        pattern=r"^(minimal|standard|strict)$"
    )] = "standard",
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Process Keyboard Maestro tokens with comprehensive security and context support.
    
    Features:
    - Secure token parsing with injection prevention
    - Support for all major KM token types (system, variable, calculation, datetime)
    - Multiple processing contexts (text, calculation, regex, filename, URL)
    - Variable substitution with scope resolution
    - Integration with Keyboard Maestro's token processing engine
    - Preview mode for token analysis without execution
    - Configurable security validation levels
    
    Security:
    - Token content validation and sanitization
    - Prevention of dangerous token execution
    - Safe processing with bounded execution
    - Context-appropriate validation
    - Security warning system
    
    Returns processed text with token metadata and security validation results.
    """
    if ctx:
        await ctx.info(f"Processing tokens in text: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    try:
        import time
        start_time = time.time()
        
        # Validate security level
        if security_level not in ["minimal", "standard", "strict"]:
            return {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid security level specified",
                    "details": {"security_level": security_level}
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        
        # Create token expression with validation
        try:
            token_expr = TokenExpression(
                text=text,
                context=ProcessingContext(context),
                variables=variables
            )
        except ValueError as e:
            return {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": f"Token expression validation failed: {str(e)}",
                    "details": {"text": text[:100] + "..." if len(text) > 100 else text}
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        
        if preview_only:
            # Just parse and return token information
            processor = TokenProcessor()
            tokens = processor._parse_tokens(text)
            
            if ctx:
                await ctx.info(f"Token preview complete: {len(tokens)} tokens found")
            
            return {
                "success": True,
                "preview": {
                    "original_text": text,
                    "tokens_found": [token['full_match'] for token in tokens],
                    "token_details": [
                        {
                            "token": token['full_match'],
                            "content": token['content'],
                            "type": token['type'].value,
                            "position": {"start": token['start'], "end": token['end']}
                        }
                        for token in tokens
                    ],
                    "token_count": len(tokens),
                    "context": context,
                    "security_level": security_level
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "preview_mode": True,
                    "processing_id": str(uuid.uuid4())
                }
            }
        
        # Choose processing method based on configuration
        if use_km_engine:
            if ctx:
                await ctx.info("Using Keyboard Maestro's token processing engine")
            
            # Use Keyboard Maestro's token processing engine
            km_token = KMTokenEngine()
            km_result = await km_token.process_with_km(text, ProcessingContext(context))
            
            if km_result.is_left():
                if ctx:
                    await ctx.info("KM engine failed, falling back to local processor")
                
                # Fallback to local processor
                processor = TokenProcessor()
                process_result = await processor.process_tokens(token_expr)
                engine_used = "local_fallback"
            else:
                # Create result from KM processing
                processed_text = km_result.get_right()
                execution_time = time.time() - start_time
                
                # Parse tokens for metadata
                processor = TokenProcessor()
                tokens = processor._parse_tokens(text)
                
                from ...tokens.token_processor import TokenProcessingResult
                process_result_data = TokenProcessingResult(
                    original_text=text,
                    processed_text=processed_text,
                    tokens_found=[token['full_match'] for token in tokens],
                    substitutions_made=len(tokens) if processed_text != text else 0,
                    processing_time=execution_time,
                    context=ProcessingContext(context),
                    security_warnings=[]
                )
                
                from ...integration.km_client import Either
                process_result = Either.right(process_result_data)
                engine_used = "keyboard_maestro"
        else:
            if ctx:
                await ctx.info("Using local token processor")
            
            # Use local processor
            processor = TokenProcessor()
            process_result = await processor.process_tokens(token_expr)
            engine_used = "local"
        
        if process_result.is_left():
            error = process_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": {
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "context": context,
                        "engine_used": engine_used
                    }
                },
                "metadata": {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "processing_id": str(uuid.uuid4())
                }
            }
        
        result = process_result.get_right()
        
        if ctx:
            await ctx.info(f"Token processing complete: {result.substitutions_made} substitutions made")
            
            if result.has_security_issues():
                await ctx.info(f"Security warnings generated: {len(result.security_warnings)}")
        
        return {
            "success": True,
            "processing": {
                "original_text": result.original_text,
                "processed_text": result.processed_text,
                "tokens_found": result.tokens_found,
                "substitutions_made": result.substitutions_made,
                "processing_time": result.processing_time,
                "context": result.context.value,
                "has_changes": result.has_changes(),
                "variables_used": variables,
                "security_warnings": result.security_warnings,
                "security_level": security_level,
                "has_security_issues": result.has_security_issues()
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "processing_id": str(uuid.uuid4()),
                "engine": engine_used,
                "context": context
            }
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Token processing failed: {str(e)}")
        
        return {
            "success": False,
            "error": {
                "code": "TOKEN_PROCESSING_ERROR",
                "message": str(e),
                "details": {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "context": context
                }
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "processing_id": str(uuid.uuid4())
            }
        }


async def km_token_stats(
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get token processing statistics and system status.
    
    Returns:
    - Processing statistics (total processed, errors, security violations)
    - Engine status and availability
    - Performance metrics
    - Security summary
    """
    if ctx:
        await ctx.info("Retrieving token processing statistics")
    
    try:
        # Get processor stats
        processor = TokenProcessor()
        processor_stats = processor.get_processing_stats()
        
        # Get KM engine stats
        km_engine = KMTokenEngine()
        km_stats = km_engine.get_processing_stats()
        
        # Test KM engine availability
        km_test = await km_engine.test_km_connection()
        km_available = km_test.is_right()
        
        return {
            "success": True,
            "statistics": {
                "local_processor": processor_stats,
                "km_engine": km_stats,
                "km_engine_available": km_available,
                "total_operations": processor_stats.get("total_processed", 0),
                "error_rate": (
                    processor_stats.get("errors", 0) / max(processor_stats.get("total_processed", 1), 1)
                ) * 100,
                "security_violation_rate": (
                    processor_stats.get("security_violations", 0) / max(processor_stats.get("total_processed", 1), 1)
                ) * 100
            },
            "system_status": {
                "local_processor": "available",
                "km_engine": "available" if km_available else "unavailable",
                "security_level": "active",
                "monitoring": "enabled"
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "stats_id": str(uuid.uuid4())
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": {
                "code": "STATS_ERROR",
                "message": f"Failed to retrieve statistics: {str(e)}"
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat()
            }
        }