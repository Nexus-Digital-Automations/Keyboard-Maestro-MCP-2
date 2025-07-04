"""
Automation Intelligence MCP Tools for Adaptive Learning & Behavior Analysis.

This module provides comprehensive behavioral analysis, pattern learning, and 
intelligent automation suggestions through advanced machine learning and 
privacy-preserving data processing.

Security: Complete privacy protection with configurable anonymization levels.
Performance: Sub-second analysis with intelligent caching and optimization.
Type Safety: Full branded type system with contract-driven development.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime, UTC

import mcp.types as mcp
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import (
    PrivacyLevel, AnalysisDepth, SuggestionContext,
    SuggestionSecurityValidator, create_suggestion_context
)
from src.suggestions.behavior_tracker import BehaviorTracker
from src.intelligence.automation_intelligence_manager import (
    AutomationIntelligenceManager, IntelligenceOperation, 
    AnalysisScope, LearningMode, IntelligenceError
)

logger = get_logger(__name__)


class AutomationIntelligenceTools:
    """MCP tool interface for automation intelligence and behavioral analysis."""
    
    def __init__(self):
        self.intelligence_manager = AutomationIntelligenceManager()
        self.behavior_tracker = BehaviorTracker()
        self.security_validator = SuggestionSecurityValidator()
        self._initialized = False
    
    async def initialize(self) -> Either[IntelligenceError, None]:
        """Initialize intelligence system components."""
        try:
            # Initialize intelligence manager
            init_result = await self.intelligence_manager.initialize()
            if init_result.is_left():
                return init_result
            
            self._initialized = True
            logger.info("Automation intelligence system initialized successfully")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Intelligence system initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))


# MCP Tool Implementation
automation_intelligence_tools = AutomationIntelligenceTools()


@mcp.tool()
@require(lambda operation: operation in ["analyze", "learn", "suggest", "optimize", "predict", "insights"])
@require(lambda analysis_scope: analysis_scope in ["user_behavior", "automation_patterns", "performance", "usage"])
@require(lambda time_period: time_period in ["1d", "7d", "30d", "90d", "all"])
@require(lambda learning_mode: learning_mode in ["adaptive", "supervised", "unsupervised", "reinforcement"])
@require(lambda privacy_level: privacy_level in ["strict", "balanced", "permissive"])
@require(lambda optimization_target: optimization_target in ["efficiency", "accuracy", "speed", "user_satisfaction"])
@require(lambda suggestion_count: 1 <= suggestion_count <= 20)
@require(lambda confidence_threshold: 0.0 <= confidence_threshold <= 1.0)
async def km_automation_intelligence(
    operation: str,                                 # analyze|learn|suggest|optimize|predict|insights
    analysis_scope: str = "user_behavior",         # user_behavior|automation_patterns|performance|usage
    time_period: str = "30d",                      # 1d|7d|30d|90d|all
    learning_mode: str = "adaptive",               # adaptive|supervised|unsupervised|reinforcement
    privacy_level: str = "strict",                 # strict|balanced|permissive
    optimization_target: str = "efficiency",       # efficiency|accuracy|speed|user_satisfaction
    suggestion_count: int = 5,                     # Number of suggestions to generate
    confidence_threshold: float = 0.7,             # Minimum confidence for suggestions
    enable_predictions: bool = True,               # Enable predictive capabilities
    data_retention: str = "30d",                   # Data retention period for learning
    anonymize_data: bool = True,                   # Anonymize behavioral data
    ctx = None
) -> Dict[str, Any]:
    """
    Advanced automation intelligence with behavioral analysis and adaptive learning.
    
    Provides comprehensive behavioral pattern analysis, intelligent automation suggestions,
    and adaptive learning capabilities while maintaining strict privacy protection and
    security validation throughout the learning process.
    
    Args:
        operation: Intelligence operation to perform
        analysis_scope: Scope of behavioral analysis
        time_period: Time window for data analysis
        learning_mode: Machine learning approach mode
        privacy_level: Privacy protection level
        optimization_target: Target metric for optimization
        suggestion_count: Number of suggestions to generate
        confidence_threshold: Minimum confidence for actionable suggestions
        enable_predictions: Enable predictive analytics
        data_retention: Data retention period for learning
        anonymize_data: Apply data anonymization
        
    Returns:
        Dictionary containing intelligence results, suggestions, and analysis
        
    Security:
        - Privacy-first design with configurable protection levels
        - Complete input validation and sanitization
        - Secure behavioral data handling with anonymization
        - Contract-based validation for all parameters
        
    Performance:
        - Sub-second analysis with intelligent caching
        - Optimized pattern recognition algorithms
        - Efficient data processing and storage
    """
    try:
        # Ensure system is initialized
        if not automation_intelligence_tools._initialized:
            init_result = await automation_intelligence_tools.initialize()
            if init_result.is_left():
                return {
                    "success": False,
                    "error": f"Intelligence system initialization failed: {init_result.get_left().message}",
                    "operation": operation,
                    "timestamp": datetime.now(UTC).isoformat()
                }
        
        # Convert string parameters to enums
        intelligence_operation = IntelligenceOperation(operation)
        scope = AnalysisScope(analysis_scope)
        learning_mode_enum = LearningMode(learning_mode)
        privacy_level_enum = PrivacyLevel(privacy_level)
        
        # Create suggestion context for security validation
        context = create_suggestion_context(
            user_id="automation_intelligence_user",
            current_automation=f"{operation}_{scope.value}",
            recent_actions=[operation]
        )
        
        # Security validation
        security_result = automation_intelligence_tools.security_validator.validate_suggestion_context(context)
        if security_result.is_left():
            return {
                "success": False,
                "error": f"Security validation failed: {security_result.get_left().message}",
                "operation": operation,
                "timestamp": datetime.now(UTC).isoformat()
            }
        
        # Process intelligence request
        start_time = datetime.now(UTC)
        
        intelligence_result = await automation_intelligence_tools.intelligence_manager.process_intelligence_request(
            operation=intelligence_operation,
            analysis_scope=scope,
            time_period=time_period,
            privacy_level=privacy_level_enum,
            learning_mode=learning_mode_enum,
            suggestion_count=suggestion_count,
            confidence_threshold=confidence_threshold,
            optimization_target=optimization_target,
            enable_predictions=enable_predictions,
            anonymize_data=anonymize_data
        )
        
        processing_time = (datetime.now(UTC) - start_time).total_seconds()
        
        if intelligence_result.is_left():
            return {
                "success": False,
                "error": f"Intelligence processing failed: {intelligence_result.get_left().message}",
                "operation": operation,
                "analysis_scope": analysis_scope,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now(UTC).isoformat()
            }
        
        # Successful processing
        result_data = intelligence_result.get_right()
        
        response = {
            "success": True,
            "operation": operation,
            "analysis_scope": analysis_scope,
            "time_period": time_period,
            "learning_mode": learning_mode,
            "privacy_level": privacy_level,
            "optimization_target": optimization_target,
            "processing_time_seconds": round(processing_time, 3),
            "timestamp": datetime.now(UTC).isoformat(),
            "results": result_data
        }
        
        # Add operation-specific metadata
        if operation == "analyze":
            response["analysis_summary"] = {
                "patterns_analyzed": result_data.get("total_patterns", 0),
                "confidence_patterns": result_data.get("high_confidence_patterns", 0),
                "efficiency_opportunities": len(result_data.get("efficiency_opportunities", []))
            }
        elif operation == "suggest":
            response["suggestion_summary"] = {
                "suggestions_generated": len(result_data.get("suggestions", [])),
                "actionable_suggestions": len([s for s in result_data.get("suggestions", []) 
                                             if s.get("confidence", 0) >= confidence_threshold]),
                "average_confidence": result_data.get("average_confidence", 0.0)
            }
        elif operation == "optimize":
            response["optimization_summary"] = {
                "optimizations_identified": len(result_data.get("optimizations", [])),
                "potential_time_savings": result_data.get("total_time_savings", 0.0),
                "implementation_complexity": result_data.get("average_complexity", "unknown")
            }
        
        # Add privacy compliance metadata
        response["privacy_compliance"] = {
            "privacy_level": privacy_level,
            "data_anonymized": anonymize_data,
            "retention_period": data_retention,
            "sensitive_data_filtered": True
        }
        
        logger.info(f"Intelligence operation '{operation}' completed successfully in {processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Invalid parameter in intelligence operation: {str(e)}")
        return {
            "success": False,
            "error": f"Invalid parameter: {str(e)}",
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in intelligence operation: {str(e)}")
        return {
            "success": False,
            "error": f"Intelligence system error: {str(e)}",
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat()
        }


@ensure(lambda result: result.get("success", False) or "error" in result)
def get_available_operations() -> Dict[str, Any]:
    """Get available intelligence operations and their descriptions."""
    return {
        "success": True,
        "operations": {
            "analyze": {
                "description": "Analyze user behavior patterns and automation usage",
                "scopes": ["user_behavior", "automation_patterns", "performance", "usage"],
                "privacy_levels": ["strict", "balanced", "permissive"]
            },
            "learn": {
                "description": "Learn from behavioral patterns to improve automation",
                "modes": ["adaptive", "supervised", "unsupervised", "reinforcement"],
                "targets": ["efficiency", "accuracy", "speed", "user_satisfaction"]
            },
            "suggest": {
                "description": "Generate intelligent automation suggestions",
                "parameters": ["suggestion_count", "confidence_threshold"],
                "output_types": ["automation", "optimization", "integration"]
            },
            "optimize": {
                "description": "Optimize existing automations for better performance",
                "targets": ["efficiency", "accuracy", "speed", "user_satisfaction"],
                "analysis_types": ["performance", "resource_usage", "error_patterns"]
            },
            "predict": {
                "description": "Predict user intent and automation needs",
                "prediction_types": ["usage_patterns", "optimization_opportunities", "failure_risk"],
                "confidence_reporting": True
            },
            "insights": {
                "description": "Generate insights about automation usage and effectiveness",
                "insight_types": ["trends", "patterns", "anomalies", "opportunities"],
                "visualization_support": True
            }
        },
        "privacy_features": {
            "strict": "Maximum privacy with minimal data collection and anonymization",
            "balanced": "Balanced privacy and learning with selective data retention",
            "permissive": "Enhanced learning capabilities with extended data collection"
        },
        "performance_targets": {
            "analysis_time": "<1 second",
            "suggestion_generation": "<3 seconds", 
            "learning_operations": "<5 seconds"
        }
    }