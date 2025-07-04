"""
Intelligent automation system for AI-powered adaptive workflows.

This module provides intelligent automation capabilities including smart triggers,
adaptive workflows, context awareness, and AI-powered decision engines that
learn from user behavior and adapt automation based on patterns and context.

Security: All automation includes comprehensive validation and safe execution.
Performance: Optimized for real-time automation with intelligent caching.
Type Safety: Complete integration with AI processing architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NewType, Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import json
import re

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError
from ..core.ai_integration import (
    AIOperation, AIRequest, AIResponse, create_ai_request,
    AIModelId, AISessionId, ProcessingMode, OutputFormat
)

# Branded Types for Intelligent Automation
AutomationRuleId = NewType('AutomationRuleId', str)
WorkflowInstanceId = NewType('WorkflowInstanceId', str)
ContextStateId = NewType('ContextStateId', str)
DecisionNodeId = NewType('DecisionNodeId', str)
AdaptationScore = NewType('AdaptationScore', float)
ConfidenceLevel = NewType('ConfidenceLevel', float)


class AutomationTriggerType(Enum):
    """Types of intelligent automation triggers."""
    PATTERN_DETECTED = "pattern_detected"        # User pattern recognition
    CONTEXT_CHANGED = "context_changed"          # Context state change
    CONTENT_ANALYZED = "content_analyzed"        # AI content analysis result
    THRESHOLD_REACHED = "threshold_reached"      # Performance threshold
    SCHEDULE_BASED = "schedule_based"            # Time-based automation
    USER_INITIATED = "user_initiated"            # Manual trigger
    SYSTEM_EVENT = "system_event"                # System state change
    ADAPTIVE_SUGGESTION = "adaptive_suggestion"  # AI-generated suggestion


class WorkflowAdaptationType(Enum):
    """Types of workflow adaptations."""
    PARAMETER_OPTIMIZATION = "parameter_optimization"  # Optimize parameters
    STEP_REORDERING = "step_reordering"              # Change step order
    CONDITIONAL_ADDITION = "conditional_addition"     # Add conditions
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement" # Speed optimization
    ERROR_PREVENTION = "error_prevention"            # Prevent common errors
    USER_PREFERENCE = "user_preference"              # User preference adaptation


class ContextDimension(Enum):
    """Context awareness dimensions."""
    TEMPORAL = "temporal"                  # Time-based context
    SPATIAL = "spatial"                    # Location/screen context
    APPLICATION = "application"            # Active applications
    CONTENT = "content"                    # Content being worked with
    USER_STATE = "user_state"             # User activity state
    SYSTEM_STATE = "system_state"         # System resource state
    WORKFLOW = "workflow"                  # Current workflow context


@dataclass(frozen=True)
class ContextState:
    """Comprehensive context state representation."""
    context_id: ContextStateId
    timestamp: datetime
    dimensions: Dict[ContextDimension, Any]
    confidence: ConfidenceLevel = ConfidenceLevel(0.8)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.dimensions) > 0)
    def __post_init__(self):
        """Validate context state."""
        pass
    
    def get_dimension_value(self, dimension: ContextDimension) -> Optional[Any]:
        """Get value for specific context dimension."""
        return self.dimensions.get(dimension)
    
    def similarity_to(self, other: 'ContextState') -> float:
        """Calculate similarity to another context state."""
        if not self.dimensions or not other.dimensions:
            return 0.0
        
        common_dimensions = set(self.dimensions.keys()) & set(other.dimensions.keys())
        if not common_dimensions:
            return 0.0
        
        similarity_scores = []
        for dim in common_dimensions:
            # Simple similarity calculation - can be enhanced
            val1, val2 = str(self.dimensions[dim]), str(other.dimensions[dim])
            if val1 == val2:
                similarity_scores.append(1.0)
            else:
                # Basic string similarity
                similarity_scores.append(self._string_similarity(val1, val2))
        
        return sum(similarity_scores) / len(similarity_scores)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate basic string similarity."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Simple character overlap similarity
        chars1, chars2 = set(s1.lower()), set(s2.lower())
        if not chars1 or not chars2:
            return 0.0
        
        return len(chars1 & chars2) / len(chars1 | chars2)


@dataclass(frozen=True)
class SmartTrigger:
    """AI-powered intelligent trigger configuration."""
    trigger_id: str
    trigger_type: AutomationTriggerType
    conditions: Dict[str, Any]
    ai_analysis_required: bool = False
    context_requirements: Set[ContextDimension] = field(default_factory=set)
    confidence_threshold: ConfidenceLevel = ConfidenceLevel(0.7)
    cooldown_period: timedelta = timedelta(minutes=5)
    adaptation_enabled: bool = True
    
    @require(lambda self: len(self.trigger_id) > 0)
    @require(lambda self: 0.0 <= self.confidence_threshold <= 1.0)
    @require(lambda self: self.cooldown_period.total_seconds() >= 0)
    def __post_init__(self):
        """Validate smart trigger configuration."""
        pass
    
    def should_trigger(self, context: ContextState, analysis_result: Optional[Dict] = None) -> bool:
        """Determine if trigger should fire based on context and analysis."""
        # Check context requirements
        if self.context_requirements:
            available_dims = set(context.dimensions.keys())
            if not self.context_requirements.issubset(available_dims):
                return False
        
        # Check confidence threshold
        if context.confidence < self.confidence_threshold:
            return False
        
        # Check AI analysis if required
        if self.ai_analysis_required and not analysis_result:
            return False
        
        # Evaluate trigger conditions
        return self._evaluate_conditions(context, analysis_result)
    
    def _evaluate_conditions(self, context: ContextState, analysis_result: Optional[Dict]) -> bool:
        """Evaluate trigger conditions against context and analysis."""
        for condition_key, condition_value in self.conditions.items():
            if condition_key.startswith("context."):
                # Context-based condition
                dim_name = condition_key[8:]  # Remove "context." prefix
                try:
                    dimension = ContextDimension(dim_name)
                    context_value = context.get_dimension_value(dimension)
                    if not self._match_condition_value(context_value, condition_value):
                        return False
                except ValueError:
                    return False  # Invalid dimension
            
            elif condition_key.startswith("analysis.") and analysis_result:
                # Analysis-based condition
                analysis_key = condition_key[9:]  # Remove "analysis." prefix
                analysis_value = analysis_result.get(analysis_key)
                if not self._match_condition_value(analysis_value, condition_value):
                    return False
            
            elif condition_key == "time_window":
                # Time-based condition
                if not self._check_time_window(condition_value):
                    return False
        
        return True
    
    def _match_condition_value(self, actual: Any, expected: Any) -> bool:
        """Match condition value with various comparison types."""
        if isinstance(expected, dict):
            operator = expected.get("operator", "equals")
            value = expected.get("value")
            
            if operator == "equals":
                return actual == value
            elif operator == "contains" and isinstance(actual, str):
                return str(value).lower() in actual.lower()
            elif operator == "greater_than" and isinstance(actual, (int, float)):
                return actual > value
            elif operator == "less_than" and isinstance(actual, (int, float)):
                return actual < value
            elif operator == "in_list":
                return actual in value if isinstance(value, list) else False
        
        return actual == expected
    
    def _check_time_window(self, time_window: Dict[str, Any]) -> bool:
        """Check if current time is within specified window."""
        now = datetime.now()
        start_hour = time_window.get("start_hour", 0)
        end_hour = time_window.get("end_hour", 23)
        days = time_window.get("days", list(range(7)))  # 0=Monday
        
        current_hour = now.hour
        current_day = now.weekday()
        
        return start_hour <= current_hour <= end_hour and current_day in days


@dataclass(frozen=True)
class AdaptiveWorkflow:
    """Self-adapting workflow that learns from execution patterns."""
    workflow_id: WorkflowInstanceId
    base_steps: List[Dict[str, Any]]
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_adaptations: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    learning_enabled: bool = True
    adaptation_score: AdaptationScore = AdaptationScore(0.0)
    
    @require(lambda self: len(self.base_steps) > 0)
    @require(lambda self: 0.0 <= self.adaptation_score <= 1.0)
    def __post_init__(self):
        """Validate adaptive workflow configuration."""
        pass
    
    def get_optimized_steps(self, context: ContextState) -> List[Dict[str, Any]]:
        """Get workflow steps optimized for current context."""
        steps = self.base_steps.copy()
        
        # Apply current adaptations
        for adaptation_type, adaptation_data in self.current_adaptations.items():
            steps = self._apply_adaptation(steps, adaptation_type, adaptation_data, context)
        
        return steps
    
    def _apply_adaptation(self, steps: List[Dict[str, Any]], adaptation_type: str, 
                         adaptation_data: Dict[str, Any], context: ContextState) -> List[Dict[str, Any]]:
        """Apply specific adaptation to workflow steps."""
        if adaptation_type == "parameter_optimization":
            return self._optimize_parameters(steps, adaptation_data, context)
        elif adaptation_type == "step_reordering":
            return self._reorder_steps(steps, adaptation_data)
        elif adaptation_type == "conditional_addition":
            return self._add_conditions(steps, adaptation_data, context)
        elif adaptation_type == "efficiency_improvement":
            return self._improve_efficiency(steps, adaptation_data)
        
        return steps
    
    def _optimize_parameters(self, steps: List[Dict[str, Any]], 
                           optimization_data: Dict[str, Any], context: ContextState) -> List[Dict[str, Any]]:
        """Optimize step parameters based on learning."""
        optimized_steps = []
        for step in steps:
            optimized_step = step.copy()
            step_id = step.get("id", "")
            
            if step_id in optimization_data:
                optimizations = optimization_data[step_id]
                for param_name, param_value in optimizations.items():
                    if self._should_apply_optimization(param_name, param_value, context):
                        optimized_step[param_name] = param_value
            
            optimized_steps.append(optimized_step)
        
        return optimized_steps
    
    def _should_apply_optimization(self, param_name: str, param_value: Any, context: ContextState) -> bool:
        """Determine if optimization should be applied in current context."""
        # Context-aware optimization application
        if param_name == "timeout" and context.get_dimension_value(ContextDimension.SYSTEM_STATE):
            system_load = context.dimensions.get(ContextDimension.SYSTEM_STATE, {}).get("cpu_usage", 0)
            if system_load > 80:  # High system load
                return isinstance(param_value, (int, float)) and param_value > 30  # Longer timeout
        
        return True
    
    def _reorder_steps(self, steps: List[Dict[str, Any]], reorder_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reorder workflow steps for efficiency."""
        if "optimal_order" not in reorder_data:
            return steps
        
        optimal_order = reorder_data["optimal_order"]
        if len(optimal_order) != len(steps):
            return steps
        
        try:
            return [steps[i] for i in optimal_order]
        except (IndexError, TypeError):
            return steps
    
    def _add_conditions(self, steps: List[Dict[str, Any]], 
                       condition_data: Dict[str, Any], context: ContextState) -> List[Dict[str, Any]]:
        """Add intelligent conditions to workflow steps."""
        enhanced_steps = []
        for i, step in enumerate(steps):
            enhanced_step = step.copy()
            
            step_conditions = condition_data.get(str(i), [])
            for condition in step_conditions:
                if self._should_add_condition(condition, context):
                    enhanced_step.setdefault("conditions", []).append(condition)
            
            enhanced_steps.append(enhanced_step)
        
        return enhanced_steps
    
    def _should_add_condition(self, condition: Dict[str, Any], context: ContextState) -> bool:
        """Determine if condition should be added based on context."""
        condition_type = condition.get("type", "")
        
        if condition_type == "application_active":
            app_name = condition.get("application")
            current_app = context.get_dimension_value(ContextDimension.APPLICATION)
            return current_app and app_name in str(current_app)
        
        elif condition_type == "time_based":
            return self._check_time_condition(condition)
        
        return True
    
    def _check_time_condition(self, condition: Dict[str, Any]) -> bool:
        """Check time-based condition."""
        now = datetime.now()
        hour_range = condition.get("hour_range", [0, 23])
        return hour_range[0] <= now.hour <= hour_range[1]
    
    def _improve_efficiency(self, steps: List[Dict[str, Any]], efficiency_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply efficiency improvements to workflow steps."""
        improved_steps = []
        for step in steps:
            improved_step = step.copy()
            
            # Apply parallel execution where possible
            if efficiency_data.get("enable_parallel") and step.get("parallelizable", False):
                improved_step["execution_mode"] = "parallel"
            
            # Optimize delays and timeouts
            if "timeout_optimization" in efficiency_data:
                timeout_factor = efficiency_data["timeout_optimization"]
                if "timeout" in improved_step:
                    improved_step["timeout"] = max(1, int(improved_step["timeout"] * timeout_factor))
            
            improved_steps.append(improved_step)
        
        return improved_steps
    
    def record_execution_result(self, execution_time: float, success: bool, 
                              context: ContextState, errors: List[str] = None) -> None:
        """Record execution result for learning."""
        if not self.learning_enabled:
            return
        
        result_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "execution_time": execution_time,
            "success": success,
            "context_snapshot": {
                "dimensions": {k.value: v for k, v in context.dimensions.items()},
                "confidence": float(context.confidence)
            },
            "errors": errors or [],
            "adaptations_used": dict(self.current_adaptations)
        }
        
        # Update adaptation history (keep last 100 records)
        updated_history = list(self.adaptation_history) + [result_record]
        if len(updated_history) > 100:
            updated_history = updated_history[-100:]
        
        object.__setattr__(self, 'adaptation_history', updated_history)
        
        # Update performance metrics
        self._update_performance_metrics(execution_time, success)
    
    def _update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update workflow performance metrics."""
        updated_metrics = dict(self.performance_metrics)
        
        # Update success rate
        total_runs = updated_metrics.get("total_runs", 0) + 1
        successful_runs = updated_metrics.get("successful_runs", 0) + (1 if success else 0)
        updated_metrics.update({
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0
        })
        
        # Update timing metrics
        avg_time = updated_metrics.get("average_execution_time", 0.0)
        updated_metrics["average_execution_time"] = ((avg_time * (total_runs - 1)) + execution_time) / total_runs
        
        if "min_execution_time" not in updated_metrics or execution_time < updated_metrics["min_execution_time"]:
            updated_metrics["min_execution_time"] = execution_time
        
        if "max_execution_time" not in updated_metrics or execution_time > updated_metrics["max_execution_time"]:
            updated_metrics["max_execution_time"] = execution_time
        
        object.__setattr__(self, 'performance_metrics', updated_metrics)


@dataclass(frozen=True)
class DecisionNode:
    """AI-powered decision node for intelligent automation."""
    node_id: DecisionNodeId
    decision_type: str
    ai_operation: AIOperation
    decision_criteria: Dict[str, Any]
    fallback_decision: str
    confidence_threshold: ConfidenceLevel = ConfidenceLevel(0.8)
    cache_duration: timedelta = timedelta(minutes=30)
    
    @require(lambda self: len(self.node_id) > 0)
    @require(lambda self: 0.0 <= self.confidence_threshold <= 1.0)
    def __post_init__(self):
        """Validate decision node configuration."""
        pass
    
    async def make_decision(self, input_data: Any, context: ContextState,
                          ai_processor: 'AIProcessingManager') -> Either[ValidationError, str]:
        """Make AI-powered decision based on input and context."""
        try:
            # Prepare decision prompt
            decision_prompt = self._prepare_decision_prompt(input_data, context)
            
            # Create AI request
            request_result = create_ai_request(
                operation=self.ai_operation,
                input_data=decision_prompt,
                processing_mode=ProcessingMode.ACCURATE,
                temperature=0.3,  # Lower temperature for consistent decisions
                context={"decision_node": self.node_id, "decision_type": self.decision_type}
            )
            
            if request_result.is_left():
                return Either.left(request_result.get_left())
            
            # Process with AI
            response_result = await ai_processor.process_ai_request(
                request_result.get_right().operation,
                request_result.get_right().input_data,
                processing_mode="accurate",
                temperature=0.3,
                enable_caching=True
            )
            
            if response_result.is_left():
                return Either.right(self.fallback_decision)
            
            response = response_result.get_right()
            
            # Extract decision from AI response
            decision = self._extract_decision(response.get("result", ""))
            
            # Validate decision confidence
            confidence = response.get("metadata", {}).get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                return Either.right(self.fallback_decision)
            
            return Either.right(decision)
            
        except Exception as e:
            return Either.left(ValidationError("decision_failed", str(e)))
    
    def _prepare_decision_prompt(self, input_data: Any, context: ContextState) -> str:
        """Prepare AI prompt for decision making."""
        base_prompt = f"Decision Type: {self.decision_type}\n\n"
        
        # Add input data
        base_prompt += f"Input Data:\n{json.dumps(input_data, indent=2)}\n\n"
        
        # Add context information
        context_info = {}
        for dim, value in context.dimensions.items():
            context_info[dim.value] = value
        
        base_prompt += f"Context:\n{json.dumps(context_info, indent=2)}\n\n"
        
        # Add decision criteria
        base_prompt += f"Decision Criteria:\n{json.dumps(self.decision_criteria, indent=2)}\n\n"
        
        # Add decision instruction
        base_prompt += f"Please analyze the input data and context, then make a decision based on the criteria. "
        base_prompt += f"Return only the decision value as a single word or short phrase. "
        base_prompt += f"If uncertain, return: {self.fallback_decision}"
        
        return base_prompt
    
    def _extract_decision(self, ai_response: str) -> str:
        """Extract decision from AI response."""
        # Clean and extract the decision
        decision = ai_response.strip()
        
        # Remove common response prefixes
        prefixes_to_remove = [
            "decision:", "the decision is:", "i decide:", "my decision:", 
            "based on the analysis:", "conclusion:"
        ]
        
        decision_lower = decision.lower()
        for prefix in prefixes_to_remove:
            if decision_lower.startswith(prefix):
                decision = decision[len(prefix):].strip()
                break
        
        # Extract first meaningful word/phrase (up to 50 characters)
        words = decision.split()
        if words:
            # Take first word or first few words if short
            if len(words[0]) > 3:
                return words[0]
            elif len(words) > 1 and len(' '.join(words[:2])) <= 50:
                return ' '.join(words[:2])
            else:
                return words[0]
        
        return self.fallback_decision


class IntelligentAutomationEngine:
    """Comprehensive intelligent automation engine with AI-powered decision making."""
    
    def __init__(self):
        self.smart_triggers: Dict[str, SmartTrigger] = {}
        self.adaptive_workflows: Dict[WorkflowInstanceId, AdaptiveWorkflow] = {}
        self.decision_nodes: Dict[DecisionNodeId, DecisionNode] = {}
        self.context_history: List[ContextState] = []
        self.automation_sessions: Dict[str, Dict[str, Any]] = {}
        self.learning_enabled = True
    
    async def evaluate_triggers(self, context: ContextState, 
                              ai_processor: Optional['AIProcessingManager'] = None) -> List[str]:
        """Evaluate all smart triggers and return triggered automation IDs."""
        triggered_automations = []
        
        for trigger_id, trigger in self.smart_triggers.items():
            try:
                analysis_result = None
                
                # Perform AI analysis if required
                if trigger.ai_analysis_required and ai_processor:
                    analysis_result = await self._perform_trigger_analysis(trigger, context, ai_processor)
                
                # Check if trigger should fire
                if trigger.should_trigger(context, analysis_result):
                    # Check cooldown period
                    if self._check_trigger_cooldown(trigger_id):
                        triggered_automations.append(trigger_id)
                        self._record_trigger_activation(trigger_id)
                
            except Exception as e:
                # Log error but continue processing other triggers
                continue
        
        return triggered_automations
    
    async def _perform_trigger_analysis(self, trigger: SmartTrigger, context: ContextState,
                                      ai_processor: 'AIProcessingManager') -> Optional[Dict[str, Any]]:
        """Perform AI analysis for trigger evaluation."""
        try:
            # Prepare analysis data
            analysis_input = {
                "context": {dim.value: value for dim, value in context.dimensions.items()},
                "trigger_type": trigger.trigger_type.value,
                "conditions": trigger.conditions
            }
            
            # Request AI analysis
            response_result = await ai_processor.process_ai_request(
                operation=AIOperation.ANALYZE,
                input_data=json.dumps(analysis_input),
                processing_mode="fast",
                output_format="json",
                enable_caching=True
            )
            
            if response_result.is_right():
                response = response_result.get_right()
                if isinstance(response.get("result"), dict):
                    return response["result"]
            
            return None
            
        except Exception:
            return None
    
    def _check_trigger_cooldown(self, trigger_id: str) -> bool:
        """Check if trigger is within cooldown period."""
        session = self.automation_sessions.get(trigger_id, {})
        last_activation = session.get("last_activation")
        
        if not last_activation:
            return True
        
        trigger = self.smart_triggers.get(trigger_id)
        if not trigger:
            return True
        
        time_since_last = datetime.now(UTC) - last_activation
        return time_since_last >= trigger.cooldown_period
    
    def _record_trigger_activation(self, trigger_id: str) -> None:
        """Record trigger activation for cooldown tracking."""
        if trigger_id not in self.automation_sessions:
            self.automation_sessions[trigger_id] = {}
        
        self.automation_sessions[trigger_id]["last_activation"] = datetime.now(UTC)
        self.automation_sessions[trigger_id]["activation_count"] = \
            self.automation_sessions[trigger_id].get("activation_count", 0) + 1
    
    def execute_adaptive_workflow(self, workflow_id: WorkflowInstanceId, 
                                context: ContextState) -> Either[ValidationError, List[Dict[str, Any]]]:
        """Execute adaptive workflow with context-aware optimization."""
        try:
            workflow = self.adaptive_workflows.get(workflow_id)
            if not workflow:
                return Either.left(ValidationError("workflow_not_found", f"Workflow {workflow_id} not found"))
            
            # Get optimized steps for current context
            optimized_steps = workflow.get_optimized_steps(context)
            
            # Record context for learning
            self._record_context_for_learning(workflow_id, context)
            
            return Either.right(optimized_steps)
            
        except Exception as e:
            return Either.left(ValidationError("workflow_execution_failed", str(e)))
    
    def _record_context_for_learning(self, workflow_id: WorkflowInstanceId, context: ContextState) -> None:
        """Record context for workflow learning and adaptation."""
        # Add to context history (keep last 1000 entries)
        self.context_history.append(context)
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
        
        # Update workflow session
        session_key = f"workflow_{workflow_id}"
        if session_key not in self.automation_sessions:
            self.automation_sessions[session_key] = {"contexts": []}
        
        session = self.automation_sessions[session_key]
        session["contexts"].append({
            "timestamp": context.timestamp.isoformat(),
            "dimensions": {k.value: v for k, v in context.dimensions.items()},
            "confidence": float(context.confidence)
        })
        
        # Keep only recent contexts
        if len(session["contexts"]) > 50:
            session["contexts"] = session["contexts"][-50:]
    
    def update_context_state(self, new_context: ContextState) -> None:
        """Update current context state and trigger analysis."""
        # Add to history
        self.context_history.append(new_context)
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
        
        # Trigger context-based adaptations if learning is enabled
        if self.learning_enabled:
            self._analyze_context_patterns(new_context)
    
    def _analyze_context_patterns(self, current_context: ContextState) -> None:
        """Analyze context patterns for automation optimization."""
        if len(self.context_history) < 10:
            return  # Need sufficient history
        
        # Find similar contexts
        similar_contexts = []
        for past_context in self.context_history[-50:]:  # Check recent history
            similarity = current_context.similarity_to(past_context)
            if similarity > 0.8:  # High similarity threshold
                similar_contexts.append((past_context, similarity))
        
        # If we have similar contexts, analyze patterns
        if len(similar_contexts) >= 3:
            self._identify_adaptation_opportunities(current_context, similar_contexts)
    
    def _identify_adaptation_opportunities(self, current_context: ContextState,
                                         similar_contexts: List[tuple]) -> None:
        """Identify opportunities for workflow adaptation based on patterns."""
        # This is a simplified implementation - in practice would use more
        # sophisticated pattern recognition and machine learning
        
        # Look for temporal patterns
        temporal_values = []
        for context, _ in similar_contexts:
            temporal_data = context.get_dimension_value(ContextDimension.TEMPORAL)
            if temporal_data:
                temporal_values.append(temporal_data)
        
        # Look for application patterns
        app_values = []
        for context, _ in similar_contexts:
            app_data = context.get_dimension_value(ContextDimension.APPLICATION)
            if app_data:
                app_values.append(app_data)
        
        # Record pattern insights (would trigger workflow adaptations)
        pattern_insights = {
            "temporal_patterns": self._analyze_temporal_patterns(temporal_values),
            "application_patterns": self._analyze_application_patterns(app_values),
            "context_frequency": len(similar_contexts),
            "analysis_timestamp": datetime.now(UTC).isoformat()
        }
        
        # Store insights for workflow adaptation
        self.automation_sessions["pattern_insights"] = pattern_insights
    
    def _analyze_temporal_patterns(self, temporal_values: List[Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in context data."""
        if not temporal_values:
            return {}
        
        # Simple temporal analysis
        hours = []
        for temporal_data in temporal_values:
            if isinstance(temporal_data, dict) and "hour" in temporal_data:
                hours.append(temporal_data["hour"])
        
        if hours:
            avg_hour = sum(hours) / len(hours)
            return {
                "common_hours": hours,
                "average_hour": avg_hour,
                "pattern_strength": len(set(hours)) / len(hours) if hours else 0
            }
        
        return {}
    
    def _analyze_application_patterns(self, app_values: List[Any]) -> Dict[str, Any]:
        """Analyze application patterns in context data."""
        if not app_values:
            return {}
        
        # Count application frequencies
        app_counts = {}
        for app_data in app_values:
            if isinstance(app_data, str):
                app_counts[app_data] = app_counts.get(app_data, 0) + 1
            elif isinstance(app_data, dict) and "name" in app_data:
                app_name = app_data["name"]
                app_counts[app_name] = app_counts.get(app_name, 0) + 1
        
        if app_counts:
            most_common = max(app_counts.items(), key=lambda x: x[1])
            return {
                "application_frequencies": app_counts,
                "most_common_app": most_common[0],
                "pattern_strength": most_common[1] / len(app_values)
            }
        
        return {}
    
    def add_smart_trigger(self, trigger: SmartTrigger) -> None:
        """Add smart trigger to the automation engine."""
        self.smart_triggers[trigger.trigger_id] = trigger
    
    def add_adaptive_workflow(self, workflow: AdaptiveWorkflow) -> None:
        """Add adaptive workflow to the automation engine."""
        self.adaptive_workflows[workflow.workflow_id] = workflow
    
    def add_decision_node(self, node: DecisionNode) -> None:
        """Add decision node to the automation engine."""
        self.decision_nodes[node.node_id] = node
    
    def get_automation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive automation system statistics."""
        total_triggers = len(self.smart_triggers)
        total_workflows = len(self.adaptive_workflows)
        total_contexts = len(self.context_history)
        
        # Calculate workflow performance
        workflow_stats = {}
        for workflow_id, workflow in self.adaptive_workflows.items():
            workflow_stats[str(workflow_id)] = {
                "adaptation_score": float(workflow.adaptation_score),
                "performance_metrics": workflow.performance_metrics,
                "adaptations_count": len(workflow.current_adaptations),
                "history_size": len(workflow.adaptation_history)
            }
        
        # Calculate trigger statistics
        trigger_stats = {}
        for trigger_id in self.smart_triggers:
            session = self.automation_sessions.get(trigger_id, {})
            trigger_stats[trigger_id] = {
                "activation_count": session.get("activation_count", 0),
                "last_activation": session.get("last_activation", "never")
            }
        
        return {
            "system_overview": {
                "total_smart_triggers": total_triggers,
                "total_adaptive_workflows": total_workflows,
                "total_decision_nodes": len(self.decision_nodes),
                "context_history_size": total_contexts,
                "learning_enabled": self.learning_enabled
            },
            "workflow_statistics": workflow_stats,
            "trigger_statistics": trigger_stats,
            "pattern_insights": self.automation_sessions.get("pattern_insights", {}),
            "timestamp": datetime.now(UTC).isoformat()
        }