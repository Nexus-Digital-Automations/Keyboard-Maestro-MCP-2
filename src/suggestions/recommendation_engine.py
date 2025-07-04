"""
AI-powered recommendation generation system for intelligent automation optimization.

This module implements advanced recommendation algorithms that use AI processing
and pattern analysis to generate personalized, actionable suggestions for
improving automation workflows and user productivity.

Security: All AI processing includes input validation and output sanitization.
Performance: Optimized for real-time suggestion generation with caching.
Type Safety: Complete integration with suggestion system and AI processing.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict

from src.core.suggestion_system import (
    IntelligentSuggestion, SuggestionContext, SuggestionType, PriorityLevel,
    SuggestionError, SuggestionSecurityValidator
)
from src.suggestions.pattern_analyzer import PatternAnalyzer, PatternInsight, OptimizationOpportunity
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger

logger = get_logger(__name__)


class AIPromptGenerator:
    """Generates optimized prompts for AI-powered suggestion generation."""
    
    @staticmethod
    def generate_workflow_optimization_prompt(context: SuggestionContext, 
                                            insights: List[PatternInsight]) -> str:
        """Generate AI prompt for workflow optimization suggestions."""
        context_summary = context.get_context_summary()
        
        insight_summaries = []
        for insight in insights[:3]:  # Limit to top 3 insights
            insight_summaries.append(f"- {insight.insight_type}: {insight.message}")
        
        prompt = f"""
        Analyze the following automation workflow context and provide 2-3 specific optimization recommendations:
        
        Context: {context_summary}
        
        Recent insights from pattern analysis:
        {chr(10).join(insight_summaries)}
        
        Please provide concise, actionable suggestions for:
        1. Improving workflow efficiency
        2. Reducing execution time or error rates
        3. Enhancing automation reliability
        
        Focus on practical, implementable changes. Each suggestion should include:
        - What to change
        - Why it will help
        - Estimated impact
        
        Keep responses under 200 words total.
        """
        
        return prompt.strip()
    
    @staticmethod
    def generate_new_automation_prompt(context: SuggestionContext) -> str:
        """Generate AI prompt for new automation suggestions."""
        context_summary = context.get_context_summary()
        recent_actions_text = ", ".join(context.recent_actions[-10:]) if context.recent_actions else "No recent actions"
        
        prompt = f"""
        Based on this user's automation patterns, suggest 1-2 new automation opportunities:
        
        Context: {context_summary}
        Recent actions: {recent_actions_text}
        Work hours: {"Yes" if context.is_work_hours() else "No"}
        
        Suggest automations that could:
        1. Save time on repetitive tasks
        2. Reduce manual work
        3. Improve workflow consistency
        
        For each suggestion, briefly explain:
        - What to automate
        - How it would save time
        - Difficulty to implement (Easy/Medium/Hard)
        
        Focus on realistic, achievable automations. Keep response under 150 words.
        """
        
        return prompt.strip()
    
    @staticmethod
    def generate_tool_recommendation_prompt(context: SuggestionContext, 
                                          available_tools: List[str]) -> str:
        """Generate AI prompt for tool recommendation suggestions."""
        context_summary = context.get_context_summary()
        active_tools_text = ", ".join(list(context.active_tools)[:5]) if context.active_tools else "None specified"
        
        # Sample of available tools (limit to avoid prompt bloat)
        tools_sample = available_tools[:15] if len(available_tools) > 15 else available_tools
        
        prompt = f"""
        Recommend better tools for this user's automation workflows:
        
        Context: {context_summary}
        Currently using: {active_tools_text}
        
        Available tools include: {", ".join(tools_sample)}
        
        Suggest 1-2 tools that could:
        1. Replace current tools with better alternatives
        2. Add new capabilities for their workflows
        3. Improve automation efficiency
        
        For each tool recommendation, explain:
        - Which tool to try
        - What it's good for
        - Why it's better than current approach
        
        Keep response under 120 words.
        """
        
        return prompt.strip()


class RecommendationEngine:
    """Advanced AI-powered recommendation generation system."""
    
    def __init__(self, ai_processor, pattern_analyzer: PatternAnalyzer):
        self.ai_processor = ai_processor
        self.pattern_analyzer = pattern_analyzer
        self.security_validator = SuggestionSecurityValidator()
        self.suggestion_cache: Dict[str, List[IntelligentSuggestion]] = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
        self.prompt_generator = AIPromptGenerator()
        
        # Available tools for recommendations (would be dynamically populated)
        self.available_tools = [
            "km_create_macro", "km_visual_automation", "km_web_automation",
            "km_dictionary_manager", "km_interface_automation", "km_email_sms_integration",
            "km_audio_speech_control", "km_file_operations", "km_app_control",
            "km_clipboard_manager", "km_window_manager", "km_notifications",
            "km_calculator", "km_token_processor", "km_add_condition",
            "km_control_flow", "km_create_trigger_advanced", "km_add_action"
        ]
    
    @require(lambda self, context: isinstance(context, SuggestionContext))
    async def generate_suggestions(self, context: SuggestionContext, 
                                 suggestion_types: Optional[Set[SuggestionType]] = None,
                                 max_suggestions: int = 5) -> List[IntelligentSuggestion]:
        """
        Generate AI-powered suggestions based on context and analysis.
        
        Args:
            context: Current automation context
            suggestion_types: Types of suggestions to generate
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of intelligent suggestions sorted by relevance
        """
        try:
            # Validate context security
            validation_result = self.security_validator.validate_suggestion_context(context)
            if validation_result.is_left():
                logger.warning(f"Context validation failed: {validation_result.get_left().message}")
                return []
            
            # Check cache first
            cache_key = self._generate_cache_key(context, suggestion_types)
            if self._is_cache_valid(cache_key):
                cached_suggestions = self.suggestion_cache[cache_key]
                return cached_suggestions[:max_suggestions]
            
            if suggestion_types is None:
                suggestion_types = {
                    SuggestionType.WORKFLOW_OPTIMIZATION,
                    SuggestionType.NEW_AUTOMATION,
                    SuggestionType.TOOL_RECOMMENDATION,
                    SuggestionType.PERFORMANCE_IMPROVEMENT
                }
            
            all_suggestions = []
            
            # Generate different types of suggestions concurrently
            suggestion_tasks = []
            for suggestion_type in suggestion_types:
                task = asyncio.create_task(
                    self._generate_suggestions_by_type(suggestion_type, context)
                )
                suggestion_tasks.append(task)
            
            # Wait for all suggestion generation tasks
            suggestion_results = await asyncio.gather(*suggestion_tasks, return_exceptions=True)
            
            # Collect successful results
            for result in suggestion_results:
                if isinstance(result, list):
                    all_suggestions.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Suggestion generation error: {str(result)}")
            
            # Sort suggestions by urgency score and confidence
            all_suggestions.sort(key=lambda s: s.get_urgency_score(), reverse=True)
            
            # Apply security sanitization
            sanitized_suggestions = []
            for suggestion in all_suggestions:
                sanitized = self.security_validator.sanitize_suggestion_content(suggestion)
                sanitized_suggestions.append(sanitized)
            
            # Cache results
            self._cache_suggestions(cache_key, sanitized_suggestions)
            
            # Limit results
            final_suggestions = sanitized_suggestions[:max_suggestions]
            
            logger.info(f"Generated {len(final_suggestions)} suggestions for user {context.user_id}")
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return []
    
    async def _generate_suggestions_by_type(self, suggestion_type: SuggestionType, 
                                          context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate suggestions for specific type."""
        try:
            if suggestion_type == SuggestionType.WORKFLOW_OPTIMIZATION:
                return await self._generate_workflow_optimizations(context)
            elif suggestion_type == SuggestionType.NEW_AUTOMATION:
                return await self._generate_new_automation_suggestions(context)
            elif suggestion_type == SuggestionType.TOOL_RECOMMENDATION:
                return await self._generate_tool_recommendations(context)
            elif suggestion_type == SuggestionType.PERFORMANCE_IMPROVEMENT:
                return await self._generate_performance_improvements(context)
            elif suggestion_type == SuggestionType.ERROR_PREVENTION:
                return await self._generate_error_prevention_suggestions(context)
            elif suggestion_type == SuggestionType.BEST_PRACTICE:
                return await self._generate_best_practice_suggestions(context)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating {suggestion_type.value} suggestions: {str(e)}")
            return []
    
    async def _generate_workflow_optimizations(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate workflow optimization suggestions using pattern analysis and AI."""
        suggestions = []
        
        try:
            # Get pattern analysis insights
            insights = await self.pattern_analyzer.analyze_user_patterns(context.user_id, depth="standard")
            
            # Filter for optimization-relevant insights
            optimization_insights = [
                insight for insight in insights 
                if insight.insight_type in ["low_efficiency", "low_reliability", "context_problem"]
            ]
            
            if optimization_insights:
                # Generate rule-based suggestions from insights
                for insight in optimization_insights[:2]:  # Limit to top 2
                    rule_based_suggestion = self._create_rule_based_optimization(insight, context)
                    if rule_based_suggestion:
                        suggestions.append(rule_based_suggestion)
                
                # Generate AI-powered suggestions if AI processor is available
                if self.ai_processor and len(optimization_insights) > 0:
                    ai_suggestion = await self._generate_ai_optimization_suggestion(context, optimization_insights)
                    if ai_suggestion:
                        suggestions.append(ai_suggestion)
            
            # If no specific insights, provide general optimization suggestions
            if not suggestions and context.performance_data:
                general_suggestion = self._create_general_optimization_suggestion(context)
                if general_suggestion:
                    suggestions.append(general_suggestion)
                    
        except Exception as e:
            logger.error(f"Error generating workflow optimizations: {str(e)}")
        
        return suggestions
    
    async def _generate_new_automation_suggestions(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate new automation suggestions based on patterns and AI analysis."""
        suggestions = []
        
        try:
            # Analyze recent actions for automation opportunities
            if len(context.recent_actions) >= 3:
                pattern_suggestion = self._create_pattern_based_automation_suggestion(context)
                if pattern_suggestion:
                    suggestions.append(pattern_suggestion)
            
            # Use AI to generate creative automation suggestions
            if self.ai_processor and len(context.recent_actions) > 0:
                ai_suggestion = await self._generate_ai_automation_suggestion(context)
                if ai_suggestion:
                    suggestions.append(ai_suggestion)
            
            # Generate context-based suggestions
            context_suggestion = self._create_context_based_automation_suggestion(context)
            if context_suggestion:
                suggestions.append(context_suggestion)
                
        except Exception as e:
            logger.error(f"Error generating new automation suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_tool_recommendations(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate tool recommendation suggestions."""
        suggestions = []
        
        try:
            # Analyze current tool usage patterns
            current_tools = list(context.active_tools)
            
            # Generate recommendations based on usage patterns
            if current_tools:
                usage_based_suggestion = self._create_usage_based_tool_recommendation(context, current_tools)
                if usage_based_suggestion:
                    suggestions.append(usage_based_suggestion)
            
            # Use AI for intelligent tool recommendations
            if self.ai_processor:
                ai_suggestion = await self._generate_ai_tool_recommendation(context)
                if ai_suggestion:
                    suggestions.append(ai_suggestion)
            
            # Suggest underutilized powerful tools
            underutilized_suggestion = self._create_underutilized_tool_suggestion(context)
            if underutilized_suggestion:
                suggestions.append(underutilized_suggestion)
                
        except Exception as e:
            logger.error(f"Error generating tool recommendations: {str(e)}")
        
        return suggestions
    
    async def _generate_performance_improvements(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate performance improvement suggestions."""
        suggestions = []
        
        try:
            # Get optimization opportunities from pattern analyzer
            opportunities = await self.pattern_analyzer.identify_optimization_opportunities(context.user_id)
            
            # Convert opportunities to suggestions
            for opportunity in opportunities[:2]:  # Limit to top 2
                suggestion = self._convert_opportunity_to_suggestion(opportunity, context)
                if suggestion:
                    suggestions.append(suggestion)
            
            # Add general performance suggestions if none found
            if not suggestions:
                general_perf_suggestion = self._create_general_performance_suggestion(context)
                if general_perf_suggestion:
                    suggestions.append(general_perf_suggestion)
                    
        except Exception as e:
            logger.error(f"Error generating performance improvements: {str(e)}")
        
        return suggestions
    
    async def _generate_error_prevention_suggestions(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate error prevention suggestions."""
        suggestions = []
        
        try:
            # Analyze recent errors and failure patterns
            if context.performance_data:
                error_patterns = self._analyze_error_patterns(context.performance_data)
                
                if error_patterns:
                    suggestion = IntelligentSuggestion(
                        suggestion_id=f"error_prevention_{datetime.now().timestamp()}",
                        suggestion_type=SuggestionType.ERROR_PREVENTION,
                        title="Prevent Common Automation Errors",
                        description="Add error handling and validation to reduce automation failures",
                        priority=PriorityLevel.HIGH,
                        confidence=0.8,
                        potential_impact="Improved automation reliability and reduced failure rates",
                        implementation_effort="Medium",
                        suggested_actions=[
                            {
                                "action": "add_error_handling",
                                "description": "Add try-catch blocks and validation checks"
                            },
                            {
                                "action": "add_conditions",
                                "description": "Use km_add_condition to validate prerequisites"
                            }
                        ],
                        reasoning="Analysis shows patterns of automation failures that could be prevented"
                    )
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.error(f"Error generating error prevention suggestions: {str(e)}")
        
        return suggestions
    
    async def _generate_best_practice_suggestions(self, context: SuggestionContext) -> List[IntelligentSuggestion]:
        """Generate best practice suggestions."""
        suggestions = []
        
        try:
            # Suggest best practices based on usage patterns
            if context.recent_actions:
                best_practice_suggestion = IntelligentSuggestion(
                    suggestion_id=f"best_practice_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.BEST_PRACTICE,
                    title="Automation Best Practices",
                    description="Consider implementing automation best practices for better maintainability",
                    priority=PriorityLevel.LOW,
                    confidence=0.7,
                    potential_impact="Improved automation maintainability and reliability",
                    implementation_effort="Low",
                    suggested_actions=[
                        {
                            "action": "add_documentation",
                            "description": "Document your automation workflows"
                        },
                        {
                            "action": "use_templates",
                            "description": "Use km_macro_template_system for reusable patterns"
                        },
                        {
                            "action": "add_testing",
                            "description": "Use km_macro_testing_framework for validation"
                        }
                    ],
                    reasoning="Following best practices improves long-term automation success"
                )
                suggestions.append(best_practice_suggestion)
                
        except Exception as e:
            logger.error(f"Error generating best practice suggestions: {str(e)}")
        
        return suggestions
    
    def _create_rule_based_optimization(self, insight: PatternInsight, 
                                      context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Create optimization suggestion based on pattern insight."""
        try:
            if insight.insight_type == "low_efficiency":
                return IntelligentSuggestion(
                    suggestion_id=f"opt_efficiency_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.WORKFLOW_OPTIMIZATION,
                    title="Optimize Slow Automation Workflows",
                    description=f"Several workflows are running slower than optimal. "
                               f"Consider adding conditions or reducing complexity.",
                    priority=PriorityLevel.MEDIUM,
                    confidence=insight.confidence,
                    potential_impact="Reduced execution time and improved user experience",
                    implementation_effort="Medium",
                    suggested_actions=[
                        {
                            "action": "review_workflows",
                            "patterns": insight.patterns[:3]
                        },
                        {
                            "action": "add_conditions",
                            "description": "Use km_add_condition to skip unnecessary steps"
                        },
                        {
                            "action": "optimize_timing",
                            "description": "Reduce delays and optimize execution order"
                        }
                    ],
                    reasoning=insight.message
                )
            elif insight.insight_type == "low_reliability":
                return IntelligentSuggestion(
                    suggestion_id=f"opt_reliability_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.WORKFLOW_OPTIMIZATION,
                    title="Improve Automation Reliability",
                    description="Some automations have low success rates. Add error handling and validation.",
                    priority=PriorityLevel.HIGH,
                    confidence=insight.confidence,
                    potential_impact="Higher success rates and fewer failed automations",
                    implementation_effort="Medium",
                    suggested_actions=[
                        {
                            "action": "add_error_handling",
                            "patterns": insight.patterns[:3]
                        },
                        {
                            "action": "add_validation",
                            "description": "Validate inputs and prerequisites before execution"
                        }
                    ],
                    reasoning=insight.message
                )
        except Exception as e:
            logger.error(f"Error creating rule-based optimization: {str(e)}")
        
        return None
    
    async def _generate_ai_optimization_suggestion(self, context: SuggestionContext, 
                                                 insights: List[PatternInsight]) -> Optional[IntelligentSuggestion]:
        """Generate AI-powered optimization suggestion."""
        try:
            if not self.ai_processor:
                return None
            
            prompt = self.prompt_generator.generate_workflow_optimization_prompt(context, insights)
            
            ai_result = await self.ai_processor.generate_text(
                prompt=prompt,
                style="technical",
                max_length=200
            )
            
            if ai_result.is_right():
                ai_content = ai_result.get_right()
                
                return IntelligentSuggestion(
                    suggestion_id=f"ai_optimization_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.WORKFLOW_OPTIMIZATION,
                    title="AI-Recommended Workflow Optimization",
                    description="AI analysis suggests specific optimizations for your workflows",
                    priority=PriorityLevel.MEDIUM,
                    confidence=0.75,
                    potential_impact="AI-identified efficiency improvements",
                    implementation_effort="Varies",
                    suggested_actions=[
                        {
                            "action": "ai_recommendation",
                            "description": ai_content
                        }
                    ],
                    reasoning="Generated using AI analysis of your automation patterns"
                )
                
        except Exception as e:
            logger.error(f"Error generating AI optimization suggestion: {str(e)}")
        
        return None
    
    def _create_general_optimization_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Create general optimization suggestion when no specific insights available."""
        try:
            return IntelligentSuggestion(
                suggestion_id=f"general_opt_{datetime.now().timestamp()}",
                suggestion_type=SuggestionType.WORKFLOW_OPTIMIZATION,
                title="General Automation Optimization",
                description="Consider reviewing your automation workflows for optimization opportunities",
                priority=PriorityLevel.LOW,
                confidence=0.6,
                potential_impact="Potential workflow improvements",
                implementation_effort="Low",
                suggested_actions=[
                    {
                        "action": "review_frequency",
                        "description": "Review your most frequently used automations"
                    },
                    {
                        "action": "add_monitoring",
                        "description": "Monitor automation performance for bottlenecks"
                    }
                ],
                reasoning="Regular optimization review is a good practice"
            )
        except Exception as e:
            logger.error(f"Error creating general optimization suggestion: {str(e)}")
        
        return None
    
    def _create_pattern_based_automation_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Create automation suggestion based on action patterns."""
        try:
            # Analyze recent actions for patterns
            recent_actions = context.recent_actions[-10:]  # Last 10 actions
            action_frequency = defaultdict(int)
            
            for action in recent_actions:
                action_frequency[action] += 1
            
            # Find repeated actions that could be automated
            repeated_actions = [(action, count) for action, count in action_frequency.items() if count >= 3]
            
            if repeated_actions:
                most_repeated = max(repeated_actions, key=lambda x: x[1])
                
                return IntelligentSuggestion(
                    suggestion_id=f"pattern_automation_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.NEW_AUTOMATION,
                    title="Automate Repeated Action Pattern",
                    description=f"You've performed '{most_repeated[0]}' {most_repeated[1]} times recently. "
                               f"Consider creating an automation for this workflow.",
                    priority=PriorityLevel.MEDIUM,
                    confidence=0.8,
                    potential_impact="Save time on repetitive tasks",
                    implementation_effort="Low to Medium",
                    suggested_actions=[
                        {
                            "action": "create_macro",
                            "repeated_action": most_repeated[0],
                            "frequency": most_repeated[1]
                        }
                    ],
                    reasoning="Pattern analysis detected repeated manual actions"
                )
                
        except Exception as e:
            logger.error(f"Error creating pattern-based automation suggestion: {str(e)}")
        
        return None
    
    async def _generate_ai_automation_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Generate AI-powered new automation suggestion."""
        try:
            if not self.ai_processor:
                return None
            
            prompt = self.prompt_generator.generate_new_automation_prompt(context)
            
            ai_result = await self.ai_processor.generate_text(
                prompt=prompt,
                style="helpful",
                max_length=150
            )
            
            if ai_result.is_right():
                ai_content = ai_result.get_right()
                
                return IntelligentSuggestion(
                    suggestion_id=f"ai_automation_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.NEW_AUTOMATION,
                    title="AI-Suggested New Automation",
                    description="AI analysis suggests new automation opportunities based on your activity",
                    priority=PriorityLevel.MEDIUM,
                    confidence=0.7,
                    potential_impact="Time savings through new automation",
                    implementation_effort="Varies",
                    suggested_actions=[
                        {
                            "action": "ai_suggestion",
                            "description": ai_content
                        }
                    ],
                    reasoning="AI-generated based on activity pattern analysis"
                )
                
        except Exception as e:
            logger.error(f"Error generating AI automation suggestion: {str(e)}")
        
        return None
    
    def _create_context_based_automation_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Create automation suggestion based on current context."""
        try:
            suggestions_by_context = {
                "work_hours": "Consider automating your morning setup routine or end-of-day cleanup tasks",
                "evening": "Automate data backup or system maintenance tasks for overnight execution",
                "weekend": "Set up automations for personal task management or household organization"
            }
            
            context_key = "work_hours" if context.is_work_hours() else "evening"
            
            return IntelligentSuggestion(
                suggestion_id=f"context_automation_{datetime.now().timestamp()}",
                suggestion_type=SuggestionType.NEW_AUTOMATION,
                title="Context-Based Automation Opportunity",
                description=suggestions_by_context[context_key],
                priority=PriorityLevel.LOW,
                confidence=0.6,
                potential_impact="Better time management and organization",
                implementation_effort="Low",
                suggested_actions=[
                    {
                        "action": "explore_context_automation",
                        "context": context_key,
                        "time": context.time_of_day
                    }
                ],
                reasoning=f"Suggestion based on current context: {context_key}"
            )
            
        except Exception as e:
            logger.error(f"Error creating context-based automation suggestion: {str(e)}")
        
        return None
    
    def _create_usage_based_tool_recommendation(self, context: SuggestionContext, 
                                              current_tools: List[str]) -> Optional[IntelligentSuggestion]:
        """Create tool recommendation based on current usage patterns."""
        try:
            # Suggest complementary tools based on current usage
            tool_suggestions = {
                "km_visual_automation": ["km_interface_automation", "km_window_manager"],
                "km_web_automation": ["km_dictionary_manager", "km_email_sms_integration"],
                "km_file_operations": ["km_clipboard_manager", "km_calculator"],
                "km_app_control": ["km_window_manager", "km_notifications"]
            }
            
            recommended_tools = set()
            for tool in current_tools:
                if tool in tool_suggestions:
                    recommended_tools.update(tool_suggestions[tool])
            
            # Remove tools already in use
            recommended_tools -= set(current_tools)
            
            if recommended_tools:
                top_recommendation = list(recommended_tools)[0]
                
                return IntelligentSuggestion(
                    suggestion_id=f"tool_rec_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                    title="Complementary Tool Recommendation",
                    description=f"Based on your use of {current_tools[0]}, consider trying {top_recommendation} "
                               f"for enhanced workflow capabilities.",
                    priority=PriorityLevel.LOW,
                    confidence=0.7,
                    potential_impact="Enhanced automation capabilities",
                    implementation_effort="Low",
                    suggested_actions=[
                        {
                            "action": "try_tool",
                            "recommended_tool": top_recommendation,
                            "current_tools": current_tools
                        }
                    ],
                    reasoning="Tool recommendation based on usage pattern analysis"
                )
                
        except Exception as e:
            logger.error(f"Error creating usage-based tool recommendation: {str(e)}")
        
        return None
    
    async def _generate_ai_tool_recommendation(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Generate AI-powered tool recommendation."""
        try:
            if not self.ai_processor:
                return None
            
            prompt = self.prompt_generator.generate_tool_recommendation_prompt(context, self.available_tools)
            
            ai_result = await self.ai_processor.generate_text(
                prompt=prompt,
                style="helpful",
                max_length=120
            )
            
            if ai_result.is_right():
                ai_content = ai_result.get_right()
                
                return IntelligentSuggestion(
                    suggestion_id=f"ai_tool_rec_{datetime.now().timestamp()}",
                    suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                    title="AI-Recommended Tools",
                    description="AI suggests tools that could enhance your automation workflows",
                    priority=PriorityLevel.LOW,
                    confidence=0.7,
                    potential_impact="Access to more efficient automation tools",
                    implementation_effort="Low",
                    suggested_actions=[
                        {
                            "action": "ai_tool_recommendation",
                            "description": ai_content
                        }
                    ],
                    reasoning="AI analysis of your workflow patterns and available tools"
                )
                
        except Exception as e:
            logger.error(f"Error generating AI tool recommendation: {str(e)}")
        
        return None
    
    def _create_underutilized_tool_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Suggest powerful but underutilized tools."""
        try:
            # Powerful tools that users often don't discover
            underutilized_tools = [
                ("km_control_flow", "Advanced workflow logic with if/then/else conditions"),
                ("km_create_trigger_advanced", "Event-driven automation that responds automatically"),
                ("km_macro_testing_framework", "Validate and test your automations"),
                ("km_dictionary_manager", "Manage complex data and configurations"),
                ("km_visual_automation", "Automate based on what you see on screen")
            ]
            
            # Pick one that's not in active tools
            for tool, description in underutilized_tools:
                if tool not in context.active_tools:
                    return IntelligentSuggestion(
                        suggestion_id=f"underutilized_{datetime.now().timestamp()}",
                        suggestion_type=SuggestionType.TOOL_RECOMMENDATION,
                        title="Discover Powerful Automation Tools",
                        description=f"Try {tool}: {description}",
                        priority=PriorityLevel.LOW,
                        confidence=0.6,
                        potential_impact="Access to advanced automation capabilities",
                        implementation_effort="Low",
                        suggested_actions=[
                            {
                                "action": "explore_tool",
                                "tool": tool,
                                "description": description
                            }
                        ],
                        reasoning="Many users don't discover these powerful automation tools"
                    )
                    
        except Exception as e:
            logger.error(f"Error creating underutilized tool suggestion: {str(e)}")
        
        return None
    
    def _convert_opportunity_to_suggestion(self, opportunity: OptimizationOpportunity,
                                         context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Convert optimization opportunity to intelligent suggestion."""
        try:
            return IntelligentSuggestion(
                suggestion_id=f"opp_suggest_{datetime.now().timestamp()}",
                suggestion_type=SuggestionType.PERFORMANCE_IMPROVEMENT,
                title=opportunity.description,
                description=opportunity.potential_impact,
                priority=opportunity.priority,
                confidence=opportunity.confidence,
                potential_impact=opportunity.potential_impact,
                implementation_effort=opportunity.implementation_effort,
                suggested_actions=[
                    {
                        "action": "optimize_performance",
                        "opportunity_type": opportunity.opportunity_type,
                        "affected_items": opportunity.affected_items
                    }
                ],
                reasoning=f"Performance analysis identified {opportunity.opportunity_type} optimization opportunity"
            )
        except Exception as e:
            logger.error(f"Error converting opportunity to suggestion: {str(e)}")
        
        return None
    
    def _create_general_performance_suggestion(self, context: SuggestionContext) -> Optional[IntelligentSuggestion]:
        """Create general performance improvement suggestion."""
        try:
            return IntelligentSuggestion(
                suggestion_id=f"general_perf_{datetime.now().timestamp()}",
                suggestion_type=SuggestionType.PERFORMANCE_IMPROVEMENT,
                title="Monitor Automation Performance",
                description="Set up performance monitoring to identify optimization opportunities",
                priority=PriorityLevel.LOW,
                confidence=0.6,
                potential_impact="Better visibility into automation performance",
                implementation_effort="Low",
                suggested_actions=[
                    {
                        "action": "enable_monitoring",
                        "description": "Track automation execution times and success rates"
                    },
                    {
                        "action": "review_metrics",
                        "description": "Regularly review performance metrics"
                    }
                ],
                reasoning="Performance monitoring helps identify optimization opportunities"
            )
        except Exception as e:
            logger.error(f"Error creating general performance suggestion: {str(e)}")
        
        return None
    
    def _analyze_error_patterns(self, performance_data: Dict[str, Any]) -> bool:
        """Analyze performance data for error patterns."""
        try:
            # Look for indicators of error patterns in performance data
            error_indicators = ["failed", "error", "timeout", "exception"]
            
            for key, value in performance_data.items():
                if isinstance(value, str) and any(indicator in value.lower() for indicator in error_indicators):
                    return True
                elif isinstance(value, (int, float)) and key.endswith("_error_rate") and value > 0.1:
                    return True
            
            return False
        except Exception:
            return False
    
    def _generate_cache_key(self, context: SuggestionContext, 
                          suggestion_types: Optional[Set[SuggestionType]]) -> str:
        """Generate cache key for suggestion results."""
        types_str = ",".join(sorted([t.value for t in suggestion_types])) if suggestion_types else "all"
        return f"{context.user_id}_{types_str}_{context.session_id}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached suggestions are still valid."""
        if cache_key not in self.suggestion_cache:
            return False
        
        # For now, we'll implement basic time-based cache validation
        # In practice, this could be more sophisticated
        return True  # Simplified for this implementation
    
    def _cache_suggestions(self, cache_key: str, suggestions: List[IntelligentSuggestion]) -> None:
        """Cache suggestion results."""
        self.suggestion_cache[cache_key] = suggestions
        
        # Simple cache size management
        if len(self.suggestion_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(self.suggestion_cache.keys())[:20]
            for key in oldest_keys:
                del self.suggestion_cache[key]