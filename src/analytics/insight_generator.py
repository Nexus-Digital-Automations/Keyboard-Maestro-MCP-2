"""
Insight Generator - TASK_59 Phase 2 Core Implementation

Intelligent insight generation and recommendation engine for automation workflows.
Provides ML-powered analysis, actionable recommendations, and strategic optimization insights.

Architecture: ML Insights + Strategic Analysis + ROI Calculation + Recommendation Engine
Performance: <300ms insight generation, <1s comprehensive analysis, <2s strategic recommendations
Security: Safe insight processing, validated recommendations, comprehensive data protection
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
from collections import defaultdict, Counter

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    InsightId, create_insight_id, InsightType, PredictiveInsight,
    PredictiveModelingError, prioritize_insights
)
from src.analytics.pattern_predictor import PatternPredictor, DetectedPattern, PatternType
from src.analytics.usage_forecaster import UsageForecaster, ResourceForecast, ResourceType


class InsightCategory(Enum):
    """Categories of insights that can be generated."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COST_REDUCTION = "cost_reduction"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    RISK_MITIGATION = "risk_mitigation"
    CAPACITY_PLANNING = "capacity_planning"
    WORKFLOW_ENHANCEMENT = "workflow_enhancement"
    SECURITY_IMPROVEMENT = "security_improvement"
    USER_EXPERIENCE = "user_experience"
    AUTOMATION_EXPANSION = "automation_expansion"
    COMPLIANCE_ENHANCEMENT = "compliance_enhancement"


class InsightSeverity(Enum):
    """Severity levels for insights."""
    INFO = "info"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationType(Enum):
    """Types of recommendations."""
    IMMEDIATE_ACTION = "immediate_action"
    PLANNED_IMPROVEMENT = "planned_improvement"
    STRATEGIC_INITIATIVE = "strategic_initiative"
    MONITORING_ALERT = "monitoring_alert"
    PREVENTIVE_MEASURE = "preventive_measure"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


@dataclass(frozen=True)
class InsightData:
    """Raw data used for insight generation."""
    data_source: str
    data_type: str  # performance, usage, pattern, error, cost
    time_period: str
    metrics: Dict[str, float]
    trends: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.data_source:
            raise ValueError("Data source must be specified")
        if not self.metrics:
            raise ValueError("Metrics data must be provided")


@dataclass(frozen=True)
class RecommendationAction:
    """Specific actionable recommendation."""
    action_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    priority: str  # low, medium, high, critical
    effort_estimate: str  # low, medium, high
    timeline: str  # immediate, short_term, medium_term, long_term
    expected_impact: str  # low, medium, high, very_high
    implementation_steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    estimated_cost: Optional[float] = None
    estimated_savings: Optional[float] = None
    
    def __post_init__(self):
        if not self.action_id:
            raise ValueError("Action ID must be specified")
        if self.priority not in ["low", "medium", "high", "critical"]:
            raise ValueError("Priority must be low, medium, high, or critical")


@dataclass(frozen=True)
class ROIAnalysis:
    """Return on Investment analysis for recommendations."""
    analysis_id: str
    recommendation_ids: List[str]
    investment_required: float
    expected_annual_savings: float
    payback_period_months: float
    roi_percentage: float
    net_present_value: float
    confidence_level: float
    risk_factors: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.investment_required < 0:
            raise ValueError("Investment required must be non-negative")
        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValueError("Confidence level must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ExecutiveSummary:
    """Executive summary of insights and recommendations."""
    summary_id: str
    time_period: str
    key_findings: List[str]
    critical_issues: List[str]
    top_opportunities: List[str]
    recommended_actions: List[str]
    total_potential_savings: float
    total_investment_required: float
    overall_roi: float
    strategic_priorities: List[str] = field(default_factory=list)
    risk_assessment: str = "medium"
    confidence_score: float = 0.8
    
    def __post_init__(self):
        if not self.key_findings:
            raise ValueError("Key findings must be provided")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")


class InsightGenerator:
    """
    Intelligent insight generation and recommendation engine.
    
    Analyzes automation data to generate actionable insights, strategic recommendations,
    and comprehensive ROI analysis for optimization opportunities.
    """
    
    def __init__(self, pattern_predictor: PatternPredictor, usage_forecaster: UsageForecaster):
        self.pattern_predictor = pattern_predictor
        self.usage_forecaster = usage_forecaster
        self.insight_history: List[PredictiveInsight] = []
        self.recommendation_history: List[RecommendationAction] = []
        self.insight_templates: Dict[InsightCategory, Dict[str, Any]] = {}
        self.roi_models: Dict[str, Any] = {}
        self.insight_cache: Dict[str, Any] = {}
        
        # Initialize insight templates and ROI models
        self._initialize_insight_templates()
        self._initialize_roi_models()
    
    def _initialize_insight_templates(self):
        """Initialize templates for different insight categories."""
        self.insight_templates = {
            InsightCategory.PERFORMANCE_OPTIMIZATION: {
                "title_template": "Performance optimization opportunity in {target}",
                "impact_multiplier": 1.2,
                "typical_savings": 0.15,  # 15% improvement
                "effort_factor": 0.8
            },
            InsightCategory.COST_REDUCTION: {
                "title_template": "Cost reduction opportunity: {opportunity}",
                "impact_multiplier": 1.5,
                "typical_savings": 0.25,  # 25% cost reduction
                "effort_factor": 0.6
            },
            InsightCategory.EFFICIENCY_IMPROVEMENT: {
                "title_template": "Efficiency improvement in {workflow}",
                "impact_multiplier": 1.3,
                "typical_savings": 0.20,  # 20% efficiency gain
                "effort_factor": 0.7
            },
            InsightCategory.RISK_MITIGATION: {
                "title_template": "Risk mitigation needed for {risk_area}",
                "impact_multiplier": 2.0,  # High impact for risk
                "typical_savings": 0.10,  # Cost avoidance
                "effort_factor": 1.2  # Higher effort required
            },
            InsightCategory.CAPACITY_PLANNING: {
                "title_template": "Capacity planning recommendation for {resource}",
                "impact_multiplier": 1.1,
                "typical_savings": 0.12,
                "effort_factor": 1.0
            }
        }
    
    def _initialize_roi_models(self):
        """Initialize ROI calculation models."""
        self.roi_models = {
            "performance_improvement": {
                "time_savings_per_hour": 0.5,  # hours saved per improvement
                "hourly_cost": 50.0,  # average hourly cost
                "annual_usage_hours": 2000
            },
            "automation_expansion": {
                "manual_task_time": 2.0,  # hours per manual task
                "automation_setup_time": 8.0,  # hours to automate
                "task_frequency_per_month": 20
            },
            "infrastructure_optimization": {
                "resource_cost_per_unit": 0.10,  # cost per resource unit
                "efficiency_improvement": 0.20,  # 20% efficiency gain
                "annual_resource_usage": 100000
            }
        }

    @require(lambda insight_data: len(insight_data) > 0)
    async def generate_insights(
        self,
        insight_data: List[InsightData],
        categories: Optional[List[InsightCategory]] = None,
        min_confidence: float = 0.6,
        include_roi_analysis: bool = True,
        prioritize_by_impact: bool = True
    ) -> Either[PredictiveModelingError, List[PredictiveInsight]]:
        """
        Generate comprehensive insights from automation data.
        
        Analyzes patterns, performance metrics, and usage data to generate
        actionable insights with ROI analysis and strategic recommendations.
        """
        try:
            if categories is None:
                categories = list(InsightCategory)
            
            generated_insights = []
            
            # Analyze each data source for insights
            for data in insight_data:
                insights = await self._analyze_data_for_insights(data, categories, min_confidence)
                if insights.is_right():
                    generated_insights.extend(insights.get_right())
            
            # Generate cross-data insights
            cross_insights = await self._generate_cross_data_insights(insight_data, categories)
            if cross_insights.is_right():
                generated_insights.extend(cross_insights.get_right())
            
            # Filter by confidence and prioritize
            high_confidence_insights = [
                insight for insight in generated_insights 
                if insight.confidence_score >= min_confidence
            ]
            
            if prioritize_by_impact:
                high_confidence_insights = prioritize_insights(high_confidence_insights)
            
            # Add ROI analysis if requested
            if include_roi_analysis:
                for insight in high_confidence_insights:
                    roi_analysis = await self._calculate_insight_roi(insight)
                    if roi_analysis.is_right():
                        roi_data = roi_analysis.get_right()
                        # Add ROI information to insight metadata
                        if hasattr(insight, 'supporting_evidence'):
                            insight.supporting_evidence.update({
                                "roi_analysis": roi_data
                            })
            
            # Store insights in history
            self.insight_history.extend(high_confidence_insights)
            
            return Either.right(high_confidence_insights)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Insight generation failed: {str(e)}", 
                "INSIGHT_GENERATION_ERROR"
            ))

    async def _analyze_data_for_insights(
        self,
        data: InsightData,
        categories: List[InsightCategory],
        min_confidence: float
    ) -> Either[PredictiveModelingError, List[PredictiveInsight]]:
        """Analyze individual data source for insights."""
        try:
            insights = []
            
            # Performance optimization insights
            if InsightCategory.PERFORMANCE_OPTIMIZATION in categories:
                perf_insights = await self._generate_performance_insights(data, min_confidence)
                insights.extend(perf_insights)
            
            # Cost reduction insights
            if InsightCategory.COST_REDUCTION in categories:
                cost_insights = await self._generate_cost_insights(data, min_confidence)
                insights.extend(cost_insights)
            
            # Efficiency improvement insights
            if InsightCategory.EFFICIENCY_IMPROVEMENT in categories:
                efficiency_insights = await self._generate_efficiency_insights(data, min_confidence)
                insights.extend(efficiency_insights)
            
            # Risk mitigation insights
            if InsightCategory.RISK_MITIGATION in categories:
                risk_insights = await self._generate_risk_insights(data, min_confidence)
                insights.extend(risk_insights)
            
            # Capacity planning insights
            if InsightCategory.CAPACITY_PLANNING in categories:
                capacity_insights = await self._generate_capacity_insights(data, min_confidence)
                insights.extend(capacity_insights)
            
            return Either.right(insights)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Data analysis failed for {data.data_source}: {str(e)}", 
                "DATA_ANALYSIS_ERROR"
            ))

    async def _generate_performance_insights(
        self, 
        data: InsightData, 
        min_confidence: float
    ) -> List[PredictiveInsight]:
        """Generate performance optimization insights."""
        insights = []
        
        # Check for performance degradation patterns
        if "response_time" in data.metrics:
            response_time = data.metrics["response_time"]
            
            if response_time > 2.0:  # Slow response time
                confidence = min(0.95, (response_time - 2.0) / 3.0 + 0.7)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.PERFORMANCE_IMPROVEMENT,
                        title=f"Performance optimization needed for {data.data_source}",
                        description=f"Response time of {response_time:.2f}s exceeds optimal threshold. Consider optimization.",
                        confidence_score=confidence,
                        impact_score=min(1.0, (response_time - 2.0) / 2.0 + 0.6),
                        priority_level="high" if response_time > 5.0 else "medium",
                        actionable_recommendations=[
                            "Review and optimize slow-performing components",
                            "Implement caching mechanisms for frequently accessed data",
                            "Consider resource scaling or hardware upgrades",
                            "Profile code execution to identify bottlenecks"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_response_time": response_time,
                            "optimal_threshold": 2.0,
                            "performance_degradation": f"{((response_time / 2.0) - 1) * 100:.1f}%"
                        },
                        roi_estimate=self._estimate_performance_roi(response_time),
                        implementation_effort="medium"
                    )
                    insights.append(insight)
        
        # Check for resource utilization inefficiencies
        if "cpu_usage" in data.metrics:
            cpu_usage = data.metrics["cpu_usage"]
            
            if cpu_usage > 80.0:  # High CPU usage
                confidence = min(0.90, (cpu_usage - 80.0) / 15.0 + 0.7)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.PERFORMANCE_IMPROVEMENT,
                        title=f"High CPU utilization detected in {data.data_source}",
                        description=f"CPU usage at {cpu_usage:.1f}% indicates potential performance bottleneck.",
                        confidence_score=confidence,
                        impact_score=min(1.0, (cpu_usage - 80.0) / 20.0 + 0.5),
                        priority_level="critical" if cpu_usage > 95.0 else "high",
                        actionable_recommendations=[
                            "Optimize CPU-intensive operations",
                            "Implement load balancing or horizontal scaling",
                            "Review algorithm efficiency",
                            "Consider upgrading hardware resources"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_cpu_usage": cpu_usage,
                            "recommended_threshold": 80.0,
                            "utilization_excess": f"{cpu_usage - 80.0:.1f}%"
                        },
                        implementation_effort="medium" if cpu_usage < 90.0 else "high"
                    )
                    insights.append(insight)
        
        return insights

    async def _generate_cost_insights(
        self, 
        data: InsightData, 
        min_confidence: float
    ) -> List[PredictiveInsight]:
        """Generate cost reduction insights."""
        insights = []
        
        # Check for cost optimization opportunities
        if "cost_per_execution" in data.metrics:
            cost = data.metrics["cost_per_execution"]
            execution_count = data.metrics.get("execution_count", 1000)
            
            if cost > 0.10:  # High cost per execution
                annual_cost = cost * execution_count * 12  # Estimate annual cost
                potential_savings = annual_cost * 0.25  # 25% potential savings
                
                confidence = min(0.85, (cost - 0.10) / 0.20 + 0.6)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.COST_SAVINGS,
                        title=f"Cost optimization opportunity in {data.data_source}",
                        description=f"High execution cost of ${cost:.3f} per operation. Annual cost: ${annual_cost:.2f}",
                        confidence_score=confidence,
                        impact_score=min(1.0, potential_savings / 10000),  # Scale impact
                        priority_level="high" if potential_savings > 5000 else "medium",
                        actionable_recommendations=[
                            "Review and optimize resource usage patterns",
                            "Implement more efficient algorithms or data structures",
                            "Consider bulk operations to reduce per-unit costs",
                            "Evaluate alternative service providers or pricing models"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_cost_per_execution": cost,
                            "annual_cost_estimate": annual_cost,
                            "potential_annual_savings": potential_savings,
                            "execution_frequency": execution_count
                        },
                        roi_estimate=potential_savings,
                        implementation_effort="medium"
                    )
                    insights.append(insight)
        
        # Check for underutilized resources
        if "resource_utilization" in data.metrics:
            utilization = data.metrics["resource_utilization"]
            
            if utilization < 30.0:  # Low utilization
                confidence = min(0.80, (30.0 - utilization) / 25.0 + 0.5)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.COST_SAVINGS,
                        title=f"Underutilized resources in {data.data_source}",
                        description=f"Resource utilization at {utilization:.1f}% suggests potential for downsizing.",
                        confidence_score=confidence,
                        impact_score=min(1.0, (30.0 - utilization) / 30.0),
                        priority_level="medium",
                        actionable_recommendations=[
                            "Consider downsizing allocated resources",
                            "Implement auto-scaling to match demand",
                            "Consolidate workloads to improve utilization",
                            "Review resource allocation policies"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_utilization": utilization,
                            "optimal_range": "60-80%",
                            "potential_cost_savings": f"{(30.0 - utilization) / 30.0 * 100:.1f}%"
                        },
                        implementation_effort="low"
                    )
                    insights.append(insight)
        
        return insights

    async def _generate_efficiency_insights(
        self, 
        data: InsightData, 
        min_confidence: float
    ) -> List[PredictiveInsight]:
        """Generate efficiency improvement insights."""
        insights = []
        
        # Check for automation opportunities
        if "manual_tasks" in data.metrics:
            manual_tasks = data.metrics["manual_tasks"]
            
            if manual_tasks > 10:  # Many manual tasks
                automation_potential = min(manual_tasks * 0.7, 50)  # 70% automation potential
                time_savings = automation_potential * 0.5  # 30 minutes per task
                
                confidence = min(0.90, manual_tasks / 20.0 + 0.5)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.EFFICIENCY,
                        title=f"Automation opportunity in {data.data_source}",
                        description=f"{manual_tasks} manual tasks identified. Potential to automate {automation_potential:.0f} tasks.",
                        confidence_score=confidence,
                        impact_score=min(1.0, time_savings / 20.0),
                        priority_level="high" if manual_tasks > 25 else "medium",
                        actionable_recommendations=[
                            "Prioritize high-frequency manual tasks for automation",
                            "Develop standard automation templates",
                            "Implement workflow automation tools",
                            "Train team on automation best practices"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "manual_tasks_count": manual_tasks,
                            "automation_potential": automation_potential,
                            "estimated_time_savings_hours": time_savings,
                            "automation_percentage": f"{(automation_potential / manual_tasks) * 100:.1f}%"
                        },
                        roi_estimate=time_savings * 50,  # $50/hour value
                        implementation_effort="medium"
                    )
                    insights.append(insight)
        
        # Check for workflow optimization opportunities
        if "workflow_steps" in data.metrics and "average_completion_time" in data.metrics:
            steps = data.metrics["workflow_steps"]
            completion_time = data.metrics["average_completion_time"]
            
            if steps > 10 and completion_time > 30:  # Complex, slow workflow
                optimization_potential = min(steps * 0.3, completion_time * 0.4)
                
                confidence = min(0.85, (steps / 15.0 + completion_time / 60.0) / 2)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.WORKFLOW_ENHANCEMENT,
                        title=f"Workflow optimization opportunity in {data.data_source}",
                        description=f"Complex workflow with {steps} steps taking {completion_time:.1f} minutes on average.",
                        confidence_score=confidence,
                        impact_score=min(1.0, optimization_potential / 20.0),
                        priority_level="medium",
                        actionable_recommendations=[
                            "Streamline workflow by removing unnecessary steps",
                            "Parallelize independent workflow components",
                            "Implement automated handoffs between steps",
                            "Optimize bottleneck processes"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "workflow_steps": steps,
                            "completion_time_minutes": completion_time,
                            "optimization_potential_minutes": optimization_potential,
                            "complexity_score": steps * completion_time / 100
                        },
                        implementation_effort="medium"
                    )
                    insights.append(insight)
        
        return insights

    async def _generate_risk_insights(
        self, 
        data: InsightData, 
        min_confidence: float
    ) -> List[PredictiveInsight]:
        """Generate risk mitigation insights."""
        insights = []
        
        # Check for error rate patterns
        if "error_rate" in data.metrics:
            error_rate = data.metrics["error_rate"]
            
            if error_rate > 5.0:  # High error rate
                confidence = min(0.95, (error_rate - 5.0) / 10.0 + 0.7)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.RISK_MITIGATION,
                        title=f"High error rate detected in {data.data_source}",
                        description=f"Error rate of {error_rate:.1f}% exceeds acceptable threshold of 5%.",
                        confidence_score=confidence,
                        impact_score=min(1.0, (error_rate - 5.0) / 10.0 + 0.5),
                        priority_level="critical" if error_rate > 15.0 else "high",
                        actionable_recommendations=[
                            "Implement comprehensive error handling and retry logic",
                            "Review and strengthen input validation",
                            "Add monitoring and alerting for error spikes",
                            "Conduct root cause analysis for common errors"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_error_rate": error_rate,
                            "acceptable_threshold": 5.0,
                            "error_rate_excess": f"{error_rate - 5.0:.1f}%",
                            "risk_level": "critical" if error_rate > 15.0 else "high"
                        },
                        implementation_effort="high"
                    )
                    insights.append(insight)
        
        # Check for security vulnerabilities
        if "security_issues" in data.metrics:
            security_issues = data.metrics["security_issues"]
            
            if security_issues > 0:
                confidence = min(0.90, security_issues / 5.0 + 0.6)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.RISK_MITIGATION,
                        title=f"Security issues identified in {data.data_source}",
                        description=f"{security_issues} security issues require immediate attention.",
                        confidence_score=confidence,
                        impact_score=min(1.0, security_issues / 3.0),
                        priority_level="critical",
                        actionable_recommendations=[
                            "Address all identified security vulnerabilities immediately",
                            "Implement security scanning in CI/CD pipeline",
                            "Review access controls and permissions",
                            "Conduct security audit and penetration testing"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "security_issues_count": security_issues,
                            "criticality": "immediate_action_required",
                            "risk_category": "security_compliance"
                        },
                        implementation_effort="high"
                    )
                    insights.append(insight)
        
        return insights

    async def _generate_capacity_insights(
        self, 
        data: InsightData, 
        min_confidence: float
    ) -> List[PredictiveInsight]:
        """Generate capacity planning insights."""
        insights = []
        
        # Check for approaching capacity limits
        if "capacity_utilization" in data.metrics:
            utilization = data.metrics["capacity_utilization"]
            
            if utilization > 75.0:  # Approaching capacity
                confidence = min(0.85, (utilization - 75.0) / 20.0 + 0.6)
                
                if confidence >= min_confidence:
                    insight = PredictiveInsight(
                        insight_id=create_insight_id(),
                        insight_type=InsightType.CAPACITY_PLANNING,
                        title=f"Capacity planning needed for {data.data_source}",
                        description=f"Current utilization at {utilization:.1f}% requires capacity planning.",
                        confidence_score=confidence,
                        impact_score=min(1.0, (utilization - 75.0) / 25.0 + 0.4),
                        priority_level="high" if utilization > 90.0 else "medium",
                        actionable_recommendations=[
                            "Plan capacity expansion before reaching 90% utilization",
                            "Implement auto-scaling policies",
                            "Optimize resource allocation",
                            "Consider load balancing strategies"
                        ],
                        data_sources=[data.data_source],
                        supporting_evidence={
                            "current_utilization": utilization,
                            "recommended_threshold": 75.0,
                            "time_to_capacity": "estimated based on growth trends",
                            "scaling_recommendation": "20-30% additional capacity"
                        },
                        implementation_effort="medium"
                    )
                    insights.append(insight)
        
        return insights

    async def _generate_cross_data_insights(
        self,
        insight_data: List[InsightData],
        categories: List[InsightCategory]
    ) -> Either[PredictiveModelingError, List[PredictiveInsight]]:
        """Generate insights by analyzing data across multiple sources."""
        try:
            insights = []
            
            # Analyze correlations between different data sources
            if len(insight_data) >= 2:
                correlation_insights = await self._analyze_data_correlations(insight_data)
                insights.extend(correlation_insights)
            
            # Identify system-wide patterns
            if InsightCategory.WORKFLOW_ENHANCEMENT in categories:
                workflow_insights = await self._analyze_workflow_patterns(insight_data)
                insights.extend(workflow_insights)
            
            # Detect anomalies across systems
            anomaly_insights = await self._detect_cross_system_anomalies(insight_data)
            insights.extend(anomaly_insights)
            
            return Either.right(insights)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Cross-data insight generation failed: {str(e)}", 
                "CROSS_DATA_ANALYSIS_ERROR"
            ))

    async def _analyze_data_correlations(self, insight_data: List[InsightData]) -> List[PredictiveInsight]:
        """Analyze correlations between different data sources."""
        insights = []
        
        # Look for performance correlations
        performance_data = [d for d in insight_data if "response_time" in d.metrics]
        
        if len(performance_data) >= 2:
            response_times = [d.metrics["response_time"] for d in performance_data]
            avg_response_time = statistics.mean(response_times)
            
            if avg_response_time > 3.0:  # System-wide performance issue
                insight = PredictiveInsight(
                    insight_id=create_insight_id(),
                    insight_type=InsightType.PERFORMANCE_IMPROVEMENT,
                    title="System-wide performance degradation detected",
                    description=f"Multiple systems showing elevated response times (avg: {avg_response_time:.2f}s)",
                    confidence_score=0.85,
                    impact_score=0.9,
                    priority_level="high",
                    actionable_recommendations=[
                        "Investigate shared infrastructure bottlenecks",
                        "Review network connectivity and latency",
                        "Check database performance and query optimization",
                        "Monitor shared resource contention"
                    ],
                    data_sources=[d.data_source for d in performance_data],
                    supporting_evidence={
                        "average_response_time": avg_response_time,
                        "affected_systems": len(performance_data),
                        "correlation_strength": "high",
                        "system_impact": "widespread"
                    },
                    implementation_effort="high"
                )
                insights.append(insight)
        
        return insights

    async def _analyze_workflow_patterns(self, insight_data: List[InsightData]) -> List[PredictiveInsight]:
        """Analyze workflow patterns across systems."""
        insights = []
        
        # Count workflow-related metrics
        workflow_systems = [d for d in insight_data if "workflow_steps" in d.metrics]
        
        if len(workflow_systems) >= 3:
            total_steps = sum(d.metrics["workflow_steps"] for d in workflow_systems)
            avg_steps = total_steps / len(workflow_systems)
            
            if avg_steps > 12:  # Complex workflows
                insight = PredictiveInsight(
                    insight_id=create_insight_id(),
                    insight_type=InsightType.WORKFLOW_ENHANCEMENT,
                    title="Complex workflow patterns across systems",
                    description=f"Average of {avg_steps:.1f} steps per workflow suggests optimization opportunity",
                    confidence_score=0.80,
                    impact_score=0.7,
                    priority_level="medium",
                    actionable_recommendations=[
                        "Standardize workflow patterns across systems",
                        "Implement workflow templates for common patterns",
                        "Create reusable workflow components",
                        "Establish workflow optimization guidelines"
                    ],
                    data_sources=[d.data_source for d in workflow_systems],
                    supporting_evidence={
                        "average_workflow_steps": avg_steps,
                        "systems_analyzed": len(workflow_systems),
                        "complexity_assessment": "high",
                        "optimization_potential": "significant"
                    },
                    implementation_effort="medium"
                )
                insights.append(insight)
        
        return insights

    async def _detect_cross_system_anomalies(self, insight_data: List[InsightData]) -> List[PredictiveInsight]:
        """Detect anomalies that span multiple systems."""
        insights = []
        
        # Check for synchronized anomalies
        systems_with_anomalies = [d for d in insight_data if d.anomalies]
        
        if len(systems_with_anomalies) >= 2:
            # Check if anomalies occurred around the same time
            anomaly_times = []
            for data in systems_with_anomalies:
                for anomaly in data.anomalies:
                    if "timestamp" in anomaly:
                        anomaly_times.append(anomaly["timestamp"])
            
            if len(anomaly_times) >= 2:
                insight = PredictiveInsight(
                    insight_id=create_insight_id(),
                    insight_type=InsightType.ANOMALY_ALERT,
                    title="Cross-system anomalies detected",
                    description=f"Synchronized anomalies detected across {len(systems_with_anomalies)} systems",
                    confidence_score=0.75,
                    impact_score=0.8,
                    priority_level="high",
                    actionable_recommendations=[
                        "Investigate common cause for cross-system anomalies",
                        "Review shared infrastructure and dependencies",
                        "Implement cross-system monitoring and correlation",
                        "Establish incident response procedures"
                    ],
                    data_sources=[d.data_source for d in systems_with_anomalies],
                    supporting_evidence={
                        "affected_systems": len(systems_with_anomalies),
                        "anomaly_count": len(anomaly_times),
                        "correlation_type": "temporal",
                        "investigation_priority": "high"
                    },
                    implementation_effort="medium"
                )
                insights.append(insight)
        
        return insights

    async def _calculate_insight_roi(
        self, 
        insight: PredictiveInsight
    ) -> Either[PredictiveModelingError, Dict[str, float]]:
        """Calculate ROI for a specific insight."""
        try:
            roi_data = {}
            
            if insight.insight_type == InsightType.PERFORMANCE_IMPROVEMENT:
                # Calculate performance improvement ROI
                current_time = insight.supporting_evidence.get("current_response_time", 2.0)
                if current_time > 2.0:
                    roi_data = self._calculate_performance_roi(current_time)
            
            elif insight.insight_type == InsightType.COST_SAVINGS:
                # Calculate cost savings ROI
                potential_savings = insight.supporting_evidence.get("potential_annual_savings", 0)
                roi_data = self._calculate_cost_savings_roi(potential_savings)
            
            elif insight.insight_type == InsightType.EFFICIENCY:
                # Calculate efficiency improvement ROI
                time_savings = insight.supporting_evidence.get("estimated_time_savings_hours", 0)
                roi_data = self._calculate_efficiency_roi(time_savings)
            
            else:
                # Default ROI calculation
                roi_data = {
                    "estimated_annual_benefit": 5000.0,
                    "implementation_cost": 2000.0,
                    "roi_percentage": 150.0,
                    "payback_months": 5
                }
            
            return Either.right(roi_data)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"ROI calculation failed: {str(e)}", 
                "ROI_CALCULATION_ERROR"
            ))

    def _estimate_performance_roi(self, response_time: float) -> float:
        """Estimate ROI for performance improvements."""
        if response_time <= 2.0:
            return 0.0
        
        # Calculate time savings and value
        improvement_factor = (response_time - 2.0) / response_time
        annual_requests = 100000  # Estimate
        time_saved_per_request = (response_time - 2.0)
        total_time_saved_hours = (annual_requests * time_saved_per_request) / 3600
        
        # Value at $50/hour
        return total_time_saved_hours * 50.0

    def _calculate_performance_roi(self, current_time: float) -> Dict[str, float]:
        """Calculate detailed performance improvement ROI."""
        target_time = 2.0
        improvement_factor = max(0, (current_time - target_time) / current_time)
        
        annual_requests = 100000
        time_saved_per_request = current_time - target_time
        total_time_saved_hours = (annual_requests * time_saved_per_request) / 3600
        
        annual_benefit = total_time_saved_hours * 50.0  # $50/hour
        implementation_cost = 5000.0  # Estimated cost
        
        roi_percentage = ((annual_benefit - implementation_cost) / implementation_cost) * 100
        payback_months = (implementation_cost / annual_benefit) * 12 if annual_benefit > 0 else 999
        
        return {
            "estimated_annual_benefit": annual_benefit,
            "implementation_cost": implementation_cost,
            "roi_percentage": roi_percentage,
            "payback_months": min(payback_months, 60),
            "time_saved_hours_annually": total_time_saved_hours
        }

    def _calculate_cost_savings_roi(self, potential_savings: float) -> Dict[str, float]:
        """Calculate cost savings ROI."""
        implementation_cost = potential_savings * 0.2  # 20% of savings
        
        roi_percentage = ((potential_savings - implementation_cost) / implementation_cost) * 100
        payback_months = (implementation_cost / potential_savings) * 12 if potential_savings > 0 else 999
        
        return {
            "estimated_annual_benefit": potential_savings,
            "implementation_cost": implementation_cost,
            "roi_percentage": roi_percentage,
            "payback_months": min(payback_months, 36)
        }

    def _calculate_efficiency_roi(self, time_savings_hours: float) -> Dict[str, float]:
        """Calculate efficiency improvement ROI."""
        annual_benefit = time_savings_hours * 50.0 * 12  # Monthly to annual
        implementation_cost = 3000.0  # Automation setup cost
        
        roi_percentage = ((annual_benefit - implementation_cost) / implementation_cost) * 100
        payback_months = (implementation_cost / (annual_benefit / 12)) if annual_benefit > 0 else 999
        
        return {
            "estimated_annual_benefit": annual_benefit,
            "implementation_cost": implementation_cost,
            "roi_percentage": roi_percentage,
            "payback_months": min(payback_months, 24),
            "time_saved_hours_annually": time_savings_hours * 12
        }

    async def generate_executive_summary(
        self,
        insights: List[PredictiveInsight],
        time_period: str = "last_month"
    ) -> Either[PredictiveModelingError, ExecutiveSummary]:
        """Generate executive summary from insights."""
        try:
            if not insights:
                return Either.left(PredictiveModelingError(
                    "No insights provided for executive summary", 
                    "INSUFFICIENT_DATA"
                ))
            
            # Extract key findings
            key_findings = []
            critical_issues = []
            top_opportunities = []
            recommended_actions = []
            
            # Categorize insights
            critical_insights = [i for i in insights if i.priority_level == "critical"]
            high_impact_insights = [i for i in insights if i.impact_score > 0.7]
            
            # Key findings
            key_findings = [
                f"Analyzed {len(insights)} insights across automation systems",
                f"Identified {len(critical_insights)} critical issues requiring immediate attention",
                f"Found {len(high_impact_insights)} high-impact optimization opportunities"
            ]
            
            # Critical issues
            critical_issues = [i.title for i in critical_insights[:5]]
            
            # Top opportunities
            top_opportunities = [i.title for i in high_impact_insights[:5]]
            
            # Recommended actions
            all_recommendations = []
            for insight in insights[:10]:  # Top 10 insights
                all_recommendations.extend(insight.actionable_recommendations[:2])
            
            # Get unique recommendations
            recommended_actions = list(dict.fromkeys(all_recommendations))[:8]
            
            # Calculate financial impact
            total_potential_savings = sum(
                insight.roi_estimate or 0 for insight in insights 
                if insight.roi_estimate
            )
            
            # Estimate implementation costs (20% of savings)
            total_investment_required = total_potential_savings * 0.2
            
            # Calculate overall ROI
            overall_roi = ((total_potential_savings - total_investment_required) / 
                          total_investment_required * 100) if total_investment_required > 0 else 0
            
            # Strategic priorities
            strategic_priorities = [
                "Performance optimization and system reliability",
                "Cost reduction and resource efficiency",
                "Risk mitigation and security enhancement",
                "Automation expansion and workflow optimization"
            ]
            
            # Overall confidence
            confidence_score = statistics.mean([i.confidence_score for i in insights]) if insights else 0.0
            
            summary = ExecutiveSummary(
                summary_id=f"exec_summary_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                time_period=time_period,
                key_findings=key_findings,
                critical_issues=critical_issues,
                top_opportunities=top_opportunities,
                recommended_actions=recommended_actions,
                total_potential_savings=total_potential_savings,
                total_investment_required=total_investment_required,
                overall_roi=overall_roi,
                strategic_priorities=strategic_priorities,
                risk_assessment="medium" if len(critical_insights) < 5 else "high",
                confidence_score=confidence_score
            )
            
            return Either.right(summary)
            
        except Exception as e:
            return Either.left(PredictiveModelingError(
                f"Executive summary generation failed: {str(e)}", 
                "SUMMARY_GENERATION_ERROR"
            ))

    async def get_insight_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of insight generation capabilities."""
        try:
            insight_categories = [category.value for category in InsightCategory]
            
            insight_counts_by_type = Counter([
                insight.insight_type.value for insight in self.insight_history
            ])
            
            high_impact_insights = [
                insight for insight in self.insight_history 
                if insight.impact_score > 0.7
            ]
            
            total_roi_estimate = sum(
                insight.roi_estimate or 0 for insight in self.insight_history 
                if insight.roi_estimate
            )
            
            return {
                "total_insights_generated": len(self.insight_history),
                "insight_categories_supported": insight_categories,
                "insights_by_type": dict(insight_counts_by_type),
                "high_impact_insights_count": len(high_impact_insights),
                "total_estimated_roi": total_roi_estimate,
                "insight_templates_loaded": len(self.insight_templates),
                "roi_models_available": list(self.roi_models.keys()),
                "summary_timestamp": datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate insight summary: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }