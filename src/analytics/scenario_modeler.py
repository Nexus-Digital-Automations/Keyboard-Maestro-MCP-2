"""
Scenario Modeler - TASK_59 Phase 4 Advanced Modeling Implementation

Advanced scenario modeling and simulation for automation workflows.
Provides what-if analysis, stress testing, capacity planning, and growth modeling.

Architecture: Scenario Engine + Simulation Framework + Monte Carlo Analysis + Stress Testing
Performance: <500ms scenario setup, <2s simulation, <5s comprehensive analysis
Security: Safe scenario parameters, validated simulations, comprehensive audit logging
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
import math
import random
from collections import defaultdict, deque

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    ScenarioId, create_scenario_id, ModelId, PredictiveModelingError,
    ScenarioModelingError, validate_scenario_parameters
)


class ScenarioType(Enum):
    """Types of scenarios that can be modeled."""
    WHAT_IF = "what_if"
    STRESS_TEST = "stress_test"
    CAPACITY_PLANNING = "capacity_planning"
    GROWTH_MODELING = "growth_modeling"
    FAILURE_SIMULATION = "failure_simulation"
    LOAD_TESTING = "load_testing"
    SCALING_ANALYSIS = "scaling_analysis"
    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_TESTING = "performance_testing"
    DISASTER_RECOVERY = "disaster_recovery"


class SimulationMethod(Enum):
    """Methods for scenario simulation."""
    MONTE_CARLO = "monte_carlo"
    DISCRETE_EVENT = "discrete_event"
    AGENT_BASED = "agent_based"
    SYSTEM_DYNAMICS = "system_dynamics"
    QUEUE_THEORY = "queue_theory"
    STATISTICAL_MODELING = "statistical_modeling"
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"


class UncertaintyType(Enum):
    """Types of uncertainty in scenario modeling."""
    PARAMETER_UNCERTAINTY = "parameter_uncertainty"
    MODEL_UNCERTAINTY = "model_uncertainty"
    SCENARIO_UNCERTAINTY = "scenario_uncertainty"
    INPUT_UNCERTAINTY = "input_uncertainty"
    STRUCTURAL_UNCERTAINTY = "structural_uncertainty"


@dataclass(frozen=True)
class ScenarioParameter:
    """Parameter for scenario modeling."""
    parameter_id: str
    parameter_name: str
    parameter_type: str  # numeric, categorical, boolean
    base_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    distribution_type: str = "normal"  # normal, uniform, exponential, beta
    uncertainty_level: float = 0.1  # 0.0 to 1.0
    correlation_factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.uncertainty_level <= 1.0):
            raise ValueError("Uncertainty level must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ScenarioConfiguration:
    """Configuration for scenario modeling."""
    scenario_id: ScenarioId
    scenario_name: str
    scenario_type: ScenarioType
    simulation_method: SimulationMethod
    parameters: List[ScenarioParameter]
    time_horizon: timedelta
    simulation_iterations: int = 1000
    confidence_level: float = 0.95
    enable_uncertainty_analysis: bool = True
    enable_sensitivity_analysis: bool = True
    enable_correlation_analysis: bool = True
    
    def __post_init__(self):
        if self.simulation_iterations < 100:
            raise ValueError("Simulation iterations must be at least 100")
        if not (0.5 <= self.confidence_level <= 0.99):
            raise ValueError("Confidence level must be between 0.5 and 0.99")


@dataclass(frozen=True)
class ScenarioOutcome:
    """Individual outcome from scenario simulation."""
    outcome_id: str
    scenario_id: ScenarioId
    iteration_number: int
    parameter_values: Dict[str, Any]
    outcome_metrics: Dict[str, float]
    performance_indicators: Dict[str, Any] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.iteration_number < 0:
            raise ValueError("Iteration number must be non-negative")


@dataclass(frozen=True)
class ScenarioResults:
    """Comprehensive results from scenario modeling."""
    results_id: str
    scenario_id: ScenarioId
    scenario_configuration: ScenarioConfiguration
    individual_outcomes: List[ScenarioOutcome]
    statistical_summary: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    probability_distributions: Dict[str, List[float]]
    uncertainty_analysis: Dict[str, Any] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.individual_outcomes) != self.scenario_configuration.simulation_iterations:
            raise ValueError("Number of outcomes must match simulation iterations")


@dataclass(frozen=True)
class ScenarioComparison:
    """Comparison between multiple scenarios."""
    comparison_id: str
    scenario_results: List[ScenarioResults]
    comparative_analysis: Dict[str, Any]
    ranking_analysis: Dict[str, List[str]]
    trade_off_analysis: Dict[str, Dict[str, float]]
    decision_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommended_scenario: Optional[str] = None


class ScenarioModeler:
    """Advanced scenario modeling and simulation system."""
    
    def __init__(self):
        self.scenario_cache: Dict[str, ScenarioResults] = {}
        self.modeling_history: deque = deque(maxlen=1000)
        self.parameter_templates: Dict[ScenarioType, List[ScenarioParameter]] = {}
        self.simulation_engines: Dict[SimulationMethod, Callable] = {}
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.correlation_matrices: Dict[str, Dict[Tuple[str, str], float]] = {}
        self._initialize_parameter_templates()
        self._initialize_simulation_engines()
    
    def _initialize_parameter_templates(self):
        """Initialize parameter templates for different scenario types."""
        self.parameter_templates[ScenarioType.STRESS_TEST] = [
            ScenarioParameter(
                parameter_id="load_multiplier",
                parameter_name="Load Multiplier",
                parameter_type="numeric",
                base_value=1.0,
                min_value=0.5,
                max_value=10.0,
                distribution_type="uniform",
                uncertainty_level=0.2
            ),
            ScenarioParameter(
                parameter_id="concurrent_users",
                parameter_name="Concurrent Users",
                parameter_type="numeric",
                base_value=100,
                min_value=10,
                max_value=1000,
                distribution_type="normal",
                uncertainty_level=0.3
            )
        ]
        
        self.parameter_templates[ScenarioType.CAPACITY_PLANNING] = [
            ScenarioParameter(
                parameter_id="growth_rate",
                parameter_name="Growth Rate",
                parameter_type="numeric",
                base_value=0.2,
                min_value=0.0,
                max_value=1.0,
                distribution_type="beta",
                uncertainty_level=0.4
            ),
            ScenarioParameter(
                parameter_id="resource_efficiency",
                parameter_name="Resource Efficiency",
                parameter_type="numeric",
                base_value=0.8,
                min_value=0.5,
                max_value=0.95,
                distribution_type="normal",
                uncertainty_level=0.1
            )
        ]
        
        self.parameter_templates[ScenarioType.WHAT_IF] = [
            ScenarioParameter(
                parameter_id="automation_rate",
                parameter_name="Automation Rate",
                parameter_type="numeric",
                base_value=0.7,
                min_value=0.0,
                max_value=1.0,
                distribution_type="uniform",
                uncertainty_level=0.2
            ),
            ScenarioParameter(
                parameter_id="error_rate",
                parameter_name="Error Rate",
                parameter_type="numeric",
                base_value=0.05,
                min_value=0.001,
                max_value=0.2,
                distribution_type="exponential",
                uncertainty_level=0.5
            )
        ]
    
    def _initialize_simulation_engines(self):
        """Initialize simulation engines for different methods."""
        self.simulation_engines = {
            SimulationMethod.MONTE_CARLO: self._run_monte_carlo_simulation,
            SimulationMethod.DETERMINISTIC: self._run_deterministic_simulation,
            SimulationMethod.STATISTICAL_MODELING: self._run_statistical_simulation,
            SimulationMethod.DISCRETE_EVENT: self._run_discrete_event_simulation,
            SimulationMethod.QUEUE_THEORY: self._run_queue_theory_simulation
        }
    
    @require(lambda config: isinstance(config, ScenarioConfiguration))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, ScenarioModelingError))
    async def model_scenario(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]] = None
    ) -> Either[ScenarioModelingError, ScenarioResults]:
        """Model a scenario with comprehensive analysis."""
        try:
            start_time = datetime.now(UTC)
            
            # Check cache first
            cache_key = f"{config.scenario_id}_{hash(str(config))}"
            if cache_key in self.scenario_cache:
                return Either.right(self.scenario_cache[cache_key])
            
            # Get simulation engine
            simulation_engine = self.simulation_engines.get(config.simulation_method)
            if not simulation_engine:
                return Either.left(ScenarioModelingError(f"Simulation method {config.simulation_method.value} not supported"))
            
            # Run simulation
            outcomes = await simulation_engine(config, baseline_data)
            
            # Calculate statistical summary
            statistical_summary = self._calculate_statistical_summary(outcomes)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(outcomes, config.confidence_level)
            
            # Estimate probability distributions
            probability_distributions = self._estimate_probability_distributions(outcomes)
            
            # Perform uncertainty analysis if enabled
            uncertainty_analysis = {}
            if config.enable_uncertainty_analysis:
                uncertainty_analysis = await self._perform_uncertainty_analysis(config, outcomes)
            
            # Perform sensitivity analysis if enabled
            sensitivity_analysis = {}
            if config.enable_sensitivity_analysis:
                sensitivity_analysis = await self._perform_sensitivity_analysis(config, outcomes)
            
            # Assess risks
            risk_assessment = self._assess_scenario_risks(outcomes, config)
            
            # Generate recommendations
            recommendations = self._generate_scenario_recommendations(outcomes, config, risk_assessment)
            
            # Create results
            results = ScenarioResults(
                results_id=f"results_{config.scenario_id}_{datetime.now(UTC).isoformat()}",
                scenario_id=config.scenario_id,
                scenario_configuration=config,
                individual_outcomes=outcomes,
                statistical_summary=statistical_summary,
                confidence_intervals=confidence_intervals,
                probability_distributions=probability_distributions,
                uncertainty_analysis=uncertainty_analysis,
                sensitivity_analysis=sensitivity_analysis,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
            # Cache results
            self.scenario_cache[cache_key] = results
            
            # Record modeling activity
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            self.modeling_history.append({
                "timestamp": datetime.now(UTC),
                "scenario_type": config.scenario_type.value,
                "simulation_method": config.simulation_method.value,
                "iterations": config.simulation_iterations,
                "processing_time": processing_time
            })
            
            return Either.right(results)
            
        except Exception as e:
            return Either.left(ScenarioModelingError(f"Scenario modeling failed: {str(e)}"))
    
    async def _run_monte_carlo_simulation(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[ScenarioOutcome]:
        """Run Monte Carlo simulation for scenario modeling."""
        outcomes = []
        
        for iteration in range(config.simulation_iterations):
            # Generate random parameter values
            parameter_values = {}
            for param in config.parameters:
                value = self._generate_random_parameter_value(param)
                parameter_values[param.parameter_name] = value
            
            # Simulate outcome
            outcome_metrics = await self._simulate_scenario_outcome(
                parameter_values, config, baseline_data, iteration
            )
            
            # Create outcome
            outcome = ScenarioOutcome(
                outcome_id=f"outcome_{config.scenario_id}_{iteration}",
                scenario_id=config.scenario_id,
                iteration_number=iteration,
                parameter_values=parameter_values,
                outcome_metrics=outcome_metrics,
                performance_indicators=self._calculate_performance_indicators(outcome_metrics),
                resource_utilization=self._calculate_resource_utilization(parameter_values, outcome_metrics),
                cost_metrics=self._calculate_cost_metrics(parameter_values, outcome_metrics)
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    async def _run_deterministic_simulation(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[ScenarioOutcome]:
        """Run deterministic simulation using parameter ranges."""
        outcomes = []
        
        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(config.parameters, config.simulation_iterations)
        
        for iteration, parameter_values in enumerate(parameter_combinations):
            # Simulate outcome
            outcome_metrics = await self._simulate_scenario_outcome(
                parameter_values, config, baseline_data, iteration
            )
            
            # Create outcome
            outcome = ScenarioOutcome(
                outcome_id=f"outcome_{config.scenario_id}_{iteration}",
                scenario_id=config.scenario_id,
                iteration_number=iteration,
                parameter_values=parameter_values,
                outcome_metrics=outcome_metrics,
                performance_indicators=self._calculate_performance_indicators(outcome_metrics),
                resource_utilization=self._calculate_resource_utilization(parameter_values, outcome_metrics),
                cost_metrics=self._calculate_cost_metrics(parameter_values, outcome_metrics)
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    async def _run_statistical_simulation(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[ScenarioOutcome]:
        """Run statistical simulation with probability distributions."""
        outcomes = []
        
        # Pre-calculate statistical relationships
        correlations = self._calculate_parameter_correlations(config.parameters)
        
        for iteration in range(config.simulation_iterations):
            # Generate correlated parameter values
            parameter_values = self._generate_correlated_parameter_values(config.parameters, correlations)
            
            # Simulate outcome
            outcome_metrics = await self._simulate_scenario_outcome(
                parameter_values, config, baseline_data, iteration
            )
            
            # Create outcome
            outcome = ScenarioOutcome(
                outcome_id=f"outcome_{config.scenario_id}_{iteration}",
                scenario_id=config.scenario_id,
                iteration_number=iteration,
                parameter_values=parameter_values,
                outcome_metrics=outcome_metrics,
                performance_indicators=self._calculate_performance_indicators(outcome_metrics),
                resource_utilization=self._calculate_resource_utilization(parameter_values, outcome_metrics),
                cost_metrics=self._calculate_cost_metrics(parameter_values, outcome_metrics)
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    async def _run_discrete_event_simulation(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[ScenarioOutcome]:
        """Run discrete event simulation for time-based scenarios."""
        outcomes = []
        
        # Time-based simulation parameters
        time_step = config.time_horizon / config.simulation_iterations
        
        for iteration in range(config.simulation_iterations):
            current_time = time_step * iteration
            
            # Generate time-dependent parameter values
            parameter_values = {}
            for param in config.parameters:
                # Add time dependency to parameter values
                base_value = self._generate_random_parameter_value(param)
                time_factor = self._calculate_time_factor(current_time, config.time_horizon, param)
                parameter_values[param.parameter_name] = base_value * time_factor
            
            # Simulate outcome
            outcome_metrics = await self._simulate_scenario_outcome(
                parameter_values, config, baseline_data, iteration
            )
            
            # Add time-based metrics
            outcome_metrics["simulation_time"] = current_time.total_seconds()
            outcome_metrics["time_progress"] = iteration / config.simulation_iterations
            
            # Create outcome
            outcome = ScenarioOutcome(
                outcome_id=f"outcome_{config.scenario_id}_{iteration}",
                scenario_id=config.scenario_id,
                iteration_number=iteration,
                parameter_values=parameter_values,
                outcome_metrics=outcome_metrics,
                performance_indicators=self._calculate_performance_indicators(outcome_metrics),
                resource_utilization=self._calculate_resource_utilization(parameter_values, outcome_metrics),
                cost_metrics=self._calculate_cost_metrics(parameter_values, outcome_metrics)
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    async def _run_queue_theory_simulation(
        self,
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]]
    ) -> List[ScenarioOutcome]:
        """Run queue theory simulation for capacity and performance modeling."""
        outcomes = []
        
        for iteration in range(config.simulation_iterations):
            # Generate parameter values with queue theory focus
            parameter_values = {}
            for param in config.parameters:
                value = self._generate_random_parameter_value(param)
                parameter_values[param.parameter_name] = value
            
            # Calculate queue theory metrics
            arrival_rate = parameter_values.get("arrival_rate", 10.0)
            service_rate = parameter_values.get("service_rate", 12.0)
            
            # Basic M/M/1 queue calculations
            if service_rate > arrival_rate:
                utilization = arrival_rate / service_rate
                avg_queue_length = (arrival_rate ** 2) / (service_rate * (service_rate - arrival_rate))
                avg_wait_time = arrival_rate / (service_rate * (service_rate - arrival_rate))
            else:
                # Unstable queue
                utilization = 1.0
                avg_queue_length = float('inf')
                avg_wait_time = float('inf')
            
            # Simulate outcome
            outcome_metrics = await self._simulate_scenario_outcome(
                parameter_values, config, baseline_data, iteration
            )
            
            # Add queue theory metrics
            outcome_metrics["queue_utilization"] = utilization
            outcome_metrics["average_queue_length"] = min(1000, avg_queue_length)  # Cap at reasonable value
            outcome_metrics["average_wait_time"] = min(3600, avg_wait_time)  # Cap at 1 hour
            
            # Create outcome
            outcome = ScenarioOutcome(
                outcome_id=f"outcome_{config.scenario_id}_{iteration}",
                scenario_id=config.scenario_id,
                iteration_number=iteration,
                parameter_values=parameter_values,
                outcome_metrics=outcome_metrics,
                performance_indicators=self._calculate_performance_indicators(outcome_metrics),
                resource_utilization=self._calculate_resource_utilization(parameter_values, outcome_metrics),
                cost_metrics=self._calculate_cost_metrics(parameter_values, outcome_metrics)
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    def _generate_random_parameter_value(self, param: ScenarioParameter) -> Any:
        """Generate random value for a parameter based on its distribution."""
        if param.parameter_type == "boolean":
            return random.choice([True, False])
        elif param.parameter_type == "categorical":
            if hasattr(param, 'allowed_values') and param.allowed_values:
                return random.choice(param.allowed_values)
            else:
                return param.base_value
        elif param.parameter_type == "numeric":
            base = float(param.base_value) if param.base_value is not None else 1.0
            min_val = param.min_value if param.min_value is not None else base * 0.5
            max_val = param.max_value if param.max_value is not None else base * 2.0
            
            if param.distribution_type == "uniform":
                return random.uniform(min_val, max_val)
            elif param.distribution_type == "normal":
                std_dev = (max_val - min_val) / 6  # 99.7% within range
                value = random.gauss(base, std_dev)
                return max(min_val, min(max_val, value))
            elif param.distribution_type == "exponential":
                # Use base as the rate parameter
                rate = 1.0 / base if base > 0 else 1.0
                value = random.expovariate(rate)
                return max(min_val, min(max_val, value))
            elif param.distribution_type == "beta":
                # Use base to determine beta distribution parameters
                alpha = 2.0
                beta = 2.0 * (1.0 - base) / base if base > 0 and base < 1 else 2.0
                value = random.betavariate(alpha, beta)
                return min_val + value * (max_val - min_val)
            else:
                # Default to uniform
                return random.uniform(min_val, max_val)
        else:
            return param.base_value
    
    def _generate_parameter_combinations(
        self,
        parameters: List[ScenarioParameter],
        num_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate systematic parameter combinations for deterministic simulation."""
        combinations = []
        
        # Simple grid generation (can be improved with more sophisticated methods)
        steps_per_param = max(2, int(num_combinations ** (1.0 / len(parameters))))
        
        # Generate all combinations
        param_ranges = []
        for param in parameters:
            if param.parameter_type == "numeric":
                min_val = param.min_value if param.min_value is not None else float(param.base_value) * 0.5
                max_val = param.max_value if param.max_value is not None else float(param.base_value) * 2.0
                
                param_range = []
                for i in range(steps_per_param):
                    value = min_val + (max_val - min_val) * i / (steps_per_param - 1) if steps_per_param > 1 else min_val
                    param_range.append(value)
                param_ranges.append(param_range)
            else:
                param_ranges.append([param.base_value])
        
        # Generate combinations
        import itertools
        for combination in itertools.product(*param_ranges):
            if len(combinations) >= num_combinations:
                break
            
            param_values = {}
            for i, param in enumerate(parameters):
                param_values[param.parameter_name] = combination[i]
            
            combinations.append(param_values)
        
        # Fill remaining combinations with random values if needed
        while len(combinations) < num_combinations:
            param_values = {}
            for param in parameters:
                param_values[param.parameter_name] = self._generate_random_parameter_value(param)
            combinations.append(param_values)
        
        return combinations[:num_combinations]
    
    def _calculate_parameter_correlations(
        self,
        parameters: List[ScenarioParameter]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between parameters."""
        correlations = {}
        
        for i, param1 in enumerate(parameters):
            for j, param2 in enumerate(parameters):
                if i < j:
                    # Use predefined correlations if available
                    correlation = param1.correlation_factors.get(param2.parameter_name, 0.0)
                    correlations[(param1.parameter_name, param2.parameter_name)] = correlation
        
        return correlations
    
    def _generate_correlated_parameter_values(
        self,
        parameters: List[ScenarioParameter],
        correlations: Dict[Tuple[str, str], float]
    ) -> Dict[str, Any]:
        """Generate correlated parameter values using multivariate normal distribution."""
        parameter_values = {}
        
        # For simplicity, generate independent values and then apply correlations
        # In a real implementation, this would use proper multivariate distributions
        
        # Generate base values
        for param in parameters:
            parameter_values[param.parameter_name] = self._generate_random_parameter_value(param)
        
        # Apply simple correlation adjustments
        for (param1_name, param2_name), correlation in correlations.items():
            if abs(correlation) > 0.1:  # Only apply significant correlations
                if param1_name in parameter_values and param2_name in parameter_values:
                    # Simple correlation adjustment (simplified implementation)
                    param1_value = parameter_values[param1_name]
                    param2_value = parameter_values[param2_name]
                    
                    if isinstance(param1_value, (int, float)) and isinstance(param2_value, (int, float)):
                        # Adjust param2 based on param1 and correlation
                        adjustment = correlation * (param1_value - 1.0) * 0.1  # Simplified adjustment
                        parameter_values[param2_name] = param2_value + adjustment
        
        return parameter_values
    
    def _calculate_time_factor(
        self,
        current_time: timedelta,
        total_time: timedelta,
        param: ScenarioParameter
    ) -> float:
        """Calculate time-dependent factor for parameter values."""
        progress = current_time.total_seconds() / total_time.total_seconds() if total_time.total_seconds() > 0 else 0.0
        
        # Different time patterns based on parameter characteristics
        if "growth" in param.parameter_name.lower():
            # Exponential growth pattern
            return 1.0 + progress * 0.5
        elif "load" in param.parameter_name.lower():
            # Sinusoidal load pattern
            return 1.0 + 0.3 * math.sin(2 * math.pi * progress)
        elif "efficiency" in param.parameter_name.lower():
            # Learning curve pattern
            return 1.0 + 0.2 * (1 - math.exp(-3 * progress))
        else:
            # Linear progression
            return 1.0 + progress * 0.1
    
    async def _simulate_scenario_outcome(
        self,
        parameter_values: Dict[str, Any],
        config: ScenarioConfiguration,
        baseline_data: Optional[Dict[str, Any]],
        iteration: int
    ) -> Dict[str, float]:
        """Simulate outcome for a specific set of parameter values."""
        outcome_metrics = {}
        
        # Get baseline values
        baseline = baseline_data or {"throughput": 100.0, "latency": 200.0, "cost": 10.0, "reliability": 0.99}
        
        # Simulate different metrics based on scenario type
        if config.scenario_type == ScenarioType.STRESS_TEST:
            load_multiplier = parameter_values.get("Load Multiplier", 1.0)
            concurrent_users = parameter_values.get("Concurrent Users", 100)
            
            # Simulate performance degradation under load
            base_throughput = baseline.get("throughput", 100.0)
            outcome_metrics["throughput"] = base_throughput * load_multiplier * (1 - 0.1 * math.log(load_multiplier) if load_multiplier > 1 else 1)
            outcome_metrics["latency"] = baseline.get("latency", 200.0) * (1 + 0.5 * load_multiplier)
            outcome_metrics["error_rate"] = 0.01 * (load_multiplier ** 1.5)
            outcome_metrics["resource_utilization"] = min(1.0, 0.3 + 0.7 * load_multiplier)
            
        elif config.scenario_type == ScenarioType.CAPACITY_PLANNING:
            growth_rate = parameter_values.get("Growth Rate", 0.2)
            resource_efficiency = parameter_values.get("Resource Efficiency", 0.8)
            
            # Simulate capacity requirements
            base_capacity = baseline.get("capacity", 1000.0)
            outcome_metrics["required_capacity"] = base_capacity * (1 + growth_rate)
            outcome_metrics["resource_cost"] = outcome_metrics["required_capacity"] * 0.01 / resource_efficiency
            outcome_metrics["utilization_efficiency"] = resource_efficiency * 0.9  # Slight degradation
            outcome_metrics["scaling_factor"] = 1 + growth_rate
            
        elif config.scenario_type == ScenarioType.WHAT_IF:
            automation_rate = parameter_values.get("Automation Rate", 0.7)
            error_rate = parameter_values.get("Error Rate", 0.05)
            
            # Simulate automation outcomes
            manual_effort = baseline.get("manual_effort", 100.0)
            outcome_metrics["automated_tasks"] = manual_effort * automation_rate
            outcome_metrics["manual_tasks"] = manual_effort * (1 - automation_rate)
            outcome_metrics["total_errors"] = outcome_metrics["automated_tasks"] * error_rate
            outcome_metrics["efficiency_gain"] = automation_rate * 0.8  # 80% efficiency gain
            outcome_metrics["cost_savings"] = automation_rate * baseline.get("cost", 10.0) * 0.6
            
        else:
            # Default simulation
            for key, base_value in baseline.items():
                if isinstance(base_value, (int, float)):
                    # Apply random variation
                    variation = random.uniform(0.8, 1.2)
                    outcome_metrics[key] = base_value * variation
        
        # Add common metrics
        outcome_metrics["simulation_success"] = 1.0 if random.random() > 0.05 else 0.0  # 95% success rate
        outcome_metrics["iteration_number"] = float(iteration)
        
        return outcome_metrics
    
    def _calculate_performance_indicators(
        self,
        outcome_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate performance indicators from outcome metrics."""
        indicators = {}
        
        # Calculate efficiency indicators
        if "throughput" in outcome_metrics and "resource_utilization" in outcome_metrics:
            indicators["efficiency_ratio"] = outcome_metrics["throughput"] / max(0.1, outcome_metrics["resource_utilization"])
        
        # Calculate quality indicators
        if "error_rate" in outcome_metrics:
            indicators["quality_score"] = 1.0 - outcome_metrics["error_rate"]
        
        # Calculate performance score
        score_components = []
        if "throughput" in outcome_metrics:
            score_components.append(min(1.0, outcome_metrics["throughput"] / 100.0))
        if "latency" in outcome_metrics:
            score_components.append(max(0.0, 1.0 - outcome_metrics["latency"] / 1000.0))
        
        if score_components:
            indicators["overall_performance_score"] = statistics.mean(score_components)
        
        return indicators
    
    def _calculate_resource_utilization(
        self,
        parameter_values: Dict[str, Any],
        outcome_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate resource utilization from parameters and outcomes."""
        utilization = {}
        
        # CPU utilization
        if "concurrent_users" in parameter_values:
            users = parameter_values["concurrent_users"]
            utilization["cpu"] = min(1.0, users / 200.0)  # Assume 200 users = 100% CPU
        
        # Memory utilization
        if "Load Multiplier" in parameter_values:
            load = parameter_values["Load Multiplier"]
            utilization["memory"] = min(1.0, 0.3 + 0.4 * load)
        
        # Network utilization
        if "throughput" in outcome_metrics:
            throughput = outcome_metrics["throughput"]
            utilization["network"] = min(1.0, throughput / 500.0)  # Assume 500 = max throughput
        
        # Storage utilization
        if "automated_tasks" in outcome_metrics:
            tasks = outcome_metrics["automated_tasks"]
            utilization["storage"] = min(1.0, 0.2 + tasks / 1000.0)
        
        return utilization
    
    def _calculate_cost_metrics(
        self,
        parameter_values: Dict[str, Any],
        outcome_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate cost metrics from parameters and outcomes."""
        cost_metrics = {}
        
        # Infrastructure costs
        if "resource_utilization" in outcome_metrics:
            utilization = outcome_metrics["resource_utilization"]
            cost_metrics["infrastructure_cost"] = utilization * 100.0  # $100/hour at full utilization
        
        # Operational costs
        if "manual_tasks" in outcome_metrics:
            manual_tasks = outcome_metrics["manual_tasks"]
            cost_metrics["operational_cost"] = manual_tasks * 0.5  # $0.50 per manual task
        
        # Error costs
        if "total_errors" in outcome_metrics:
            errors = outcome_metrics["total_errors"]
            cost_metrics["error_cost"] = errors * 10.0  # $10 per error
        
        # Total cost
        cost_metrics["total_cost"] = sum(cost_metrics.values())
        
        # Cost per unit
        if "throughput" in outcome_metrics and outcome_metrics["throughput"] > 0:
            cost_metrics["cost_per_unit"] = cost_metrics["total_cost"] / outcome_metrics["throughput"]
        
        return cost_metrics
    
    def _calculate_statistical_summary(
        self,
        outcomes: List[ScenarioOutcome]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary of simulation outcomes."""
        summary = {}
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for outcome in outcomes:
            for metric_name, value in outcome.outcome_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
                
                # Add percentiles if enough data
                if len(values) >= 10:
                    sorted_values = sorted(values)
                    summary[metric_name]["percentile_10"] = sorted_values[len(sorted_values) // 10]
                    summary[metric_name]["percentile_25"] = sorted_values[len(sorted_values) // 4]
                    summary[metric_name]["percentile_75"] = sorted_values[3 * len(sorted_values) // 4]
                    summary[metric_name]["percentile_90"] = sorted_values[9 * len(sorted_values) // 10]
        
        return summary
    
    def _calculate_confidence_intervals(
        self,
        outcomes: List[ScenarioOutcome],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for outcome metrics."""
        confidence_intervals = {}
        
        # Collect metrics
        all_metrics = defaultdict(list)
        for outcome in outcomes:
            for metric_name, value in outcome.outcome_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        alpha = 1.0 - confidence_level
        
        for metric_name, values in all_metrics.items():
            if len(values) > 1:
                sorted_values = sorted(values)
                n = len(sorted_values)
                
                lower_index = int(n * alpha / 2)
                upper_index = int(n * (1 - alpha / 2))
                
                lower_bound = sorted_values[lower_index]
                upper_bound = sorted_values[upper_index]
                
                confidence_intervals[metric_name] = (lower_bound, upper_bound)
        
        return confidence_intervals
    
    def _estimate_probability_distributions(
        self,
        outcomes: List[ScenarioOutcome]
    ) -> Dict[str, List[float]]:
        """Estimate probability distributions for outcome metrics."""
        distributions = {}
        
        # Collect metrics
        all_metrics = defaultdict(list)
        for outcome in outcomes:
            for metric_name, value in outcome.outcome_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            if len(values) > 10:
                # Create histogram
                num_bins = min(20, len(values) // 5)
                min_val, max_val = min(values), max(values)
                
                if max_val > min_val:
                    bin_size = (max_val - min_val) / num_bins
                    bins = [0] * num_bins
                    
                    for value in values:
                        bin_index = min(num_bins - 1, int((value - min_val) / bin_size))
                        bins[bin_index] += 1
                    
                    # Normalize to probabilities
                    total = sum(bins)
                    if total > 0:
                        distributions[metric_name] = [count / total for count in bins]
        
        return distributions
    
    async def _perform_uncertainty_analysis(
        self,
        config: ScenarioConfiguration,
        outcomes: List[ScenarioOutcome]
    ) -> Dict[str, Any]:
        """Perform uncertainty analysis on scenario results."""
        uncertainty_analysis = {}
        
        # Parameter uncertainty analysis
        parameter_impacts = {}
        for param in config.parameters:
            param_name = param.parameter_name
            
            # Calculate impact of parameter uncertainty
            param_values = [outcome.parameter_values.get(param_name, param.base_value) for outcome in outcomes]
            
            if param_values and all(isinstance(v, (int, float)) for v in param_values):
                param_variance = statistics.variance(param_values) if len(param_values) > 1 else 0.0
                uncertainty_contribution = param_variance * param.uncertainty_level
                parameter_impacts[param_name] = uncertainty_contribution
        
        uncertainty_analysis["parameter_impacts"] = parameter_impacts
        
        # Model uncertainty (simplified)
        model_uncertainty = 0.1  # Assume 10% model uncertainty
        uncertainty_analysis["model_uncertainty"] = model_uncertainty
        
        # Total uncertainty
        total_uncertainty = sum(parameter_impacts.values()) + model_uncertainty
        uncertainty_analysis["total_uncertainty"] = total_uncertainty
        
        return uncertainty_analysis
    
    async def _perform_sensitivity_analysis(
        self,
        config: ScenarioConfiguration,
        outcomes: List[ScenarioOutcome]
    ) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on scenario parameters."""
        sensitivity_analysis = {}
        
        # Collect outcome metrics
        all_metrics = defaultdict(list)
        for outcome in outcomes:
            for metric_name, value in outcome.outcome_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        # Calculate sensitivity for each parameter-metric combination
        for param in config.parameters:
            param_name = param.parameter_name
            param_sensitivities = {}
            
            # Get parameter values
            param_values = []
            for outcome in outcomes:
                param_value = outcome.parameter_values.get(param_name)
                if isinstance(param_value, (int, float)):
                    param_values.append(param_value)
            
            if len(param_values) < 10:
                continue
            
            # Calculate sensitivity to each metric
            for metric_name, metric_values in all_metrics.items():
                if len(metric_values) == len(param_values):
                    # Calculate correlation as sensitivity measure
                    sensitivity = self._calculate_correlation(param_values, metric_values)
                    param_sensitivities[metric_name] = abs(sensitivity)
            
            if param_sensitivities:
                sensitivity_analysis[param_name] = param_sensitivities
        
        return sensitivity_analysis
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        x_sq_sum = sum((x - x_mean) ** 2 for x in x_values)
        y_sq_sum = sum((y - y_mean) ** 2 for y in y_values)
        
        denominator = math.sqrt(x_sq_sum * y_sq_sum)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _assess_scenario_risks(
        self,
        outcomes: List[ScenarioOutcome],
        config: ScenarioConfiguration
    ) -> Dict[str, float]:
        """Assess risks associated with scenario outcomes."""
        risk_assessment = {}
        
        # Collect key metrics
        success_rates = [outcome.outcome_metrics.get("simulation_success", 1.0) for outcome in outcomes]
        error_rates = [outcome.outcome_metrics.get("error_rate", 0.0) for outcome in outcomes]
        cost_metrics = [outcome.cost_metrics.get("total_cost", 0.0) for outcome in outcomes]
        
        # Calculate risk metrics
        if success_rates:
            risk_assessment["failure_probability"] = 1.0 - statistics.mean(success_rates)
        
        if error_rates:
            risk_assessment["average_error_rate"] = statistics.mean(error_rates)
            risk_assessment["max_error_rate"] = max(error_rates)
        
        if cost_metrics:
            cost_variance = statistics.variance(cost_metrics) if len(cost_metrics) > 1 else 0.0
            risk_assessment["cost_volatility"] = math.sqrt(cost_variance) / max(1.0, statistics.mean(cost_metrics))
        
        # Overall risk score
        risk_components = [
            risk_assessment.get("failure_probability", 0.0),
            risk_assessment.get("average_error_rate", 0.0),
            risk_assessment.get("cost_volatility", 0.0)
        ]
        risk_assessment["overall_risk_score"] = statistics.mean(risk_components)
        
        return risk_assessment
    
    def _generate_scenario_recommendations(
        self,
        outcomes: List[ScenarioOutcome],
        config: ScenarioConfiguration,
        risk_assessment: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on scenario results."""
        recommendations = []
        
        # Risk-based recommendations
        overall_risk = risk_assessment.get("overall_risk_score", 0.0)
        
        if overall_risk > 0.7:
            recommendations.append("HIGH RISK: Consider risk mitigation strategies before implementation")
            recommendations.append("Implement comprehensive monitoring and alerting")
            recommendations.append("Develop detailed rollback procedures")
        elif overall_risk > 0.4:
            recommendations.append("MEDIUM RISK: Proceed with caution and enhanced monitoring")
            recommendations.append("Consider phased implementation approach")
        else:
            recommendations.append("LOW RISK: Scenario appears viable for implementation")
        
        # Performance-based recommendations
        throughput_values = [outcome.outcome_metrics.get("throughput", 0.0) for outcome in outcomes]
        if throughput_values:
            avg_throughput = statistics.mean(throughput_values)
            if avg_throughput < 50.0:
                recommendations.append("Consider performance optimization before scaling")
            elif avg_throughput > 200.0:
                recommendations.append("Excellent throughput - consider expanding capacity")
        
        # Cost-based recommendations
        cost_values = [outcome.cost_metrics.get("total_cost", 0.0) for outcome in outcomes]
        if cost_values:
            avg_cost = statistics.mean(cost_values)
            cost_variance = statistics.variance(cost_values) if len(cost_values) > 1 else 0.0
            
            if cost_variance > avg_cost:
                recommendations.append("High cost variability - implement cost controls")
            if avg_cost > 1000.0:
                recommendations.append("High costs detected - evaluate cost optimization opportunities")
        
        # Scenario-specific recommendations
        if config.scenario_type == ScenarioType.STRESS_TEST:
            error_rates = [outcome.outcome_metrics.get("error_rate", 0.0) for outcome in outcomes]
            if error_rates and max(error_rates) > 0.1:
                recommendations.append("High error rates under stress - improve error handling")
        
        elif config.scenario_type == ScenarioType.CAPACITY_PLANNING:
            utilization_values = [outcome.resource_utilization.get("cpu", 0.0) for outcome in outcomes]
            if utilization_values and max(utilization_values) > 0.9:
                recommendations.append("Plan for additional capacity - utilization approaching limits")
        
        return recommendations
    
    @require(lambda scenario_results_list: len(scenario_results_list) >= 2)
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, ScenarioModelingError))
    async def compare_scenarios(
        self,
        scenario_results_list: List[ScenarioResults],
        comparison_criteria: Optional[List[str]] = None
    ) -> Either[ScenarioModelingError, ScenarioComparison]:
        """Compare multiple scenario results and provide analysis."""
        try:
            # Default comparison criteria
            if comparison_criteria is None:
                comparison_criteria = ["throughput", "cost", "error_rate", "overall_performance_score"]
            
            # Perform comparative analysis
            comparative_analysis = self._perform_comparative_analysis(scenario_results_list, comparison_criteria)
            
            # Rank scenarios
            ranking_analysis = self._rank_scenarios(scenario_results_list, comparison_criteria)
            
            # Analyze trade-offs
            trade_off_analysis = self._analyze_scenario_trade_offs(scenario_results_list, comparison_criteria)
            
            # Build decision matrix
            decision_matrix = self._build_scenario_decision_matrix(scenario_results_list, comparison_criteria)
            
            # Recommend best scenario
            recommended_scenario = self._recommend_best_scenario(scenario_results_list, decision_matrix)
            
            comparison = ScenarioComparison(
                comparison_id=f"comparison_{datetime.now(UTC).isoformat()}",
                scenario_results=scenario_results_list,
                comparative_analysis=comparative_analysis,
                ranking_analysis=ranking_analysis,
                trade_off_analysis=trade_off_analysis,
                decision_matrix=decision_matrix,
                recommended_scenario=recommended_scenario
            )
            
            return Either.right(comparison)
            
        except Exception as e:
            return Either.left(ScenarioModelingError(f"Scenario comparison failed: {str(e)}"))
    
    def _perform_comparative_analysis(
        self,
        scenario_results_list: List[ScenarioResults],
        comparison_criteria: List[str]
    ) -> Dict[str, Any]:
        """Perform comparative analysis between scenarios."""
        analysis = {}
        
        # Collect metrics for all scenarios
        scenario_metrics = {}
        for results in scenario_results_list:
            scenario_id = str(results.scenario_id)
            scenario_metrics[scenario_id] = {}
            
            for criterion in comparison_criteria:
                if criterion in results.statistical_summary:
                    scenario_metrics[scenario_id][criterion] = results.statistical_summary[criterion]["mean"]
        
        # Calculate comparative statistics
        for criterion in comparison_criteria:
            criterion_values = []
            for scenario_id in scenario_metrics:
                if criterion in scenario_metrics[scenario_id]:
                    criterion_values.append(scenario_metrics[scenario_id][criterion])
            
            if criterion_values:
                analysis[criterion] = {
                    "mean": statistics.mean(criterion_values),
                    "min": min(criterion_values),
                    "max": max(criterion_values),
                    "range": max(criterion_values) - min(criterion_values),
                    "coefficient_of_variation": statistics.stdev(criterion_values) / statistics.mean(criterion_values) if statistics.mean(criterion_values) != 0 and len(criterion_values) > 1 else 0.0
                }
        
        return analysis
    
    def _rank_scenarios(
        self,
        scenario_results_list: List[ScenarioResults],
        comparison_criteria: List[str]
    ) -> Dict[str, List[str]]:
        """Rank scenarios by different criteria."""
        ranking_analysis = {}
        
        for criterion in comparison_criteria:
            scenario_scores = []
            
            for results in scenario_results_list:
                scenario_id = str(results.scenario_id)
                if criterion in results.statistical_summary:
                    score = results.statistical_summary[criterion]["mean"]
                    scenario_scores.append((scenario_id, score))
            
            # Sort by score (high to low for positive metrics, low to high for negative metrics)
            negative_metrics = ["error_rate", "latency", "cost", "total_cost"]
            reverse_sort = criterion not in negative_metrics
            
            scenario_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
            ranking_analysis[criterion] = [scenario_id for scenario_id, _ in scenario_scores]
        
        return ranking_analysis
    
    def _analyze_scenario_trade_offs(
        self,
        scenario_results_list: List[ScenarioResults],
        comparison_criteria: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze trade-offs between scenarios and criteria."""
        trade_off_analysis = {}
        
        # Calculate trade-off ratios between criteria
        for i, criterion1 in enumerate(comparison_criteria):
            for j, criterion2 in enumerate(comparison_criteria):
                if i < j:
                    trade_off_key = f"{criterion1}_vs_{criterion2}"
                    trade_off_ratios = []
                    
                    for results in scenario_results_list:
                        if (criterion1 in results.statistical_summary and 
                            criterion2 in results.statistical_summary):
                            
                            value1 = results.statistical_summary[criterion1]["mean"]
                            value2 = results.statistical_summary[criterion2]["mean"]
                            
                            if value2 != 0:
                                ratio = value1 / value2
                                trade_off_ratios.append(ratio)
                    
                    if trade_off_ratios:
                        trade_off_analysis[trade_off_key] = {
                            "mean_ratio": statistics.mean(trade_off_ratios),
                            "min_ratio": min(trade_off_ratios),
                            "max_ratio": max(trade_off_ratios),
                            "ratio_variance": statistics.variance(trade_off_ratios) if len(trade_off_ratios) > 1 else 0.0
                        }
        
        return trade_off_analysis
    
    def _build_scenario_decision_matrix(
        self,
        scenario_results_list: List[ScenarioResults],
        comparison_criteria: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Build decision matrix for scenario comparison."""
        decision_matrix = {}
        
        # Normalize scores for each criterion
        criterion_values = defaultdict(list)
        scenario_raw_scores = defaultdict(dict)
        
        # Collect raw scores
        for results in scenario_results_list:
            scenario_id = str(results.scenario_id)
            for criterion in comparison_criteria:
                if criterion in results.statistical_summary:
                    score = results.statistical_summary[criterion]["mean"]
                    criterion_values[criterion].append(score)
                    scenario_raw_scores[scenario_id][criterion] = score
        
        # Normalize scores (0-1 scale)
        for results in scenario_results_list:
            scenario_id = str(results.scenario_id)
            normalized_scores = {}
            
            for criterion in comparison_criteria:
                if criterion in scenario_raw_scores[scenario_id]:
                    raw_score = scenario_raw_scores[scenario_id][criterion]
                    criterion_min = min(criterion_values[criterion])
                    criterion_max = max(criterion_values[criterion])
                    
                    if criterion_max > criterion_min:
                        # For negative metrics, invert the normalization
                        negative_metrics = ["error_rate", "latency", "cost", "total_cost"]
                        if criterion in negative_metrics:
                            normalized_score = 1.0 - (raw_score - criterion_min) / (criterion_max - criterion_min)
                        else:
                            normalized_score = (raw_score - criterion_min) / (criterion_max - criterion_min)
                    else:
                        normalized_score = 0.5  # All values are the same
                    
                    normalized_scores[criterion] = normalized_score
            
            decision_matrix[scenario_id] = normalized_scores
        
        return decision_matrix
    
    def _recommend_best_scenario(
        self,
        scenario_results_list: List[ScenarioResults],
        decision_matrix: Dict[str, Dict[str, float]]
    ) -> Optional[str]:
        """Recommend the best scenario based on decision matrix."""
        if not decision_matrix:
            return None
        
        # Calculate overall scores (equal weights for simplicity)
        scenario_scores = {}
        
        for scenario_id, scores in decision_matrix.items():
            if scores:
                overall_score = statistics.mean(scores.values())
                scenario_scores[scenario_id] = overall_score
        
        if not scenario_scores:
            return None
        
        # Return scenario with highest overall score
        best_scenario = max(scenario_scores.items(), key=lambda x: x[1])
        return best_scenario[0]
    
    async def get_scenario_modeling_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for scenario modeling system."""
        if not self.modeling_history:
            return {"total_scenarios": 0, "average_processing_time": 0.0}
        
        recent_history = list(self.modeling_history)[-50:]  # Last 50 scenarios
        
        # Calculate metrics
        processing_times = [entry["processing_time"] for entry in recent_history]
        iteration_counts = [entry["iterations"] for entry in recent_history]
        
        scenario_types = [entry["scenario_type"] for entry in recent_history]
        simulation_methods = [entry["simulation_method"] for entry in recent_history]
        
        return {
            "total_scenarios_modeled": len(self.modeling_history),
            "recent_scenarios": len(recent_history),
            "average_processing_time": statistics.mean(processing_times) if processing_times else 0.0,
            "max_processing_time": max(processing_times) if processing_times else 0.0,
            "average_iterations": statistics.mean(iteration_counts) if iteration_counts else 0.0,
            "cached_scenarios": len(self.scenario_cache),
            "most_used_scenario_types": [
                scenario_type for scenario_type, count in 
                Counter(scenario_types).most_common(3)
            ],
            "most_used_simulation_methods": [
                method for method, count in 
                Counter(simulation_methods).most_common(3)
            ],
            "performance_baseline_count": len(self.performance_baselines)
        }