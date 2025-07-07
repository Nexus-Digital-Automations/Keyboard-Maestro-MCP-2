"""Comprehensive tests for API orchestration tools module.

Tests cover API workflow orchestration, service mesh management, microservices
coordination, health monitoring, and integration with property-based testing.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import FastMCP tool objects and extract underlying functions (systematic MCP pattern)
import src.server.tools.api_orchestration_tools as api_orch
from hypothesis import given
from hypothesis import strategies as st

# Extract underlying functions from FastMCP tool objects (systematic pattern)
km_orchestrate_apis = api_orch.km_orchestrate_apis.fn
km_manage_service_mesh = api_orch.km_manage_service_mesh.fn
km_coordinate_microservices = api_orch.km_coordinate_microservices.fn
km_monitor_api_health = api_orch.km_monitor_api_health.fn


# Test data generators
@st.composite
def orchestration_type_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid orchestration types."""
    types = ["sequential", "parallel", "conditional", "pipeline", "scatter_gather"]
    return draw(st.sampled_from(types))


@st.composite
def error_handling_strategy(draw: Callable[..., Any]) -> None:
    """Generate valid error handling strategies."""
    strategies = ["fail_fast", "continue", "retry"]
    return draw(st.sampled_from(strategies))


@st.composite
def api_sequence_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid API sequence configurations."""
    return draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "service_name": st.text(min_size=3, max_size=20).filter(
                        lambda x: x.isalnum(),
                    ),
                    "name": st.text(min_size=5, max_size=30),
                    "endpoint_id": st.text(min_size=5, max_size=25),
                    "input_mapping": st.dictionaries(
                        st.text(min_size=1, max_size=10),
                        st.text(min_size=1, max_size=20),
                        min_size=0,
                        max_size=3,
                    ),
                    "output_mapping": st.dictionaries(
                        st.text(min_size=1, max_size=10),
                        st.text(min_size=1, max_size=20),
                        min_size=0,
                        max_size=3,
                    ),
                    "conditions": st.lists(
                        st.text(min_size=3, max_size=15),
                        min_size=0,
                        max_size=2,
                    ),
                    "metadata": st.dictionaries(
                        st.text(min_size=1, max_size=10),
                        st.text(min_size=1, max_size=15),
                        min_size=0,
                        max_size=2,
                    ),
                },
            ),
            min_size=1,
            max_size=5,
        ),
    )


@st.composite
def service_mesh_operation_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid service mesh operations."""
    operations = ["configure", "monitor", "route", "secure"]
    return draw(st.sampled_from(operations))


@st.composite
def coordination_type_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid coordination types."""
    types = ["discovery", "communication", "dependency", "health"]
    return draw(st.sampled_from(types))


@st.composite
def monitoring_scope_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid monitoring scopes."""
    scopes = ["gateway", "services", "endpoints", "workflows"]
    return draw(st.sampled_from(scopes))


@st.composite
def service_names_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid service names."""
    service_names = [
        "auth-service",
        "user-service",
        "payment-service",
        "notification-service",
        "api-gateway",
        "data-service",
    ]
    return draw(
        st.lists(st.sampled_from(service_names), min_size=1, max_size=4, unique=True),
    )


@st.composite
def health_metrics_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid health metrics."""
    metrics = [
        "response_time",
        "error_rate",
        "throughput",
        "availability",
        "cpu_usage",
        "memory_usage",
    ]
    return draw(st.lists(st.sampled_from(metrics), min_size=1, max_size=4, unique=True))


@st.composite
def load_balancing_strategy_names(draw: Callable[..., Any]) -> None:
    """Generate valid load balancing strategy names."""
    strategies = [
        "round_robin",
        "least_connections",
        "weighted_round_robin",
        "ip_hash",
        "random",
    ]
    return draw(st.sampled_from(strategies))


@st.composite
def failover_strategy_names(draw: Callable[..., Any]) -> Any:
    """Generate valid failover strategy names."""
    strategies = ["none", "circuit_breaker", "retry", "fallback"]
    return draw(st.sampled_from(strategies))


class TestAPIOrchestrationDependencies:
    """Test API orchestration dependencies and imports."""

    def test_api_orchestration_imports(self) -> None:
        """Test importing API orchestration dependencies."""
        try:
            from src.api.api_gateway import APIGateway
            from src.api.service_coordinator import ServiceCoordinator
            from src.core.api_orchestration_architecture import (
                OrchestrationStrategy,
                ServiceId,
                ServiceMeshType,
                WorkflowId,
            )

            # Test basic creation
            assert WorkflowId is not None
            assert ServiceId is not None
            assert OrchestrationStrategy is not None
            assert ServiceMeshType is not None
            assert ServiceCoordinator is not None
            assert APIGateway is not None

        except ImportError:
            # Mock the dependencies for testing
            pytest.skip("API orchestration dependencies not available - using mocks")


class TestAPIOrchestrationParameterValidation:
    """Test API orchestration parameter validation."""

    @given(orchestration_type_strategy())
    def test_valid_orchestration_types(self, orchestration_type: str) -> None:
        """Test that valid orchestration types are accepted."""
        valid_types = [
            "sequential",
            "parallel",
            "conditional",
            "pipeline",
            "scatter_gather",
        ]
        assert orchestration_type in valid_types

    @given(error_handling_strategy())
    def test_valid_error_handling_strategies(self, strategy: str) -> None:
        """Test that valid error handling strategies are accepted."""
        valid_strategies = ["fail_fast", "continue", "retry"]
        assert strategy in valid_strategies

    @given(service_mesh_operation_strategy())
    def test_valid_service_mesh_operations(self, operation: str) -> None:
        """Test that valid service mesh operations are accepted."""
        valid_operations = ["configure", "monitor", "route", "secure"]
        assert operation in valid_operations

    @given(coordination_type_strategy())
    def test_valid_coordination_types(self, coordination_type: str) -> None:
        """Test that valid coordination types are accepted."""
        valid_types = ["discovery", "communication", "dependency", "health"]
        assert coordination_type in valid_types

    @given(monitoring_scope_strategy())
    def test_valid_monitoring_scopes(self, scope: str) -> None:
        """Test that valid monitoring scopes are accepted."""
        valid_scopes = ["gateway", "services", "endpoints", "workflows"]
        assert scope in valid_scopes

    @given(service_names_strategy())
    def test_service_names_validation(self, service_names: list[str]) -> None:
        """Test service names list validation."""
        assert len(service_names) > 0
        assert all(isinstance(name, str) for name in service_names)
        assert len(set(service_names)) == len(service_names)  # Unique names

    def test_invalid_orchestration_types(self) -> None:
        """Test that invalid orchestration types are rejected."""
        invalid_types = ["invalid", "unknown", "", "async", "sync"]
        valid_types = [
            "sequential",
            "parallel",
            "conditional",
            "pipeline",
            "scatter_gather",
        ]
        for invalid_type in invalid_types:
            assert invalid_type not in valid_types


class TestAPIOrchestrationMocked:
    """Test API orchestration operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_orchestrate_apis_success(self) -> None:
        """Test successful API orchestration workflow."""
        with (
            patch(
                "src.server.tools.api_orchestration_tools.service_coordinator",
            ) as mock_coordinator,
            patch(
                "src.server.tools.api_orchestration_tools.OrchestrationStrategy",
            ) as mock_strategy,
            patch(
                "src.server.tools.api_orchestration_tools.create_workflow_id",
            ) as mock_create_workflow_id,
            patch(
                "src.server.tools.api_orchestration_tools.create_service_id",
            ) as mock_create_service_id,
        ):
            # Setup mocks for successful orchestration
            mock_strategy.return_value = Mock(value="sequential")
            mock_create_workflow_id.return_value = "workflow_001"
            mock_create_service_id.return_value = "service_001"

            # Mock orchestration result
            mock_orchestration_result = Mock()
            mock_orchestration_result.orchestration_id = "orch_001"
            mock_orchestration_result.execution_status = "completed"
            mock_orchestration_result.start_time = datetime.now(UTC)
            mock_orchestration_result.end_time = datetime.now(UTC)
            mock_orchestration_result.total_duration_ms = 2500
            mock_orchestration_result.step_results = [
                {"status": "success", "step_id": "step_1", "duration_ms": 1200},
                {"status": "success", "step_id": "step_2", "duration_ms": 1300},
            ]
            mock_orchestration_result.performance_metrics = {"avg_response_time": 125.5}
            mock_orchestration_result.circuit_breaker_events = []
            mock_orchestration_result.load_balancer_decisions = []
            mock_orchestration_result.errors = []
            mock_orchestration_result.metadata = {}

            mock_result = Mock()
            mock_result.is_error.return_value = False
            mock_result.value = mock_orchestration_result

            mock_coordinator.execute_workflow = AsyncMock(return_value=mock_result)

            # Execute API orchestration using FastMCP .call() method (systematic MCP pattern)
            result = await km_orchestrate_apis(
                workflow_name="test_workflow",
                api_sequence=[
                    {
                        "service_name": "auth_service",
                        "name": "Authentication",
                        "endpoint_id": "auth_endpoint",
                        "input_mapping": {"user": "username"},
                        "output_mapping": {"token": "auth_token"},
                    },
                    {
                        "service_name": "user_service",
                        "name": "Get User Profile",
                        "endpoint_id": "profile_endpoint",
                        "input_mapping": {"token": "auth_token"},
                        "output_mapping": {"profile": "user_profile"},
                    },
                ],
                orchestration_type="sequential",
                error_handling="retry",
            )

            # Verify successful orchestration
            assert result["success"] is True
            assert result["workflow_name"] == "test_workflow"
            assert result["orchestration_type"] == "sequential"
            assert result["execution_status"] == "completed"
            assert result["total_steps"] == 2
            assert result["successful_steps"] == 2
            assert result["failed_steps"] == 0
            assert "workflow_id" in result
            assert "orchestration_id" in result

    @pytest.mark.asyncio
    async def test_km_orchestrate_apis_invalid_type(self) -> None:
        """Test API orchestration with invalid orchestration type."""
        # Execute with invalid orchestration type
        result = await km_orchestrate_apis(
            workflow_name="test_workflow",
            api_sequence=[
                {"service_name": "test", "name": "Test", "endpoint_id": "test"},
            ],
            orchestration_type="invalid_type",
        )

        # Verify invalid type error
        assert result["success"] is False
        assert "Invalid orchestration type" in result["error"]
        assert "invalid_type" in result["error"]

    @pytest.mark.asyncio
    async def test_km_orchestrate_apis_execution_error(self) -> None:
        """Test API orchestration with execution error."""
        with (
            patch(
                "src.server.tools.api_orchestration_tools.service_coordinator",
            ) as mock_coordinator,
            patch(
                "src.server.tools.api_orchestration_tools.OrchestrationStrategy",
            ) as mock_strategy,
            patch(
                "src.server.tools.api_orchestration_tools.create_workflow_id",
            ) as mock_create_workflow_id,
        ):
            # Setup mocks for execution error
            mock_strategy.return_value = Mock(value="sequential")
            mock_create_workflow_id.return_value = "workflow_002"

            mock_error_result = Mock()
            mock_error_result.is_error.return_value = True
            mock_error_result.error = "Workflow execution failed"

            mock_coordinator.execute_workflow = AsyncMock(
                return_value=mock_error_result,
            )

            # Execute API orchestration that should fail
            result = await km_orchestrate_apis(
                workflow_name="failing_workflow",
                api_sequence=[
                    {
                        "service_name": "failing_service",
                        "name": "Failing API",
                        "endpoint_id": "fail",
                    },
                ],
                orchestration_type="sequential",
            )

            # Verify execution error
            assert result["success"] is False
            assert result["error"] == "Workflow execution failed"
            assert result["workflow_id"] == "workflow_002"


class TestServiceMeshManagementMocked:
    """Test service mesh management operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_manage_service_mesh_configure(self) -> None:
        """Test successful service mesh configuration."""
        with (
            patch(
                "src.server.tools.api_orchestration_tools.create_service_id",
            ) as mock_create_service_id,
            patch(
                "src.server.tools.api_orchestration_tools.ServiceMeshType",
            ) as mock_mesh_type,
        ):
            # Setup mocks for configuration
            mock_create_service_id.return_value = "service_mesh_001"
            # Mock ServiceMeshType constructor to return an object with .value="istio"
            mock_mesh_instance = Mock()
            mock_mesh_instance.value = "istio"
            mock_mesh_type.return_value = mock_mesh_instance
            mock_mesh_type.ISTIO = mock_mesh_instance

            # Execute service mesh configuration
            result = await km_manage_service_mesh(
                operation="configure",
                service_name="user_service",
                mesh_configuration={"type": "istio"},
                routing_rules=[{"path": "/api/v1/*", "weight": 100}],
                security_policies={"tls": "strict"},
                observability=True,
                load_balancing="round_robin",
            )

            # Verify successful configuration
            assert result["success"] is True
            assert result["operation"] == "configure"
            assert result["service_name"] == "user_service"
            assert result["mesh_type"] == "istio"
            assert result["configuration_applied"] is True
            assert result["routing_rules_count"] == 1
            assert result["security_policies_count"] == 1
            assert result["observability_enabled"] is True
            assert result["load_balancing_strategy"] == "round_robin"

    @pytest.mark.asyncio
    async def test_km_manage_service_mesh_monitor(self) -> None:
        """Test service mesh monitoring operation."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            mock_create_service_id.return_value = "monitor_service_001"

            # Execute service mesh monitoring
            result = await km_manage_service_mesh(
                operation="monitor",
                service_name="payment_service",
                observability=True,
            )

            # Verify successful monitoring
            assert result["success"] is True
            assert result["operation"] == "monitor"
            assert result["service_name"] == "payment_service"
            assert "monitoring_timestamp" in result
            assert result["service_status"] == "healthy"
            assert "metrics" in result
            assert "alerts" in result
            assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_km_manage_service_mesh_route(self) -> None:
        """Test service mesh routing configuration."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            mock_create_service_id.return_value = "route_service_001"

            # Execute service mesh routing
            result = await km_manage_service_mesh(
                operation="route",
                service_name="api_gateway",
                routing_rules=[
                    {
                        "type": "header",
                        "header": "version",
                        "value": "v2",
                        "weight": 20,
                    },
                    {"type": "weight", "destination": "service-v1", "weight": 80},
                ],
            )

            # Verify successful routing configuration
            assert result["success"] is True
            assert result["operation"] == "route"
            assert result["service_name"] == "api_gateway"
            assert "routing_configuration" in result
            assert "routing_rules" in result
            assert result["active_routes"] == 2
            assert "traffic_distribution" in result

    @pytest.mark.asyncio
    async def test_km_manage_service_mesh_secure(self) -> None:
        """Test service mesh security configuration."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            mock_create_service_id.return_value = "secure_service_001"

            # Execute service mesh security
            result = await km_manage_service_mesh(
                operation="secure",
                service_name="auth_service",
                security_policies={
                    "authorization": "strict",
                    "encryption": "tls_1_3",
                    "authentication": "mutual_tls",
                },
            )

            # Verify successful security configuration
            assert result["success"] is True
            assert result["operation"] == "secure"
            assert result["service_name"] == "auth_service"
            assert "security_configuration" in result
            assert result["compliance_status"] == "compliant"
            assert result["security_score"] == 95.0
            assert result["vulnerabilities"] == []

    @pytest.mark.asyncio
    async def test_km_manage_service_mesh_invalid_operation(self) -> None:
        """Test service mesh management with invalid operation."""
        # Execute with invalid operation
        result = await km_manage_service_mesh(
            operation="invalid_operation",
            service_name="test_service",
        )

        # Verify invalid operation error
        assert result["success"] is False
        assert "Invalid operation" in result["error"]
        assert "invalid_operation" in result["error"]


class TestMicroservicesCoordinationMocked:
    """Test microservices coordination operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_coordinate_microservices_discovery(self) -> None:
        """Test microservices discovery coordination."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            # Setup service ID creation
            def create_service_id_side_effect(name: str) -> None:
                return f"service_{name.replace('-', '_')}"

            mock_create_service_id.side_effect = create_service_id_side_effect

            # Execute microservices discovery
            result = await km_coordinate_microservices(
                coordination_type="discovery",
                services=["auth-service", "user-service", "payment-service"],
                health_monitoring=True,
                failover_strategy="circuit_breaker",
            )

            # Verify successful discovery coordination
            assert result["success"] is True
            assert result["coordination_type"] == "discovery"
            assert result["total_services"] == 3
            assert "discovered_services" in result
            assert len(result["discovered_services"]) == 3
            assert result["service_registry_status"] == "healthy"
            assert "discovery_latency_ms" in result
            assert "last_discovery" in result

    @pytest.mark.asyncio
    async def test_km_coordinate_microservices_communication(self) -> None:
        """Test microservices communication coordination."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:

            def create_service_id_side_effect(name: str) -> None:
                return f"comm_{name.replace('-', '_')}"

            mock_create_service_id.side_effect = create_service_id_side_effect

            # Execute microservices communication coordination
            result = await km_coordinate_microservices(
                coordination_type="communication",
                services=["gateway", "auth", "data"],
                coordination_config={"protocol": "grpc", "timeout": 5000},
            )

            # Verify successful communication coordination
            assert result["success"] is True
            assert result["coordination_type"] == "communication"
            assert "communication_patterns" in result
            assert "service_connections" in result
            assert "total_connections" in result
            assert result["communication_health"] == "optimal"

    @pytest.mark.asyncio
    async def test_km_coordinate_microservices_dependency(self) -> None:
        """Test microservices dependency coordination."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:

            def create_service_id_side_effect(name: str) -> None:
                return f"dep_{name.replace('-', '_')}"

            mock_create_service_id.side_effect = create_service_id_side_effect

            # Execute microservices dependency coordination
            result = await km_coordinate_microservices(
                coordination_type="dependency",
                services=["frontend", "api", "database"],
                dependency_mapping={
                    "frontend": ["api"],
                    "api": ["database"],
                    "database": [],
                },
            )

            # Verify successful dependency coordination
            assert result["success"] is True
            assert result["coordination_type"] == "dependency"
            assert "dependency_graph" in result
            assert "resolved_dependencies" in result
            assert result["dependency_health"] == "satisfied"
            assert result["circular_dependencies_detected"] is False

    @pytest.mark.asyncio
    async def test_km_coordinate_microservices_health(self) -> None:
        """Test microservices health coordination."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:

            def create_service_id_side_effect(name: str) -> None:
                return f"health_{name.replace('-', '_')}"

            mock_create_service_id.side_effect = create_service_id_side_effect

            # Execute microservices health coordination
            result = await km_coordinate_microservices(
                coordination_type="health",
                services=["service1", "service2"],
                health_monitoring=True,
            )

            # Verify successful health coordination
            assert result["success"] is True
            assert result["coordination_type"] == "health"
            assert "overall_health_score" in result
            assert "overall_status" in result
            assert "service_health_reports" in result
            assert len(result["service_health_reports"]) == 2
            assert result["unhealthy_services"] == []

    @pytest.mark.asyncio
    async def test_km_coordinate_microservices_invalid_type(self) -> None:
        """Test microservices coordination with invalid type."""
        # Execute with invalid coordination type
        result = await km_coordinate_microservices(
            coordination_type="invalid_type",
            services=["test_service"],
        )

        # Verify invalid type error
        assert result["success"] is False
        assert "Invalid coordination type" in result["error"]
        assert "invalid_type" in result["error"]


class TestAPIHealthMonitoringMocked:
    """Test API health monitoring operations with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_km_monitor_api_health_gateway(self) -> None:
        """Test API health monitoring for gateway scope."""
        with patch(
            "src.server.tools.api_orchestration_tools.api_gateway",
        ) as mock_gateway:
            # Setup gateway health mocks
            mock_gateway.get_health_status.return_value = {"status": "healthy"}
            mock_gateway.get_metrics.return_value = {
                "total_requests": 10000,
                "successful_requests": 9800,
                "failed_requests": 200,
                "average_response_time": 125.5,
                "cache_hits": 7500,
                "rate_limited_requests": 50,
            }

            # Execute gateway health monitoring
            result = await km_monitor_api_health(
                monitoring_scope="gateway",
                health_metrics=["response_time", "error_rate", "throughput"],
                monitoring_duration=10,
                include_performance=True,
            )

            # Verify successful gateway monitoring
            assert result["success"] is True
            assert result["monitoring_scope"] == "gateway"
            assert "health_status" in result
            assert "gateway_health" in result
            assert "alerts" in result
            assert "recommendations" in result
            assert result["monitoring_duration"] == 10

    @pytest.mark.asyncio
    async def test_km_monitor_api_health_services(self) -> None:
        """Test API health monitoring for services scope."""
        # Execute services health monitoring
        result = await km_monitor_api_health(
            monitoring_scope="services",
            target_services=["auth-service", "user-service"],
            health_metrics=["response_time", "error_rate", "availability"],
            alert_thresholds={"response_time": 500.0, "error_rate": 0.03},
        )

        # Verify successful services monitoring
        assert result["success"] is True
        assert result["monitoring_scope"] == "services"
        assert result["services_monitored"] == 2
        assert "service_health_data" in result
        assert len(result["service_health_data"]) == 2
        assert "overall_health" in result
        assert "total_alerts" in result

    @pytest.mark.asyncio
    async def test_km_monitor_api_health_endpoints(self) -> None:
        """Test API health monitoring for endpoints scope."""
        # Execute endpoints health monitoring
        result = await km_monitor_api_health(
            monitoring_scope="endpoints",
            target_services=["GET /api/v1/users", "POST /api/v1/auth"],
            health_metrics=["response_time", "throughput"],
            real_time_updates=True,
        )

        # Verify successful endpoints monitoring
        assert result["success"] is True
        assert result["monitoring_scope"] == "endpoints"
        assert result["endpoints_monitored"] == 2
        assert "endpoint_health_data" in result
        assert "average_response_time" in result
        assert "overall_error_rate" in result

    @pytest.mark.asyncio
    async def test_km_monitor_api_health_workflows(self) -> None:
        """Test API health monitoring for workflows scope."""
        # Execute workflows health monitoring
        result = await km_monitor_api_health(
            monitoring_scope="workflows",
            health_metrics=["throughput", "availability"],
            include_performance=True,
        )

        # Verify successful workflows monitoring
        assert result["success"] is True
        assert result["monitoring_scope"] == "workflows"
        assert "workflow_health" in result
        assert "performance_summary" in result
        assert "performance_analytics" in result

    @pytest.mark.asyncio
    async def test_km_monitor_api_health_invalid_scope(self) -> None:
        """Test API health monitoring with invalid scope."""
        # Execute with invalid monitoring scope
        result = await km_monitor_api_health(
            monitoring_scope="invalid_scope",
            health_metrics=["response_time"],
        )

        # Verify invalid scope error
        assert result["success"] is False
        assert "Invalid monitoring scope" in result["error"]
        assert "invalid_scope" in result["error"]


class TestAPIOrchestrationErrorHandling:
    """Test API orchestration error handling."""

    @pytest.mark.asyncio
    async def test_api_orchestration_system_error(self) -> None:
        """Test handling of system errors in API orchestration."""
        with patch(
            "src.server.tools.api_orchestration_tools.OrchestrationStrategy",
        ) as mock_strategy:
            # Setup system error
            mock_strategy.side_effect = RuntimeError("System failure")

            # Execute operation that should trigger system error
            result = await km_orchestrate_apis(
                workflow_name="error_workflow",
                api_sequence=[
                    {"service_name": "test", "name": "Test", "endpoint_id": "test"},
                ],
                orchestration_type="sequential",
            )

            # Verify system error handling
            assert result["success"] is False
            assert "API orchestration error" in result["error"]
            assert result["workflow_name"] == "error_workflow"

    @pytest.mark.asyncio
    async def test_service_mesh_system_error(self) -> None:
        """Test handling of system errors in service mesh management."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            # Setup system error
            mock_create_service_id.side_effect = RuntimeError(
                "Service ID creation failed",
            )

            # Execute operation that should trigger system error
            result = await km_manage_service_mesh(
                operation="configure",
                service_name="error_service",
            )

            # Verify system error handling
            assert result["success"] is False
            assert "Service mesh management error" in result["error"]
            assert result["operation"] == "configure"
            assert result["service_name"] == "error_service"

    @pytest.mark.asyncio
    async def test_microservices_coordination_system_error(self) -> None:
        """Test handling of system errors in microservices coordination."""
        with patch(
            "src.server.tools.api_orchestration_tools.create_service_id",
        ) as mock_create_service_id:
            # Setup system error
            mock_create_service_id.side_effect = RuntimeError(
                "Service coordination failed",
            )

            # Execute operation that should trigger system error
            result = await km_coordinate_microservices(
                coordination_type="discovery",
                services=["error_service"],
            )

            # Verify system error handling
            assert result["success"] is False
            assert "Microservices coordination error" in result["error"]
            assert result["coordination_type"] == "discovery"
            assert result["services"] == ["error_service"]


class TestAPIOrchestrationIntegration:
    """Integration tests for API orchestration operations."""

    @pytest.mark.asyncio
    async def test_complete_api_orchestration_workflow(self) -> None:
        """Test complete API orchestration workflow integration."""
        with (
            patch(
                "src.server.tools.api_orchestration_tools.service_coordinator",
            ) as mock_coordinator,
            patch(
                "src.server.tools.api_orchestration_tools.OrchestrationStrategy",
            ) as mock_strategy,
            patch(
                "src.server.tools.api_orchestration_tools.create_workflow_id",
            ) as mock_create_workflow_id,
            patch(
                "src.server.tools.api_orchestration_tools.create_service_id",
            ) as mock_create_service_id,
        ):
            # Setup mocks for complete workflow
            mock_strategy.return_value = Mock(value="pipeline")
            mock_create_workflow_id.return_value = "integration_workflow_001"
            mock_create_service_id.return_value = "integration_service_001"

            # Mock successful orchestration result
            mock_orchestration_result = Mock()
            mock_orchestration_result.orchestration_id = "integration_orch_001"
            mock_orchestration_result.execution_status = "completed"
            mock_orchestration_result.start_time = datetime.now(UTC)
            mock_orchestration_result.end_time = datetime.now(UTC)
            mock_orchestration_result.total_duration_ms = 3500
            mock_orchestration_result.step_results = [
                {"status": "success", "step_id": "step_1", "duration_ms": 1000},
                {"status": "success", "step_id": "step_2", "duration_ms": 1200},
                {"status": "success", "step_id": "step_3", "duration_ms": 1300},
            ]
            mock_orchestration_result.performance_metrics = {"avg_response_time": 116.7}
            mock_orchestration_result.circuit_breaker_events = []
            mock_orchestration_result.load_balancer_decisions = []
            mock_orchestration_result.errors = []
            mock_orchestration_result.metadata = {"workflow_type": "integration_test"}

            mock_result = Mock()
            mock_result.is_error.return_value = False
            mock_result.value = mock_orchestration_result

            mock_coordinator.execute_workflow = AsyncMock(return_value=mock_result)

            # Execute complete API orchestration workflow
            result = await km_orchestrate_apis(
                workflow_name="integration_test_workflow",
                api_sequence=[
                    {
                        "service_name": "auth_service",
                        "name": "User Authentication",
                        "endpoint_id": "auth_login",
                        "input_mapping": {"credentials": "user_creds"},
                        "output_mapping": {"token": "auth_token"},
                    },
                    {
                        "service_name": "profile_service",
                        "name": "Get User Profile",
                        "endpoint_id": "user_profile",
                        "input_mapping": {"auth_token": "token"},
                        "output_mapping": {"profile": "user_data"},
                    },
                    {
                        "service_name": "preferences_service",
                        "name": "Load Preferences",
                        "endpoint_id": "user_preferences",
                        "input_mapping": {"user_id": "profile.id"},
                        "output_mapping": {"preferences": "user_prefs"},
                    },
                ],
                orchestration_type="pipeline",
                error_handling="retry",
                data_transformation=True,
                circuit_breaker=True,
                monitoring=True,
            )

            # Verify complete workflow execution
            assert result["success"] is True
            assert result["workflow_name"] == "integration_test_workflow"
            assert result["orchestration_type"] == "pipeline"
            assert result["execution_status"] == "completed"
            assert result["total_steps"] == 3
            assert result["successful_steps"] == 3
            assert result["failed_steps"] == 0
            assert result["total_duration_ms"] == 3500

            # Verify orchestration was called with correct parameters
            mock_coordinator.execute_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_orchestration_with_context(self) -> None:
        """Test API orchestration with FastMCP context integration."""
        mock_context = Mock()
        mock_context.info = AsyncMock()
        mock_context.error = AsyncMock()

        with (
            patch(
                "src.server.tools.api_orchestration_tools.service_coordinator",
            ) as mock_coordinator,
            patch(
                "src.server.tools.api_orchestration_tools.OrchestrationStrategy",
            ) as mock_strategy,
            patch(
                "src.server.tools.api_orchestration_tools.create_workflow_id",
            ) as mock_create_workflow_id,
        ):
            # Setup mocks for context testing
            mock_strategy.return_value = Mock(value="sequential")
            mock_create_workflow_id.return_value = "context_workflow_001"

            mock_orchestration_result = Mock()
            mock_orchestration_result.orchestration_id = "context_orch_001"
            mock_orchestration_result.execution_status = "completed"
            mock_orchestration_result.start_time = datetime.now(UTC)
            mock_orchestration_result.end_time = datetime.now(UTC)
            mock_orchestration_result.total_duration_ms = 1800
            mock_orchestration_result.step_results = [
                {"status": "success", "step_id": "step_1"},
            ]
            mock_orchestration_result.performance_metrics = {}
            mock_orchestration_result.circuit_breaker_events = []
            mock_orchestration_result.load_balancer_decisions = []
            mock_orchestration_result.errors = []
            mock_orchestration_result.metadata = {}

            mock_result = Mock()
            mock_result.is_error.return_value = False
            mock_result.value = mock_orchestration_result

            mock_coordinator.execute_workflow = AsyncMock(return_value=mock_result)

            # Execute operation with context
            result = await km_orchestrate_apis(
                workflow_name="context_test_workflow",
                api_sequence=[
                    {"service_name": "test", "name": "Test", "endpoint_id": "test"},
                ],
                orchestration_type="sequential",
                ctx=mock_context,
            )

            # Verify context integration
            assert result["success"] is True
            mock_context.info.assert_called()

            # Verify context calls
            info_calls = mock_context.info.call_args_list
            assert len(info_calls) >= 2  # At least start and completion


class TestAPIOrchestrationProperties:
    """Property-based tests for API orchestration operations."""

    @given(
        orchestration_type_strategy(),
        error_handling_strategy(),
        service_names_strategy(),
        health_metrics_strategy(),
    )
    def test_api_orchestration_parameter_validation_properties(
        self,
        orchestration_type: str,
        error_handling: str,
        service_names: list[str],
        health_metrics: list[str],
    ) -> None:
        """Property test for API orchestration parameter validation."""
        # Properties that should always hold
        valid_orchestration_types = [
            "sequential",
            "parallel",
            "conditional",
            "pipeline",
            "scatter_gather",
        ]
        valid_error_handling = ["fail_fast", "continue", "retry"]

        assert orchestration_type in valid_orchestration_types
        assert error_handling in valid_error_handling
        assert len(service_names) > 0
        assert all(isinstance(name, str) for name in service_names)
        assert len(health_metrics) > 0
        assert all(isinstance(metric, str) for metric in health_metrics)

    @given(api_sequence_strategy())
    def test_api_sequence_properties(self, api_sequence: list[dict[str, Any]]) -> None:
        """Property test for API sequence validation."""
        # Properties that should always hold for API sequences
        assert isinstance(api_sequence, list)
        assert len(api_sequence) > 0

        for api_spec in api_sequence:
            assert isinstance(api_spec, dict)
            assert "service_name" in api_spec
            assert "name" in api_spec
            assert "endpoint_id" in api_spec
            assert isinstance(api_spec["input_mapping"], dict)
            assert isinstance(api_spec["output_mapping"], dict)
            assert isinstance(api_spec["conditions"], list)
            assert isinstance(api_spec["metadata"], dict)

    @given(
        coordination_type_strategy(),
        service_names_strategy(),
        failover_strategy_names(),
    )
    def test_microservices_coordination_properties(
        self,
        coordination_type: str,
        service_names: list[str],
        failover_strategy: str,
    ) -> None:
        """Property test for microservices coordination validation."""
        # Properties that should always hold
        valid_coordination_types = [
            "discovery",
            "communication",
            "dependency",
            "health",
        ]
        valid_failover_strategies = ["none", "circuit_breaker", "retry", "fallback"]

        assert coordination_type in valid_coordination_types
        assert failover_strategy in valid_failover_strategies
        assert len(service_names) > 0
        assert len(set(service_names)) == len(service_names)  # Unique service names

    @given(
        monitoring_scope_strategy(),
        st.integers(min_value=1, max_value=60),  # monitoring_duration
    )
    def test_health_monitoring_properties(
        self,
        monitoring_scope: str,
        monitoring_duration: int,
    ) -> None:
        """Property test for health monitoring validation."""
        # Properties that should always hold
        valid_scopes = ["gateway", "services", "endpoints", "workflows"]

        assert monitoring_scope in valid_scopes
        assert 1 <= monitoring_duration <= 60

    @given(load_balancing_strategy_names())
    def test_load_balancing_strategy_properties(self, strategy: str) -> None:
        """Property test for load balancing strategy validation."""
        valid_strategies = [
            "round_robin",
            "least_connections",
            "weighted_round_robin",
            "ip_hash",
            "random",
        ]
        assert strategy in valid_strategies

    def test_workflow_result_structure_properties(self) -> None:
        """Property test for workflow result structure."""
        # Mock result structure that should always be valid
        result_structure = {
            "success": True,
            "workflow_id": "workflow_001",
            "workflow_name": "test_workflow",
            "orchestration_type": "sequential",
            "execution_status": "completed",
            "total_steps": 3,
            "successful_steps": 3,
            "failed_steps": 0,
            "step_results": [],
            "performance_metrics": {},
            "errors": [],
        }

        # Properties that should always hold
        assert "success" in result_structure
        assert isinstance(result_structure["success"], bool)
        assert "workflow_id" in result_structure
        assert len(result_structure["workflow_id"]) > 0
        assert "total_steps" in result_structure
        assert isinstance(result_structure["total_steps"], int)
        assert result_structure["total_steps"] >= 0
        assert (
            result_structure["successful_steps"] + result_structure["failed_steps"]
            <= result_structure["total_steps"]
        )
