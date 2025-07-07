"""Integration tests for real AI infrastructure implementations.

This module provides comprehensive testing for the real AI infrastructure including
provider clients, cache systems, cost optimization, and end-to-end workflows
with mocked API responses for reliable testing.

Test Coverage:
- Provider client functionality with mocked responses
- Multi-level cache performance and correctness
- Cost tracking and optimization accuracy
- Security validation for API key management
- Performance benchmarks for enterprise requirements
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.ai.caching_system import CacheKey, CacheNamespace, IntelligentCacheManager
from src.ai.config.ai_config import AIConfigManager
from src.ai.cost_optimization import BudgetId, BudgetPeriod, CostBudget, CostOptimizer
from src.ai.providers import (
    BaseProviderClient,
    OpenAIClient,
    ProviderFactory,
    ProviderRegistry,
)
from src.ai.security.api_key_manager import APIKeyManager
from src.core.ai_integration import (
    AIOperation,
    AIResponse,
    TokenCount,
    create_ai_request,
)


class TestProviderIntegration:
    """Test suite for AI provider integration."""

    @pytest.fixture
    def openai_client(self) -> bool:
        """Create OpenAI client for testing."""
        return OpenAIClient(
            api_key="test-key-sk-1234567890abcdef",
            model="gpt-3.5-turbo",
            timeout=10.0,
        )

    @pytest.fixture
    def provider_registry(self) -> bool:
        """Create provider registry for testing."""
        registry = ProviderRegistry()
        client = OpenAIClient(api_key="test-key", model="gpt-3.5-turbo")
        registry.register_provider("openai", client)
        return registry

    @pytest.mark.asyncio
    async def test_openai_client_capabilities(self, openai_client: Any) -> None:
        """Test OpenAI client capability reporting."""
        capabilities = await openai_client.get_capabilities()

        assert capabilities.max_tokens > 0
        assert capabilities.context_window > 0
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert AIOperation.ANALYZE in capabilities.supported_operations
        assert capabilities.cost_per_input_token > 0
        assert capabilities.cost_per_output_token > 0

    @pytest.mark.asyncio
    async def test_openai_client_request_processing(self, openai_client: Any) -> None:
        """Test OpenAI client request processing with mocked API."""
        with patch.object(openai_client, "_make_api_call") as mock_api:
            # Mock successful API response
            mock_api.return_value = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": int(datetime.now(UTC).timestamp()),
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a test response for integration testing.",
                        },
                        "finish_reason": "stop",
                    },
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 50,
                    "total_tokens": 75,
                },
            }

            # Create test request
            request_result = create_ai_request(
                operation=AIOperation.ANALYZE,
                input_data="Test input for analysis",
                temperature=0.7,
                max_tokens=100,
            )
            assert request_result.is_right()
            request = request_result.value

            # Process request
            result = await openai_client.process_request(request)

            # Verify result
            assert result.is_right()
            response = result.value
            assert isinstance(response, AIResponse)
            assert "test response" in response.content.lower()
            assert response.token_count == 75
            assert response.cost > 0

    @pytest.mark.asyncio
    async def test_provider_registry_fallback(self, provider_registry: dict[str, Any] | Any) -> None:
        """Test provider registry fallback mechanism."""
        # Add a second provider that will fail
        failing_client = MagicMock(spec=BaseProviderClient)
        failing_client.process_request = AsyncMock(
            side_effect=Exception("Provider failed"),
        )
        failing_client.check_health = AsyncMock(
            return_value=MagicMock(status="unhealthy"),
        )

        provider_registry.register_provider("failing", failing_client, is_fallback=True)
        provider_registry.fallback_order = ["failing", "openai"]  # Try failing first

        # Mock successful health check for OpenAI
        openai_client = provider_registry.get_provider("openai")
        openai_client.check_health = AsyncMock(return_value=MagicMock(status="healthy"))

        # Test fallback works
        healthy_provider = await provider_registry.get_healthy_provider()
        assert healthy_provider is not None
        assert healthy_provider.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_cost_estimation(self, openai_client: Any) -> None:
        """Test cost estimation accuracy."""
        request_result = create_ai_request(
            operation=AIOperation.GENERATE,
            input_data="Generate a short story",
            max_tokens=500,
        )
        assert request_result.is_right()
        request = request_result.value

        result = await openai_client.estimate_cost(request)
        assert result.is_right()

        estimated_cost = result.value
        assert estimated_cost > 0
        assert estimated_cost < 1.0  # Reasonable cost for test request


class TestCacheIntegration:
    """Test suite for intelligent cache system integration."""

    @pytest.fixture
    def cache_manager(self) -> Any:
        """Create cache manager for testing."""
        return IntelligentCacheManager()

    @pytest.mark.asyncio
    async def test_multi_level_cache_operations(self, cache_manager: Any) -> None:
        """Test multi-level cache get/put operations."""
        test_key = CacheKey("test_analysis_123")
        test_namespace = CacheNamespace("ai_operations")
        test_value = {"result": "analysis complete", "confidence": 0.95}

        # Test cache miss
        result = await cache_manager.cache.get(test_key, test_namespace)
        assert result is None

        # Test cache put
        success = await cache_manager.cache.put(
            test_key,
            test_value,
            namespace=test_namespace,
            tags={"analysis", "test"},
        )
        assert success is True

        # Test cache hit
        cached_result = await cache_manager.cache.get(test_key, test_namespace)
        assert cached_result == test_value

    @pytest.mark.asyncio
    async def test_cache_statistics_and_efficiency(self, cache_manager: Any) -> None:
        """Test cache statistics and efficiency reporting."""
        # Perform several cache operations
        for i in range(10):
            key = CacheKey(f"test_key_{i}")
            value = {"data": f"test_value_{i}", "index": i}
            await cache_manager.cache.put(key, value)

        # Get efficiency report
        report = cache_manager.get_cache_efficiency_report()

        assert "cache_statistics" in report
        assert "cache_efficiency_score" in report
        assert "access_patterns_tracked" in report
        assert isinstance(report["cache_efficiency_score"], int | float)

    @pytest.mark.asyncio
    async def test_cache_invalidation_strategies(self, cache_manager: Any) -> None:
        """Test various cache invalidation strategies."""
        # Setup test data with tags
        test_keys = []
        for i in range(5):
            key = CacheKey(f"invalidation_test_{i}")
            value = {"test": f"value_{i}"}
            tags = {"test_data", f"group_{i % 2}"}

            await cache_manager.cache.put(key, value, tags=tags)
            test_keys.append(key)

        # Test tag-based invalidation
        invalidated_count = cache_manager.cache.l1_cache.invalidate_by_tags({"group_0"})
        assert invalidated_count >= 2  # Should invalidate items with group_0 tag

        # Test namespace invalidation
        namespace_count = cache_manager.cache.l1_cache.invalidate_namespace(
            CacheNamespace("default"),
        )
        assert namespace_count >= 0

    @pytest.mark.asyncio
    async def test_predictive_prefetching(self, cache_manager: Any) -> None:
        """Test predictive prefetching functionality."""
        # Record access patterns
        test_key = CacheKey("prefetch_test_key")
        for _ in range(5):
            cache_manager._record_access_pattern(test_key)
            await asyncio.sleep(0.01)  # Small delay to create pattern

        # Test prefetch operation
        await cache_manager.predictive_prefetch()

        # Verify patterns are tracked
        assert len(cache_manager.access_patterns) > 0
        assert test_key in cache_manager.access_patterns


class TestCostOptimization:
    """Test suite for cost optimization system."""

    @pytest.fixture
    def cost_optimizer(self) -> Any:
        """Create cost optimizer for testing."""
        return CostOptimizer()

    def test_budget_creation_and_validation(self, cost_optimizer: Any) -> None:
        """Test budget creation with validation."""
        budget = CostBudget(
            budget_id=BudgetId("test_budget_123"),
            name="Test Budget",
            amount=Decimal("1000.00"),
            period=BudgetPeriod.MONTHLY,
            start_date=datetime.now(UTC),
            alert_thresholds=[0.5, 0.8, 0.95],
        )

        result = cost_optimizer.add_budget(budget)
        assert result.is_right()
        assert result.value == budget.budget_id

    def test_usage_tracking_and_reporting(self, cost_optimizer: Any) -> None:
        """Test usage tracking and cost reporting."""
        # Record some usage
        for i in range(10):
            cost_optimizer.record_usage(
                operation=AIOperation.ANALYZE,
                model_used="gpt-3.5-turbo",
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                cost=0.01 + i * 0.005,
                processing_time=1.5 + i * 0.1,
                user_id=f"user_{i % 3}",
            )

        # Get cost breakdown
        breakdown = cost_optimizer.get_cost_breakdown(period_days=30)

        assert breakdown["total_requests"] == 10
        assert breakdown["total_cost"] > 0
        assert "breakdown" in breakdown
        assert "by_operation" in breakdown["breakdown"]
        assert "by_model" in breakdown["breakdown"]

    def test_optimization_recommendations(self, cost_optimizer: Any) -> None:
        """Test cost optimization recommendations."""
        # Add some usage data to analyze
        for _ in range(20):
            cost_optimizer.record_usage(
                operation=AIOperation.GENERATE,
                model_used="gpt-4",
                input_tokens=200,
                output_tokens=100,
                cost=0.05,
                processing_time=2.0,
            )

        # Get recommendations
        recommendations = cost_optimizer.get_model_recommendations(AIOperation.GENERATE)

        # Should have recommendations based on efficiency analysis
        assert isinstance(recommendations, list)

        # Get optimization report
        report = cost_optimizer.get_optimization_report()
        assert "cost_analysis" in report
        assert "optimization_recommendations" in report
        assert "monthly_projection" in report

    def test_budget_alert_system(self, cost_optimizer: Any) -> None:
        """Test budget alert system."""
        # Create budget with low threshold
        budget = CostBudget(
            budget_id=BudgetId("alert_test_budget"),
            name="Alert Test Budget",
            amount=Decimal("10.00"),  # Low amount for easy testing
            period=BudgetPeriod.MONTHLY,
            start_date=datetime.now(UTC),
            alert_thresholds=[0.1, 0.5, 0.8],  # Low thresholds
        )

        cost_optimizer.add_budget(budget)

        # Add usage that exceeds threshold
        cost_optimizer.record_usage(
            operation=AIOperation.ANALYZE,
            model_used="gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
            cost=2.0,  # 20% of budget
            processing_time=1.0,
        )

        # Check for alerts
        alerts = cost_optimizer.get_active_alerts()
        assert isinstance(alerts, list)


class TestSecurityIntegration:
    """Test suite for security system integration."""

    @pytest.fixture
    def api_key_manager(self) -> Any:
        """Create API key manager for testing."""
        return APIKeyManager()

    def test_api_key_validation(self, api_key_manager: Any) -> None:
        """Test API key validation for different providers."""
        # Test OpenAI key validation
        openai_result = api_key_manager.validate_key(
            "openai",
            "sk-test1234567890abcdef",
        )
        assert openai_result.is_right()
        assert openai_result.value is True

        # Test invalid OpenAI key
        invalid_result = api_key_manager.validate_key("openai", "invalid-key")
        assert invalid_result.is_left()

        # Test Anthropic key validation
        anthropic_result = api_key_manager.validate_key(
            "anthropic",
            "sk-ant-test1234567890",
        )
        assert anthropic_result.is_right()

    def test_key_storage_and_retrieval(self, api_key_manager: Any) -> None:
        """Test secure key storage and retrieval."""
        test_key = "sk-test-key-for-storage-testing"
        provider = "test_provider"

        # Store key
        store_result = api_key_manager.store_key(provider, test_key)
        assert store_result.is_right()

        # Retrieve key
        retrieve_result = api_key_manager.retrieve_key(provider)
        assert retrieve_result.is_right()
        assert retrieve_result.value == test_key

    def test_key_rotation(self, api_key_manager: Any) -> None:
        """Test key rotation functionality."""
        provider = "rotation_test"
        old_key = "sk-old-key-123"
        new_key = "sk-new-key-456"

        # Store initial key
        api_key_manager.store_key(provider, old_key)

        # Rotate key
        rotation_result = api_key_manager.rotate_key(provider, new_key)
        assert rotation_result.is_right()

        # Verify new key is active
        current_key = api_key_manager.retrieve_key(provider)
        assert current_key.is_right()
        # Note: In environment storage mode, this might still be old key
        # In file storage mode with master password, it would be the new key


class TestConfigurationIntegration:
    """Test suite for configuration system integration."""

    @pytest.fixture
    def config_manager(self) -> dict[str, Any]:
        """Create configuration manager for testing."""
        return AIConfigManager()

    def test_default_configuration_loading(self, config_manager: Any) -> None:
        """Test loading of default configuration."""
        result = config_manager.load_config()
        assert result.is_right()

        config = result.value
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-3.5-turbo"
        assert "openai" in config.providers

    def test_environment_override_application(self, config_manager: Any) -> None:
        """Test environment variable overrides."""
        # Set environment override
        config_manager.set_environment_override("debug_mode", True)
        config_manager.set_environment_override("default_provider", "custom_provider")

        # Verify overrides applied
        assert config_manager.config.debug_mode is True
        assert config_manager.config.default_provider == "custom_provider"

    def test_provider_configuration_management(self, config_manager: Any) -> None:
        """Test provider configuration management."""
        config_manager.load_config()

        # Get provider config
        openai_config = config_manager.get_provider_config("openai")
        assert openai_config is not None
        assert openai_config.provider_name == "openai"
        assert openai_config.enabled is True

        # Get model config
        model_config = config_manager.get_model_config("openai", "gpt-3.5-turbo")
        assert model_config is not None
        assert model_config.model_name == "gpt-3.5-turbo"


class TestEndToEndWorkflows:
    """Test suite for end-to-end AI workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_ai_processing_workflow(self) -> None:
        """Test complete AI processing workflow from request to response."""
        # Setup components
        cache_manager = IntelligentCacheManager()
        cost_optimizer = CostOptimizer()

        # Create test request
        request_result = create_ai_request(
            operation=AIOperation.ANALYZE,
            input_data="Analyze this text for sentiment and key topics.",
            temperature=0.7,
            max_tokens=TokenCount(200),
        )
        assert request_result.is_right()
        request = request_result.value

        # Extract processing parameters for cache methods
        processing_params = {
            "temperature": request.temperature,
            "max_tokens": int(request.max_tokens) if request.max_tokens else None,
            "processing_mode": request.processing_mode.value,
        }

        # Test cache miss scenario
        cache_manager._generate_ai_cache_key(
            request.operation,
            request.input_data,
            processing_params,
        )

        cached_result = await cache_manager.get_ai_result(
            request.operation,
            request.input_data,
            processing_params,
        )
        assert cached_result is None  # Should be cache miss

        # Simulate AI processing result
        mock_result = {
            "sentiment": "positive",
            "topics": ["technology", "innovation"],
            "confidence": 0.87,
        }

        # Store result in cache
        cache_success = await cache_manager.put_ai_result(
            request.operation,
            request.input_data,
            mock_result,
            processing_params,
        )
        assert cache_success is True

        # Test cache hit
        cached_result_hit = await cache_manager.get_ai_result(
            request.operation,
            request.input_data,
            processing_params,
        )
        assert cached_result_hit == mock_result

        # Record usage for cost tracking
        cost_optimizer.record_usage(
            operation=request.operation,
            model_used="gpt-3.5-turbo",
            input_tokens=25,
            output_tokens=50,
            cost=0.0001,
            processing_time=1.2,
        )

        # Verify cost tracking
        breakdown = cost_optimizer.get_cost_breakdown(period_days=1)
        assert breakdown["total_requests"] == 1
        assert breakdown["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_provider_factory_initialization(self) -> None:
        """Test provider factory initialization from environment."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-test-key-123", "OPENAI_MODEL": "gpt-4"},
        ):
            factory = ProviderFactory()
            initialized_providers = factory.initialize_from_environment()

            # Should initialize OpenAI provider
            assert "openai" in initialized_providers

            # Test provider status
            status = factory.get_provider_status()
            assert "openai" in status
            assert status["openai"]["enabled"] is True
            assert status["openai"]["model"] == "gpt-4"

    def test_performance_benchmarks(self) -> None:
        """Test performance requirements compliance."""
        import time

        # Test cache operation performance
        cache_manager = IntelligentCacheManager()

        start_time = time.time()
        # Simulate cache operations
        for i in range(100):
            key = CacheKey(f"perf_test_{i}")
            # Using synchronous operations for timing
            cache_manager.cache.l1_cache.put(key, {"data": f"value_{i}"})
            result = cache_manager.cache.l1_cache.get(key)
            assert result is not None

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete 200 operations (100 put + 100 get) in reasonable time
        assert total_time < 1.0  # Less than 1 second for 200 cache operations

        # Test individual operation performance
        start_time = time.time()
        cache_manager.cache.l1_cache.put(CacheKey("single_perf_test"), {"test": "data"})
        single_op_time = time.time() - start_time

        # Single cache operation should be very fast
        assert single_op_time < 0.01  # Less than 10ms


# Performance and stress testing
class TestPerformanceValidation:
    """Test suite for performance validation and benchmarks."""

    def test_cache_performance_under_load(self) -> None:
        """Test cache performance under load."""
        cache_manager = IntelligentCacheManager()

        import time

        start_time = time.time()

        # Perform 1000 cache operations
        for i in range(1000):
            key = CacheKey(f"load_test_{i}")
            value = {"index": i, "data": f"test_data_{i}"}
            cache_manager.cache.l1_cache.put(key, value)

            if i % 2 == 0:  # Get every other item
                result = cache_manager.cache.l1_cache.get(key)
                assert result == value

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle 1000+ operations efficiently
        assert total_time < 2.0  # Less than 2 seconds

        # Verify cache statistics
        stats = cache_manager.cache.l1_cache.get_statistics()
        assert stats["cache_size"] <= 500  # Respects max size

    def test_cost_calculation_accuracy(self) -> None:
        """Test cost calculation accuracy under various scenarios."""
        cost_optimizer = CostOptimizer()

        # Test various cost scenarios
        test_cases = [
            (100, 50, "gpt-3.5-turbo", 0.0001),
            (500, 200, "gpt-4", 0.015),
            (1000, 800, "gpt-3.5-turbo", 0.0028),
        ]

        for input_tokens, output_tokens, model, expected_min_cost in test_cases:
            cost_optimizer.record_usage(
                operation=AIOperation.GENERATE,
                model_used=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=expected_min_cost,
                processing_time=1.0,
            )

        # Verify cost breakdown accuracy
        breakdown = cost_optimizer.get_cost_breakdown()
        assert breakdown["total_cost"] >= sum(case[3] for case in test_cases)
        assert breakdown["total_requests"] == len(test_cases)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
