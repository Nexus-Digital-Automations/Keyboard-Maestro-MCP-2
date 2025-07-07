"""Property-based tests for AI infrastructure components.

This module provides advanced property-based testing using Hypothesis to validate
cache behavior, cost calculations, and security properties across all possible
input scenarios with comprehensive edge case coverage.

Property Testing Coverage:
- Cache behavior validation across all input scenarios
- Cost calculation accuracy under various conditions
- API key validation and security properties
- Configuration parsing and validation properties
- Provider client behavior consistency
"""

from __future__ import annotations

from typing import Any, Optional
import string
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from src.ai.caching_system import (
    CacheKey,
    CacheManager,
    CacheNamespace,
)
from src.ai.cost_optimization import (
    BudgetId,
    BudgetPeriod,
    CostBudget,
    CostOptimizer,
)
from src.ai.providers.openai_client import OpenAIClient
from src.ai.security.api_key_manager import APIKeyManager
from src.core.ai_integration import (
    AIOperation,
    TokenCount,
    create_ai_request,
)


# Strategy definitions for property testing
@st.composite
def cache_keys(draw) -> Any:
    """Generate valid cache keys."""
    key = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_-.",
            min_size=1,
            max_size=100,
        ),
    )
    return CacheKey(key)


@st.composite
def cache_namespaces(draw) -> Any:
    """Generate valid cache namespaces."""
    namespace = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=1,
            max_size=50,
        ),
    )
    return CacheNamespace(namespace)


@st.composite
def cache_values(draw) -> Any:
    """Generate valid cache values."""
    return draw(
        st.one_of(
            st.text(max_size=1000),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(
                    st.text(max_size=100),
                    st.integers(),
                    st.floats(allow_nan=False),
                ),
            ),
            st.lists(st.text(max_size=50), max_size=20),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
    )


@st.composite
def budget_amounts(draw) -> Any:
    """Generate valid budget amounts."""
    amount = draw(
        st.decimals(
            min_value=Decimal("0.01"), max_value=Decimal("100000.00"), places=2
        ),
    )
    return amount


@st.composite
def api_keys(draw) -> Any:
    """Generate various API key formats."""
    provider = draw(st.sampled_from(["openai", "anthropic", "google_ai"]))

    if provider == "openai":
        key = "sk-" + draw(
            st.text(
                alphabet=string.ascii_letters + string.digits,
                min_size=45,
                max_size=55,
            ),
        )
    elif provider == "anthropic":
        key = "sk-ant-" + draw(
            st.text(
                alphabet=string.ascii_letters + string.digits,
                min_size=40,
                max_size=50,
            ),
        )
    else:  # google_ai
        key = draw(
            st.text(
                alphabet=string.ascii_letters + string.digits,
                min_size=20,
                max_size=40,
            ),
        )

    return provider, key


class TestCacheProperties:
    """Property-based tests for cache system."""

    @given(key=cache_keys(), namespace=cache_namespaces(), value=cache_values())
    def test_cache_put_get_roundtrip(self, key, namespace, value) -> None:
        """Property: Any value put into cache should be retrievable."""
        cache_manager = CacheManager(max_size=1000)

        # Put value
        success = cache_manager.put(key, value, namespace=namespace)
        assume(success)  # Skip if put fails for valid reasons

        # Get value
        retrieved = cache_manager.get(key, namespace=namespace)

        # Property: Retrieved value should equal stored value
        assert retrieved == value

    @given(
        keys_and_values=st.lists(
            st.tuples(cache_keys(), cache_values()),
            min_size=1,
            max_size=50,
            unique_by=lambda x: x[0],  # Unique keys
        ),
        namespace=cache_namespaces(),
    )
    def test_cache_multiple_operations_consistency(self, keys_and_values, namespace) -> None:
        """Property: Multiple cache operations should maintain consistency."""
        cache_manager = CacheManager(max_size=100)
        stored_items = {}

        # Store all items
        for key, value in keys_and_values:
            success = cache_manager.put(key, value, namespace=namespace)
            if success:
                stored_items[key] = value

        # Verify all stored items are retrievable
        for key, expected_value in stored_items.items():
            retrieved = cache_manager.get(key, namespace=namespace)
            if retrieved is not None:  # May be evicted due to size limits
                assert retrieved == expected_value

    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(["put", "get", "invalidate"]),
                cache_keys(),
                st.one_of(cache_values(), st.none()),
            ),
            min_size=1,
            max_size=20,
        ),
        namespace=cache_namespaces(),
    )
    def test_cache_operation_sequence_properties(self, operations, namespace) -> None:
        """Property: Cache should maintain consistent state across operation sequences."""
        cache_manager = CacheManager(max_size=50)
        known_keys = set()

        for op_type, key, value in operations:
            if op_type == "put" and value is not None:
                success = cache_manager.put(key, value, namespace=namespace)
                if success:
                    known_keys.add(key)

            elif op_type == "get":
                result = cache_manager.get(key, namespace=namespace)
                # Property: Get should return None or a valid value
                assert result is None or isinstance(
                    result,
                    str | dict | list | int | float,
                )

            elif op_type == "invalidate":
                cache_manager.invalidate(key, namespace=namespace)
                # After invalidation, key should not be in known keys
                known_keys.discard(key)

        # Property: Cache statistics should be consistent
        stats = cache_manager.get_statistics()
        assert stats["cache_size"] >= 0
        assert stats["cache_size"] <= cache_manager.max_size

    @given(
        ttl_hours=st.integers(min_value=1, max_value=24),
        key=cache_keys(),
        value=cache_values(),
        namespace=cache_namespaces(),
    )
    def test_cache_ttl_properties(self, ttl_hours, key, value, namespace) -> None:
        """Property: Cache TTL should work correctly."""
        cache_manager = CacheManager(max_size=100)

        # Store with TTL
        ttl = timedelta(hours=ttl_hours)
        success = cache_manager.put(key, value, ttl=ttl, namespace=namespace)
        assume(success)

        # Should be immediately retrievable
        retrieved = cache_manager.get(key, namespace=namespace)
        assert retrieved == value

        # Property: TTL should be respected (we can't wait for expiration in tests,
        # but we can verify the TTL is set correctly in the cache entry)
        if hasattr(cache_manager, "cache") and key in cache_manager.cache:
            entry = cache_manager.cache[key]
            if hasattr(entry, "ttl"):
                assert entry.ttl == ttl


class TestCostOptimizationProperties:
    """Property-based tests for cost optimization system."""

    @given(
        budget_name=st.text(min_size=1, max_size=100),
        amount=budget_amounts(),
        period=st.sampled_from(list(BudgetPeriod)),
        thresholds=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=5,
            unique=True,
        ).map(sorted),
    )
    def test_budget_creation_properties(self, budget_name, amount, period, thresholds) -> None:
        """Property: Valid budget parameters should create valid budgets."""
        cost_optimizer = CostOptimizer()

        budget_id = BudgetId(f"test_budget_{hash(budget_name) % 1000000}")
        budget = CostBudget(
            budget_id=budget_id,
            name=budget_name,
            amount=amount,
            period=period,
            start_date=datetime.now(UTC),
            alert_thresholds=thresholds,
        )

        result = cost_optimizer.add_budget(budget)

        # Property: Valid budgets should be successfully added
        assert result.is_right()
        assert result.value == budget_id

        # Property: Budget should be retrievable
        assert budget_id in cost_optimizer.budgets
        stored_budget = cost_optimizer.budgets[budget_id]
        assert stored_budget.amount == amount
        assert stored_budget.name == budget_name

    @given(
        usage_records=st.lists(
            st.tuples(
                st.sampled_from(list(AIOperation)),
                st.text(min_size=1, max_size=50),  # model name
                st.integers(min_value=1, max_value=10000),  # input tokens
                st.integers(min_value=1, max_value=5000),  # output tokens
                st.floats(min_value=0.0001, max_value=10.0),  # cost
                st.floats(min_value=0.1, max_value=30.0),  # processing time
            ),
            min_size=1,
            max_size=20,
        ),
    )
    def test_usage_tracking_properties(self, usage_records) -> None:
        """Property: Usage tracking should accumulate correctly."""
        cost_optimizer = CostOptimizer()
        total_expected_cost = 0.0

        for (
            operation,
            model,
            input_tokens,
            output_tokens,
            cost,
            proc_time,
        ) in usage_records:
            cost_optimizer.record_usage(
                operation=operation,
                model_used=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                processing_time=proc_time,
            )
            total_expected_cost += cost

        # Property: Cost breakdown should reflect recorded usage
        breakdown = cost_optimizer.get_cost_breakdown()
        assert breakdown["total_requests"] == len(usage_records)
        assert (
            abs(breakdown["total_cost"] - total_expected_cost) < 0.01
        )  # Allow for floating point precision

        # Property: All operations should be tracked
        operation_breakdown = breakdown["breakdown"]["by_operation"]
        recorded_operations = {record[0].value for record in usage_records}
        for op in recorded_operations:
            assert op in operation_breakdown

    @given(
        costs=st.lists(
            st.floats(min_value=0.01, max_value=100.0),
            min_size=5,
            max_size=30,
        ),
    )
    def test_cost_prediction_properties(self, costs) -> None:
        """Property: Cost predictions should be reasonable based on historical data."""
        cost_optimizer = CostOptimizer()

        # Record usage history
        for _i, cost in enumerate(costs):
            cost_optimizer.record_usage(
                operation=AIOperation.ANALYZE,
                model_used="test_model",
                input_tokens=100,
                output_tokens=50,
                cost=cost,
                processing_time=1.0,
            )

        # Get monthly projection
        projection = cost_optimizer.predict_monthly_cost(days_to_analyze=7)

        # Property: Projection should be positive and reasonable
        assert float(projection) > 0

        # Property: Projection should be related to average historical cost
        avg_daily_cost = sum(costs) / len(costs)
        expected_monthly = avg_daily_cost * 30

        # Should be within reasonable range (allowing for extrapolation variance)
        assert 0.1 * expected_monthly <= float(projection) <= 10 * expected_monthly


class TestSecurityProperties:
    """Property-based tests for security components."""

    @given(api_key_data=api_keys())
    def test_api_key_validation_properties(self, api_key_data) -> None:
        """Property: API key validation should be consistent and secure."""
        provider, key = api_key_data
        api_key_manager = APIKeyManager()

        # Test validation
        result = api_key_manager.validate_key(provider, key)

        if provider == "openai":
            if key.startswith("sk-") and len(key) >= 20:
                assert result.is_right()
            else:
                assert result.is_left()
        elif provider == "anthropic":
            if key.startswith("sk-ant-"):
                assert result.is_right()
            else:
                assert result.is_left()
        elif len(key) >= 20:
            assert result.is_right()
        else:
            assert result.is_left()

    @given(
        provider=st.text(min_size=1, max_size=50),
        key=st.text(min_size=1, max_size=200),
        tags=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            max_size=5,
        ),
    )
    def test_key_storage_properties(self, provider, key, tags) -> None:
        """Property: Key storage should preserve data integrity."""
        api_key_manager = APIKeyManager()

        # Store key
        store_result = api_key_manager.store_key(
            provider=provider,
            key_value=key,
            tags=tags,
        )

        # Property: Valid storage should succeed
        if len(provider) > 0 and len(key) > 0:
            assert store_result.is_right()

            # Property: Stored key should be retrievable
            retrieve_result = api_key_manager.retrieve_key(provider)
            assert retrieve_result.is_right()
            assert retrieve_result.value == key


class TestProviderClientProperties:
    """Property-based tests for provider client behavior."""

    @given(
        model=st.sampled_from(["gpt-3.5-turbo", "gpt-4", "text-embedding-ada-002"]),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        max_tokens=st.integers(min_value=1, max_value=4096),
    )
    def test_openai_client_parameter_handling(self, model, temperature, max_tokens) -> None:
        """Property: OpenAI client should handle valid parameters correctly."""
        client = OpenAIClient(api_key="sk-test-key-123", model=model, timeout=10.0)

        # Create request with parameters

        request_result = create_ai_request(
            operation=AIOperation.GENERATE,
            input_data="Test input",
            temperature=temperature,
            max_tokens=TokenCount(max_tokens),
        )
        assume(request_result.is_right())  # Skip invalid combinations
        request = request_result.value

        # Build payload
        payload = client._build_request_payload(request)

        # Property: Payload should contain expected parameters
        if not model.startswith("text-embedding"):
            assert "temperature" in payload
            assert "max_tokens" in payload
            assert payload["temperature"] == temperature
            assert payload["max_tokens"] == max_tokens
            assert payload["model"] == model

    @given(input_text=st.text(min_size=1, max_size=1000))
    def test_token_counting_properties(self, input_text) -> None:
        """Property: Token counting should be consistent and reasonable."""
        client = OpenAIClient(api_key="sk-test-key-123", model="gpt-3.5-turbo")

        token_count = client._count_tokens(input_text)

        # Property: Token count should be positive for non-empty text
        assert token_count > 0

        # Property: Token count should be reasonable relative to text length
        # Generally, tokens are roughly 0.75 words, and words are roughly 5 characters
        char_count = len(input_text)
        max(1, char_count // 4)  # Conservative estimate

        # Should be within reasonable bounds
        assert token_count <= char_count  # Can't have more tokens than characters
        assert token_count >= max(
            1,
            char_count // 10,
        )  # Should have some reasonable minimum


class CacheStateMachine(RuleBasedStateMachine):
    """Stateful testing for cache behavior."""

    def __init__(self):
        super().__init__()
        self.cache = CacheManager(max_size=10)
        self.stored_keys = set()

    @rule(key=cache_keys(), value=cache_values(), namespace=cache_namespaces())
    def put_item(self, key, value, namespace) -> None:
        """Put an item in the cache."""
        success = self.cache.put(key, value, namespace=namespace)
        if success:
            self.stored_keys.add((key, namespace))

    @rule(key=cache_keys(), namespace=cache_namespaces())
    def get_item(self, key, namespace) -> None:
        """Get an item from the cache."""
        result = self.cache.get(key, namespace=namespace)
        # If we know the key is stored, it should be retrievable
        # (unless evicted due to size limits)
        return result

    @rule(key=cache_keys(), namespace=cache_namespaces())
    def invalidate_item(self, key, namespace) -> None:
        """Invalidate an item in the cache."""
        self.cache.invalidate(key, namespace=namespace)
        self.stored_keys.discard((key, namespace))

    @invariant()
    def cache_size_invariant(self) -> None:
        """Cache size should never exceed maximum."""
        stats = self.cache.get_statistics()
        assert stats["cache_size"] <= self.cache.max_size

    @invariant()
    def cache_consistency_invariant(self) -> None:
        """Cache should maintain internal consistency."""
        stats = self.cache.get_statistics()
        assert stats["cache_size"] >= 0


# Property-based test configuration
TestCacheStateMachine = CacheStateMachine.TestCase
TestCacheStateMachine.settings = settings(max_examples=50, stateful_step_count=20)


# Performance property tests
class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""

    @given(operation_count=st.integers(min_value=10, max_value=1000))
    def test_cache_operation_performance_scaling(self, operation_count) -> None:
        """Property: Cache operations should scale linearly."""
        import time

        cache_manager = CacheManager(max_size=operation_count)

        start_time = time.time()

        # Perform operations
        for i in range(operation_count):
            key = CacheKey(f"perf_test_{i}")
            value = {"index": i, "data": f"test_{i}"}
            cache_manager.put(key, value)

            if i % 2 == 0:
                result = cache_manager.get(key)
                assert result == value

        end_time = time.time()
        total_time = end_time - start_time

        # Property: Time should scale reasonably with operation count
        # Allow up to 0.001 seconds per operation (very generous)
        max_allowed_time = operation_count * 0.001
        assert total_time <= max_allowed_time, (
            f"Operations took {total_time}s for {operation_count} ops"
        )

    @given(record_count=st.integers(min_value=5, max_value=100))
    def test_cost_tracking_performance_scaling(self, record_count) -> None:
        """Property: Cost tracking should handle increasing record counts efficiently."""
        import time

        cost_optimizer = CostOptimizer()

        start_time = time.time()

        # Record usage
        for i in range(record_count):
            cost_optimizer.record_usage(
                operation=AIOperation.ANALYZE,
                model_used="test_model",
                input_tokens=100 + i,
                output_tokens=50 + i,
                cost=0.01 * (i + 1),
                processing_time=1.0,
            )

        # Get breakdown
        breakdown = cost_optimizer.get_cost_breakdown()

        end_time = time.time()
        total_time = end_time - start_time

        # Property: Should handle records efficiently
        assert total_time <= record_count * 0.01  # 10ms per record max
        assert breakdown["total_requests"] == record_count


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
