"""Performance benchmark tests for AI infrastructure components.

This module provides comprehensive performance testing and validation for the AI
infrastructure including cache operations, cost calculations, provider clients,
and end-to-end workflows with enterprise-grade performance requirements.

Performance Requirements:
- Cache operations: <10ms for L1, <100ms for L2/L3
- Cost calculations: <5ms for standard operations
- Provider response times: <500ms for cached, <2s for API calls
- Memory efficiency: <100MB for standard operations
- Throughput: >100 operations/second sustained
"""

from __future__ import annotations

from typing import Any, Optional
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import psutil
import pytest
from src.ai.caching_system import (
    CacheKey,
    CacheManager,
    IntelligentCacheManager,
)
from src.ai.cost_optimization import (
    CostOptimizer,
)
from src.ai.providers.openai_client import OpenAIClient
from src.ai.security.api_key_manager import APIKeyManager
from src.core.ai_integration import (
    AIOperation,
    TokenCount,
    create_ai_request,
)


class TestCachePerformance:
    """Performance tests for caching system."""

    def test_l1_cache_performance_requirements(self) -> None:
        """Test L1 cache meets <10ms performance requirements."""
        cache_manager = CacheManager(max_size=1000)

        # Test single operation performance
        key = CacheKey("performance_test_key")
        value = {"test": "data", "timestamp": datetime.now(UTC).isoformat()}

        # Test PUT performance
        start_time = time.time()
        success = cache_manager.put(key, value)
        put_time = time.time() - start_time

        assert success is True
        assert put_time < 0.01  # <10ms requirement

        # Test GET performance
        start_time = time.time()
        result = cache_manager.get(key)
        get_time = time.time() - start_time

        assert result == value
        assert get_time < 0.01  # <10ms requirement

    def test_cache_bulk_operations_performance(self) -> None:
        """Test cache performance under bulk operations."""
        cache_manager = CacheManager(max_size=10000)

        # Test bulk PUT operations
        start_time = time.time()

        for i in range(1000):
            key = CacheKey(f"bulk_test_{i}")
            value = {
                "index": i,
                "data": f"test_data_{i}",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            cache_manager.put(key, value)

        bulk_put_time = time.time() - start_time

        # Should handle 1000 operations in reasonable time
        assert bulk_put_time < 1.0  # <1000ms for 1000 operations

        # Test bulk GET operations
        start_time = time.time()

        for i in range(1000):
            key = CacheKey(f"bulk_test_{i}")
            result = cache_manager.get(key)
            assert result is not None

        bulk_get_time = time.time() - start_time

        # Should handle 1000 GET operations quickly
        assert bulk_get_time < 0.5  # <500ms for 1000 operations

    def test_cache_memory_efficiency(self) -> None:
        """Test cache memory usage efficiency."""
        cache_manager = CacheManager(max_size=5000)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Add items to cache
        for i in range(5000):
            key = CacheKey(f"memory_test_{i}")
            value = {
                "index": i,
                "data": "x" * 100,  # 100 character string
                "metadata": {"created": datetime.now(UTC).isoformat()},
            }
            cache_manager.put(key, value)

        # Get memory usage after caching
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # Should use reasonable memory (allowing for test overhead)
        assert memory_increase < 50  # <50MB for 5000 items

    def test_concurrent_cache_operations(self) -> None:
        """Test cache performance under concurrent access."""
        cache_manager = CacheManager(max_size=10000)

        def cache_worker(worker_id: int, operation_count: int) -> None:
            """Worker function for concurrent testing."""
            for i in range(operation_count):
                key = CacheKey(f"concurrent_{worker_id}_{i}")
                value = {"worker": worker_id, "index": i}

                # Mix of PUT and GET operations
                cache_manager.put(key, value)
                result = cache_manager.get(key)
                assert result == value

        # Test with multiple concurrent workers
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for worker_id in range(10):
                future = executor.submit(cache_worker, worker_id, 100)
                futures.append(future)

            # Wait for all workers to complete
            for future in futures:
                future.result()

        concurrent_time = time.time() - start_time

        # Should handle 1000 operations (10 workers * 100 ops) concurrently
        assert concurrent_time < 2.0  # <2 seconds for concurrent operations


class TestCostOptimizationPerformance:
    """Performance tests for cost optimization system."""

    def test_usage_recording_performance(self) -> None:
        """Test cost tracking performance requirements."""
        cost_optimizer = CostOptimizer()

        # Test single usage recording performance
        start_time = time.time()

        cost_optimizer.record_usage(
            operation=AIOperation.ANALYZE,
            model_used="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            processing_time=1.0,
        )

        record_time = time.time() - start_time

        # Should record usage quickly
        assert record_time < 0.005  # <5ms requirement

    def test_bulk_usage_recording_performance(self) -> None:
        """Test bulk usage recording performance."""
        cost_optimizer = CostOptimizer()

        # Test bulk usage recording
        start_time = time.time()

        for i in range(1000):
            cost_optimizer.record_usage(
                operation=AIOperation.GENERATE,
                model_used="gpt-3.5-turbo",
                input_tokens=100 + i,
                output_tokens=50 + i,
                cost=0.001 * (i + 1),
                processing_time=1.0 + (i * 0.001),
            )

        bulk_record_time = time.time() - start_time

        # Should handle 1000 usage records efficiently
        assert bulk_record_time < 5.0  # <5 seconds for 1000 records

    def test_cost_calculation_performance(self) -> None:
        """Test cost calculation and reporting performance."""
        cost_optimizer = CostOptimizer()

        # Add some usage data
        for _i in range(100):
            cost_optimizer.record_usage(
                operation=AIOperation.ANALYZE,
                model_used="gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                processing_time=1.0,
            )

        # Test cost breakdown performance
        start_time = time.time()
        breakdown = cost_optimizer.get_cost_breakdown()
        breakdown_time = time.time() - start_time

        assert breakdown["total_requests"] == 100
        assert breakdown_time < 0.1  # <100ms for cost breakdown

        # Test cost prediction performance
        start_time = time.time()
        prediction = cost_optimizer.predict_monthly_cost(days_to_analyze=7)
        prediction_time = time.time() - start_time

        assert float(prediction) > 0
        assert prediction_time < 0.1  # <100ms for cost prediction


class TestProviderClientPerformance:
    """Performance tests for provider clients."""

    def test_openai_client_initialization_performance(self) -> None:
        """Test OpenAI client initialization performance."""
        start_time = time.time()

        OpenAIClient(
            api_key="test-key-performance",
            model="gpt-3.5-turbo",
            timeout=10.0,
        )

        init_time = time.time() - start_time

        # Should initialize quickly
        assert init_time < 0.1  # <100ms for initialization

    def test_token_counting_performance(self) -> None:
        """Test token counting performance."""
        client = OpenAIClient(api_key="test-key-performance", model="gpt-3.5-turbo")

        # Test various text lengths
        test_texts = [
            "Short text",
            "Medium length text that contains several words and should be tokenized efficiently",
            "Long text " * 100,  # Very long text
            "Mixed content with numbers 123, symbols @#$%, and unicode: éñ中文",
        ]

        for text in test_texts:
            start_time = time.time()
            token_count = client._count_tokens(text)
            count_time = time.time() - start_time

            assert token_count > 0
            assert count_time < 0.05  # <50ms for token counting

    def test_request_payload_building_performance(self) -> None:
        """Test request payload building performance."""
        client = OpenAIClient(api_key="test-key-performance", model="gpt-3.5-turbo")

        request_result = create_ai_request(
            operation=AIOperation.ANALYZE,
            input_data="Test input for performance analysis",
            temperature=0.7,
            max_tokens=TokenCount(100),
        )
        assert request_result.is_right()
        request = request_result.value

        start_time = time.time()
        payload = client._build_request_payload(request)
        build_time = time.time() - start_time

        assert "messages" in payload
        assert "temperature" in payload
        assert build_time < 0.01  # <10ms for payload building


class TestSecurityPerformance:
    """Performance tests for security components."""

    def test_api_key_validation_performance(self) -> None:
        """Test API key validation performance."""
        api_key_manager = APIKeyManager()

        # Test various key formats
        test_keys = [
            ("openai", "sk-test1234567890abcdef"),
            ("anthropic", "sk-ant-test1234567890"),
            ("google_ai", "AIzaSyTest1234567890"),
            ("invalid", "invalid-key-format"),
        ]

        for provider, key in test_keys:
            start_time = time.time()
            api_key_manager.validate_key(provider, key)
            validation_time = time.time() - start_time

            # Should validate quickly regardless of result
            assert validation_time < 0.01  # <10ms for validation

    def test_key_storage_performance(self) -> None:
        """Test key storage and retrieval performance."""
        api_key_manager = APIKeyManager()

        # Test key storage performance
        start_time = time.time()

        store_result = api_key_manager.store_key(
            provider="performance_test",
            key_value="test-key-for-performance",
            tags={"environment": "test", "purpose": "performance"},
        )

        store_time = time.time() - start_time

        assert store_result.is_right()
        assert store_time < 0.1  # <100ms for key storage

        # Test key retrieval performance
        start_time = time.time()

        retrieve_result = api_key_manager.retrieve_key("performance_test")

        retrieve_time = time.time() - start_time

        assert retrieve_result.is_right()
        assert retrieve_time < 0.05  # <50ms for key retrieval


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow_performance(self) -> None:
        """Test complete AI workflow performance."""
        # Setup components
        cache_manager = IntelligentCacheManager()
        cost_optimizer = CostOptimizer()

        # Create test request
        request_result = create_ai_request(
            operation=AIOperation.ANALYZE,
            input_data="Test input for end-to-end performance testing",
            temperature=0.7,
            max_tokens=TokenCount(100),
        )
        assert request_result.is_right()
        request = request_result.value

        # Extract processing parameters for cache methods
        processing_params = {
            "temperature": request.temperature,
            "max_tokens": int(request.max_tokens) if request.max_tokens else None,
            "processing_mode": request.processing_mode.value,
        }

        # Test cache miss scenario (simulated)
        start_time = time.time()

        # Generate cache key
        cache_manager._generate_ai_cache_key(
            request.operation,
            request.input_data,
            processing_params,
        )

        # Check cache (miss)
        await cache_manager.get_ai_result(
            request.operation,
            request.input_data,
            processing_params,
        )

        # Simulate AI processing result
        mock_result = {
            "analysis": "Test analysis result",
            "confidence": 0.95,
            "topics": ["performance", "testing"],
        }

        # Store in cache
        await cache_manager.put_ai_result(
            request.operation,
            request.input_data,
            mock_result,
            processing_params,
        )

        # Record usage
        cost_optimizer.record_usage(
            operation=request.operation,
            model_used="gpt-3.5-turbo",
            input_tokens=25,
            output_tokens=50,
            cost=0.0001,
            processing_time=1.0,
        )

        workflow_time = time.time() - start_time

        # Complete workflow should be fast
        assert workflow_time < 0.5  # <500ms for complete workflow

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self) -> None:
        """Test cached request performance."""
        cache_manager = IntelligentCacheManager()

        # Pre-populate cache
        test_result = {"cached_analysis": "Fast cached result", "confidence": 0.99}

        await cache_manager.put_ai_result(
            AIOperation.ANALYZE,
            "cached test input",
            test_result,
            {"temperature": 0.7},
        )

        # Test cache hit performance
        start_time = time.time()

        cached_result = await cache_manager.get_ai_result(
            AIOperation.ANALYZE,
            "cached test input",
            {"temperature": 0.7},
        )

        cache_hit_time = time.time() - start_time

        assert cached_result == test_result
        assert cache_hit_time < 0.1  # <100ms for cache hit

    def test_system_resource_usage(self) -> None:
        """Test overall system resource usage."""
        # Get initial resource usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        process.cpu_percent()

        # Perform intensive operations
        cache_manager = IntelligentCacheManager()
        cost_optimizer = CostOptimizer()

        # Cache operations
        for i in range(1000):
            key = CacheKey(f"resource_test_{i}")
            value = {"index": i, "data": f"test_{i}"}
            cache_manager.cache.l1_cache.put(key, value)

        # Cost operations
        for _i in range(500):
            cost_optimizer.record_usage(
                operation=AIOperation.ANALYZE,
                model_used="gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                processing_time=1.0,
            )

        # Check final resource usage
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # Should use reasonable resources
        assert memory_increase < 100  # <100MB increase


class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""

    def test_cache_scalability(self) -> None:
        """Test cache performance at scale."""
        cache_manager = CacheManager(max_size=50000)

        # Test performance with increasing load
        operation_counts = [100, 1000, 10000]

        for count in operation_counts:
            start_time = time.time()

            # Perform operations
            for i in range(count):
                key = CacheKey(f"scale_test_{count}_{i}")
                value = {"count": count, "index": i}
                cache_manager.put(key, value)

                if i % 2 == 0:  # Get every other item
                    result = cache_manager.get(key)
                    assert result == value

            operation_time = time.time() - start_time

            # Performance should scale reasonably
            # Allow more time for larger operations but should be sub-linear
            max_time = count * 0.001  # 1ms per operation max
            assert operation_time < max_time

    def test_cost_tracking_scalability(self) -> None:
        """Test cost tracking performance at scale."""
        cost_optimizer = CostOptimizer()

        # Test with increasing numbers of usage records
        record_counts = [100, 1000, 5000]

        for count in record_counts:
            start_time = time.time()

            # Record usage
            for i in range(count):
                cost_optimizer.record_usage(
                    operation=AIOperation.GENERATE,
                    model_used="gpt-3.5-turbo",
                    input_tokens=100 + i,
                    output_tokens=50 + i,
                    cost=0.001 * (i + 1),
                    processing_time=1.0,
                )

            record_time = time.time() - start_time

            # Should scale well
            max_time = count * 0.005  # 5ms per record max
            assert record_time < max_time

            # Test reporting performance doesn't degrade
            start_time = time.time()
            breakdown = cost_optimizer.get_cost_breakdown()
            report_time = time.time() - start_time

            assert breakdown["total_requests"] == count
            assert report_time < 1.0  # <1 second for reporting regardless of scale


if __name__ == "__main__":
    # Run performance tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
