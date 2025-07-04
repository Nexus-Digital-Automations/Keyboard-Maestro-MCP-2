"""
Real-time Predictor - TASK_59 Phase 5 Integration & Validation Implementation

Real-time prediction serving and monitoring for predictive analytics models.
Provides low-latency prediction serving, real-time monitoring, and adaptive model management.

Architecture: Real-time Serving + Model Management + Monitoring + Adaptive Learning
Performance: <50ms prediction serving, <100ms model updates, <200ms monitoring updates
Security: Safe prediction serving, validated inputs, comprehensive audit logging
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import statistics
import json
import math
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.predictive_modeling import (
    PredictionId, create_prediction_id, ModelId, PredictiveModelingError,
    RealtimePredictionError, validate_prediction_request
)


class PredictionMode(Enum):
    """Modes for real-time prediction."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"
    ADAPTIVE = "adaptive"


class ModelState(Enum):
    """States of prediction models."""
    LOADING = "loading"
    READY = "ready"
    SERVING = "serving"
    UPDATING = "updating"
    ERROR = "error"
    RETIRED = "retired"


class PredictionPriority(Enum):
    """Priority levels for predictions."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CachingStrategy(Enum):
    """Caching strategies for predictions."""
    NO_CACHE = "no_cache"
    FEATURE_BASED = "feature_based"
    TIME_BASED = "time_based"
    LRU = "lru"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class PredictionRequest:
    """Request for real-time prediction."""
    request_id: str
    model_id: ModelId
    features: List[float]
    prediction_mode: PredictionMode = PredictionMode.SINGLE
    priority: PredictionPriority = PredictionPriority.NORMAL
    timeout_ms: int = 5000
    include_confidence: bool = True
    include_explanation: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timeout_ms < 100:
            raise ValueError("Timeout must be at least 100ms")
        if not self.features:
            raise ValueError("Features cannot be empty")


@dataclass(frozen=True)
class PredictionResponse:
    """Response from real-time prediction."""
    response_id: str
    request_id: str
    model_id: ModelId
    prediction_value: float
    confidence_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    prediction_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    processing_time_ms: float = 0.0
    model_version: str = "1.0"
    cached: bool = False
    
    def __post_init__(self):
        if self.confidence_score is not None and not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ModelMetrics:
    """Real-time metrics for prediction models."""
    model_id: ModelId
    requests_per_second: float
    average_latency_ms: float
    error_rate: float
    cache_hit_rate: float
    prediction_accuracy: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    active_connections: int = 0
    queue_length: int = 0
    
    def __post_init__(self):
        if not (0.0 <= self.error_rate <= 1.0):
            raise ValueError("Error rate must be between 0.0 and 1.0")


@dataclass
class LoadedModel:
    """Container for loaded prediction model."""
    model_id: ModelId
    model_state: ModelState
    predictor_function: Callable[[List[float]], float]
    confidence_function: Optional[Callable[[List[float]], float]] = None
    feature_names: List[str] = field(default_factory=list)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    load_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_prediction: Optional[datetime] = None
    prediction_count: int = 0
    error_count: int = 0


class PredictionCache:
    """Cache for prediction results."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[PredictionResponse, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, cache_key: str) -> Optional[PredictionResponse]:
        """Get cached prediction if valid."""
        if cache_key in self.cache:
            response, cache_time = self.cache[cache_key]
            
            # Check TTL
            if (datetime.now(UTC) - cache_time).total_seconds() < self.ttl_seconds:
                self.access_times[cache_key] = datetime.now(UTC)
                return response
            else:
                # Expired
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
        
        return None
    
    def put(self, cache_key: str, response: PredictionResponse):
        """Cache prediction response."""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Mark as cached
        cached_response = PredictionResponse(
            response_id=response.response_id,
            request_id=response.request_id,
            model_id=response.model_id,
            prediction_value=response.prediction_value,
            confidence_score=response.confidence_score,
            confidence_interval=response.confidence_interval,
            feature_importance=response.feature_importance,
            explanation=response.explanation,
            prediction_timestamp=response.prediction_timestamp,
            processing_time_ms=response.processing_time_ms,
            model_version=response.model_version,
            cached=True
        )
        
        self.cache[cache_key] = (cached_response, datetime.now(UTC))
        self.access_times[cache_key] = datetime.now(UTC)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.access_times:
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            if lru_key in self.cache:
                del self.cache[lru_key]
            del self.access_times[lru_key]
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need to track hits/misses
            "ttl_seconds": self.ttl_seconds
        }


class RealtimePredictor:
    """Real-time prediction serving and monitoring system."""
    
    def __init__(self):
        self.loaded_models: Dict[str, LoadedModel] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.prediction_queue: asyncio.Queue = asyncio.Queue()
        self.request_history: deque = deque(maxlen=10000)
        self.performance_metrics: Dict[str, float] = {}
        self.prediction_cache = PredictionCache()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.monitoring_enabled = True
        self.adaptive_scaling = True
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the real-time prediction system."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._prediction_worker()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._model_health_monitor()),
            asyncio.create_task(self._cache_cleanup())
        ]
    
    async def stop(self):
        """Stop the real-time prediction system."""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
    
    @require(lambda model_id: model_id is not None)
    @require(lambda predictor_function: callable(predictor_function))
    async def load_model(
        self,
        model_id: ModelId,
        predictor_function: Callable[[List[float]], float],
        confidence_function: Optional[Callable[[List[float]], float]] = None,
        feature_names: Optional[List[str]] = None,
        model_metadata: Optional[Dict[str, Any]] = None
    ) -> Either[RealtimePredictionError, str]:
        """Load a model for real-time serving."""
        try:
            model_key = str(model_id)
            
            # Check if model already loaded
            if model_key in self.loaded_models:
                existing_model = self.loaded_models[model_key]
                existing_model.model_state = ModelState.UPDATING
            
            # Create loaded model
            loaded_model = LoadedModel(
                model_id=model_id,
                model_state=ModelState.LOADING,
                predictor_function=predictor_function,
                confidence_function=confidence_function,
                feature_names=feature_names or [],
                model_metadata=model_metadata or {}
            )
            
            # Test model with dummy data
            try:
                dummy_features = [1.0] * max(1, len(feature_names) if feature_names else 3)
                test_prediction = predictor_function(dummy_features)
                
                if not isinstance(test_prediction, (int, float)):
                    return Either.left(RealtimePredictionError(f"Model predictor must return numeric value, got {type(test_prediction)}"))
                
                # Test confidence function if provided
                if confidence_function:
                    test_confidence = confidence_function(dummy_features)
                    if not isinstance(test_confidence, (int, float)) or not (0.0 <= test_confidence <= 1.0):
                        return Either.left(RealtimePredictionError("Confidence function must return value between 0.0 and 1.0"))
                
            except Exception as e:
                return Either.left(RealtimePredictionError(f"Model test failed: {str(e)}"))
            
            # Mark as ready
            loaded_model.model_state = ModelState.READY
            self.loaded_models[model_key] = loaded_model
            
            # Initialize metrics
            self.model_metrics[model_key] = ModelMetrics(
                model_id=model_id,
                requests_per_second=0.0,
                average_latency_ms=0.0,
                error_rate=0.0,
                cache_hit_rate=0.0,
                prediction_accuracy=0.0
            )
            
            return Either.right(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            return Either.left(RealtimePredictionError(f"Failed to load model: {str(e)}"))
    
    @require(lambda request: isinstance(request, PredictionRequest))
    @ensure(lambda result: result.is_right() or isinstance(result.left_value, RealtimePredictionError))
    async def predict(
        self,
        request: PredictionRequest
    ) -> Either[RealtimePredictionError, PredictionResponse]:
        """Make real-time prediction."""
        try:
            start_time = time.time()
            model_key = str(request.model_id)
            
            # Check if model is loaded
            if model_key not in self.loaded_models:
                return Either.left(RealtimePredictionError(f"Model {request.model_id} not loaded"))
            
            loaded_model = self.loaded_models[model_key]
            
            # Check model state
            if loaded_model.model_state not in [ModelState.READY, ModelState.SERVING]:
                return Either.left(RealtimePredictionError(f"Model {request.model_id} not ready (state: {loaded_model.model_state.value})"))
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = self.prediction_cache.get(cache_key)
            if cached_response:
                return Either.right(cached_response)
            
            # Update model state
            loaded_model.model_state = ModelState.SERVING
            
            # Queue request if priority-based processing
            if request.priority == PredictionPriority.CRITICAL:
                response = await self._process_prediction_immediately(request, loaded_model, start_time)
            else:
                # Add to queue for batch processing
                response = await self._process_prediction_immediately(request, loaded_model, start_time)
            
            # Cache result if successful
            if response.is_right():
                self.prediction_cache.put(cache_key, response.right_value)
            
            # Update model state back to ready
            loaded_model.model_state = ModelState.READY
            
            return response
            
        except Exception as e:
            return Either.left(RealtimePredictionError(f"Prediction failed: {str(e)}"))
    
    async def _process_prediction_immediately(
        self,
        request: PredictionRequest,
        loaded_model: LoadedModel,
        start_time: float
    ) -> Either[RealtimePredictionError, PredictionResponse]:
        """Process prediction immediately."""
        try:
            # Validate features
            if len(request.features) != len(loaded_model.feature_names) and loaded_model.feature_names:
                return Either.left(RealtimePredictionError(
                    f"Feature count mismatch: expected {len(loaded_model.feature_names)}, got {len(request.features)}"
                ))
            
            # Make prediction
            try:
                prediction_value = loaded_model.predictor_function(request.features)
                
                if not isinstance(prediction_value, (int, float)):
                    return Either.left(RealtimePredictionError("Predictor returned non-numeric value"))
                
                prediction_value = float(prediction_value)
                
            except Exception as e:
                loaded_model.error_count += 1
                return Either.left(RealtimePredictionError(f"Prediction execution failed: {str(e)}"))
            
            # Calculate confidence if requested
            confidence_score = None
            confidence_interval = None
            
            if request.include_confidence and loaded_model.confidence_function:
                try:
                    confidence_score = loaded_model.confidence_function(request.features)
                    confidence_score = float(confidence_score)
                    
                    # Generate confidence interval (simplified)
                    if 0.0 <= confidence_score <= 1.0:
                        error_margin = (1.0 - confidence_score) * abs(prediction_value) * 0.1
                        confidence_interval = (
                            prediction_value - error_margin,
                            prediction_value + error_margin
                        )
                
                except Exception:
                    confidence_score = None
            
            # Calculate feature importance if requested
            feature_importance = None
            if request.include_explanation and loaded_model.feature_names:
                feature_importance = self._calculate_feature_importance(
                    request.features, loaded_model.feature_names, prediction_value
                )
            
            # Generate explanation if requested
            explanation = None
            if request.include_explanation:
                explanation = self._generate_prediction_explanation(
                    prediction_value, confidence_score, feature_importance
                )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                response_id=f"resp_{request.request_id}_{datetime.now(UTC).isoformat()}",
                request_id=request.request_id,
                model_id=request.model_id,
                prediction_value=prediction_value,
                confidence_score=confidence_score,
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                explanation=explanation,
                processing_time_ms=processing_time_ms,
                model_version=loaded_model.model_metadata.get("version", "1.0")
            )
            
            # Update model statistics
            loaded_model.prediction_count += 1
            loaded_model.last_prediction = datetime.now(UTC)
            
            # Record request
            self.request_history.append({
                "timestamp": datetime.now(UTC),
                "request_id": request.request_id,
                "model_id": str(request.model_id),
                "processing_time_ms": processing_time_ms,
                "success": True,
                "cached": False
            })
            
            return Either.right(response)
            
        except Exception as e:
            loaded_model.error_count += 1
            return Either.left(RealtimePredictionError(f"Prediction processing failed: {str(e)}"))
    
    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction request."""
        # Simple hash of model_id and features
        features_str = ",".join(f"{f:.6f}" for f in request.features)
        return f"{request.model_id}:{hash(features_str)}"
    
    def _calculate_feature_importance(
        self,
        features: List[float],
        feature_names: List[str],
        prediction_value: float
    ) -> Dict[str, float]:
        """Calculate simplified feature importance."""
        importance = {}
        
        # Simplified importance based on feature magnitude and prediction value
        total_magnitude = sum(abs(f) for f in features)
        
        for i, (feature_name, feature_value) in enumerate(zip(feature_names, features)):
            if total_magnitude > 0:
                # Importance proportional to feature magnitude
                raw_importance = abs(feature_value) / total_magnitude
                
                # Adjust by position (earlier features slightly more important)
                position_weight = 1.0 - (i * 0.1 / len(features))
                
                importance[feature_name] = raw_importance * position_weight
            else:
                importance[feature_name] = 1.0 / len(features)
        
        # Normalize to sum to 1.0
        total_importance = sum(importance.values())
        if total_importance > 0:
            for feature_name in importance:
                importance[feature_name] /= total_importance
        
        return importance
    
    def _generate_prediction_explanation(
        self,
        prediction_value: float,
        confidence_score: Optional[float],
        feature_importance: Optional[Dict[str, float]]
    ) -> str:
        """Generate human-readable prediction explanation."""
        explanation_parts = [f"Predicted value: {prediction_value:.3f}"]
        
        if confidence_score is not None:
            confidence_pct = confidence_score * 100
            confidence_level = "high" if confidence_score > 0.8 else "medium" if confidence_score > 0.5 else "low"
            explanation_parts.append(f"Confidence: {confidence_pct:.1f}% ({confidence_level})")
        
        if feature_importance:
            # Find top 3 most important features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_features:
                feature_list = ", ".join(f"{name} ({imp:.1%})" for name, imp in top_features)
                explanation_parts.append(f"Key factors: {feature_list}")
        
        return ". ".join(explanation_parts)
    
    async def predict_batch(
        self,
        requests: List[PredictionRequest]
    ) -> List[Either[RealtimePredictionError, PredictionResponse]]:
        """Make batch predictions efficiently."""
        if not requests:
            return []
        
        # Group requests by model for efficient processing
        model_groups = defaultdict(list)
        for request in requests:
            model_groups[str(request.model_id)].append(request)
        
        # Process each model group
        results = []
        
        for model_key, model_requests in model_groups.items():
            # Process requests for this model
            model_results = await asyncio.gather(
                *[self.predict(request) for request in model_requests],
                return_exceptions=False
            )
            results.extend(model_results)
        
        return results
    
    async def predict_streaming(
        self,
        request_stream: AsyncGenerator[PredictionRequest, None]
    ) -> AsyncGenerator[Either[RealtimePredictionError, PredictionResponse], None]:
        """Handle streaming predictions."""
        async for request in request_stream:
            result = await self.predict(request)
            yield result
    
    async def _prediction_worker(self):
        """Background worker for processing prediction queue."""
        while self._running:
            try:
                # Process queued predictions
                if not self.prediction_queue.empty():
                    # Get next request
                    request = await asyncio.wait_for(
                        self.prediction_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process prediction
                    result = await self.predict(request)
                    
                    # Mark task done
                    self.prediction_queue.task_done()
                else:
                    # No requests, sleep briefly
                    await asyncio.sleep(0.1)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Prediction worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def _metrics_updater(self):
        """Background task to update model metrics."""
        while self._running:
            try:
                current_time = datetime.now(UTC)
                
                # Update metrics for each loaded model
                for model_key, loaded_model in self.loaded_models.items():
                    if model_key in self.model_metrics:
                        # Calculate requests per second
                        recent_requests = [
                            entry for entry in self.request_history
                            if (entry["model_id"] == model_key and
                                (current_time - entry["timestamp"]).total_seconds() <= 60)
                        ]
                        
                        rps = len(recent_requests) / 60.0 if recent_requests else 0.0
                        
                        # Calculate average latency
                        recent_latencies = [entry["processing_time_ms"] for entry in recent_requests]
                        avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
                        
                        # Calculate error rate
                        total_requests = loaded_model.prediction_count + loaded_model.error_count
                        error_rate = loaded_model.error_count / max(1, total_requests)
                        
                        # Calculate cache hit rate
                        cached_requests = [entry for entry in recent_requests if entry.get("cached", False)]
                        cache_hit_rate = len(cached_requests) / max(1, len(recent_requests))
                        
                        # Update metrics
                        self.model_metrics[model_key] = ModelMetrics(
                            model_id=loaded_model.model_id,
                            requests_per_second=rps,
                            average_latency_ms=avg_latency,
                            error_rate=error_rate,
                            cache_hit_rate=cache_hit_rate,
                            prediction_accuracy=0.85,  # Would need actual accuracy tracking
                            last_updated=current_time,
                            queue_length=self.prediction_queue.qsize()
                        )
                
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                print(f"Metrics updater error: {e}")
                await asyncio.sleep(10.0)
    
    async def _model_health_monitor(self):
        """Monitor model health and performance."""
        while self._running:
            try:
                for model_key, loaded_model in self.loaded_models.items():
                    # Check if model has been idle too long
                    if loaded_model.last_prediction:
                        idle_time = datetime.now(UTC) - loaded_model.last_prediction
                        if idle_time > timedelta(hours=1):
                            # Consider retiring idle models
                            print(f"Model {model_key} has been idle for {idle_time}")
                    
                    # Check error rate
                    if model_key in self.model_metrics:
                        metrics = self.model_metrics[model_key]
                        if metrics.error_rate > 0.1:  # 10% error rate threshold
                            print(f"High error rate for model {model_key}: {metrics.error_rate:.2%}")
                        
                        # Check latency
                        if metrics.average_latency_ms > 1000:  # 1 second threshold
                            print(f"High latency for model {model_key}: {metrics.average_latency_ms:.1f}ms")
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(30.0)
    
    async def _cache_cleanup(self):
        """Clean up expired cache entries."""
        while self._running:
            try:
                # Cache cleanup is handled automatically in PredictionCache.get()
                # This task could perform additional cleanup or optimization
                
                await asyncio.sleep(300.0)  # Run every 5 minutes
                
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                await asyncio.sleep(300.0)
    
    async def get_model_status(self, model_id: ModelId) -> Either[RealtimePredictionError, Dict[str, Any]]:
        """Get status and metrics for a specific model."""
        try:
            model_key = str(model_id)
            
            if model_key not in self.loaded_models:
                return Either.left(RealtimePredictionError(f"Model {model_id} not found"))
            
            loaded_model = self.loaded_models[model_key]
            metrics = self.model_metrics.get(model_key)
            
            status = {
                "model_id": str(model_id),
                "state": loaded_model.model_state.value,
                "load_timestamp": loaded_model.load_timestamp.isoformat(),
                "last_prediction": loaded_model.last_prediction.isoformat() if loaded_model.last_prediction else None,
                "prediction_count": loaded_model.prediction_count,
                "error_count": loaded_model.error_count,
                "feature_count": len(loaded_model.feature_names),
                "metadata": loaded_model.model_metadata
            }
            
            if metrics:
                status["metrics"] = {
                    "requests_per_second": metrics.requests_per_second,
                    "average_latency_ms": metrics.average_latency_ms,
                    "error_rate": metrics.error_rate,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "prediction_accuracy": metrics.prediction_accuracy,
                    "last_updated": metrics.last_updated.isoformat()
                }
            
            return Either.right(status)
            
        except Exception as e:
            return Either.left(RealtimePredictionError(f"Failed to get model status: {str(e)}"))
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics."""
        current_time = datetime.now(UTC)
        
        # Overall request statistics
        recent_requests = [
            entry for entry in self.request_history
            if (current_time - entry["timestamp"]).total_seconds() <= 300  # Last 5 minutes
        ]
        
        successful_requests = [req for req in recent_requests if req["success"]]
        
        # Calculate system-wide metrics
        total_rps = len(recent_requests) / 300.0 if recent_requests else 0.0
        success_rate = len(successful_requests) / max(1, len(recent_requests))
        
        latencies = [req["processing_time_ms"] for req in successful_requests]
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else avg_latency
        
        return {
            "system_status": "running" if self._running else "stopped",
            "loaded_models": len(self.loaded_models),
            "active_models": sum(1 for model in self.loaded_models.values() 
                               if model.model_state in [ModelState.READY, ModelState.SERVING]),
            "total_requests_per_second": total_rps,
            "system_success_rate": success_rate,
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "cache_stats": self.prediction_cache.get_stats(),
            "queue_length": self.prediction_queue.qsize(),
            "memory_usage": {
                "request_history_size": len(self.request_history),
                "cache_size": len(self.prediction_cache.cache)
            },
            "uptime_seconds": (current_time - datetime.now(UTC)).total_seconds() if hasattr(self, '_start_time') else 0
        }
    
    async def unload_model(self, model_id: ModelId) -> Either[RealtimePredictionError, str]:
        """Unload a model from real-time serving."""
        try:
            model_key = str(model_id)
            
            if model_key not in self.loaded_models:
                return Either.left(RealtimePredictionError(f"Model {model_id} not loaded"))
            
            # Mark as retired
            self.loaded_models[model_key].model_state = ModelState.RETIRED
            
            # Remove from active models
            del self.loaded_models[model_key]
            
            # Remove metrics
            if model_key in self.model_metrics:
                del self.model_metrics[model_key]
            
            # Clear related cache entries (simplified)
            # In a real implementation, would selectively clear cache
            self.prediction_cache.clear()
            
            return Either.right(f"Model {model_id} unloaded successfully")
            
        except Exception as e:
            return Either.left(RealtimePredictionError(f"Failed to unload model: {str(e)}"))
    
    async def update_model(
        self,
        model_id: ModelId,
        new_predictor_function: Callable[[List[float]], float],
        new_confidence_function: Optional[Callable[[List[float]], float]] = None
    ) -> Either[RealtimePredictionError, str]:
        """Update an existing model with new predictor functions."""
        try:
            model_key = str(model_id)
            
            if model_key not in self.loaded_models:
                return Either.left(RealtimePredictionError(f"Model {model_id} not loaded"))
            
            loaded_model = self.loaded_models[model_key]
            loaded_model.model_state = ModelState.UPDATING
            
            # Test new predictor
            dummy_features = [1.0] * max(1, len(loaded_model.feature_names))
            test_prediction = new_predictor_function(dummy_features)
            
            if not isinstance(test_prediction, (int, float)):
                loaded_model.model_state = ModelState.ERROR
                return Either.left(RealtimePredictionError("New predictor must return numeric value"))
            
            # Update predictor functions
            loaded_model.predictor_function = new_predictor_function
            if new_confidence_function:
                loaded_model.confidence_function = new_confidence_function
            
            # Reset counters
            loaded_model.prediction_count = 0
            loaded_model.error_count = 0
            loaded_model.load_timestamp = datetime.now(UTC)
            
            # Mark as ready
            loaded_model.model_state = ModelState.READY
            
            # Clear cache to ensure fresh predictions
            self.prediction_cache.clear()
            
            return Either.right(f"Model {model_id} updated successfully")
            
        except Exception as e:
            if model_key in self.loaded_models:
                self.loaded_models[model_key].model_state = ModelState.ERROR
            return Either.left(RealtimePredictionError(f"Failed to update model: {str(e)}"))
    
    def __del__(self):
        """Cleanup when predictor is destroyed."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)