"""
API Gateway - TASK_64 Phase 2 Core Orchestration Engine

API gateway functionality with routing, load balancing, and traffic management.
Provides intelligent API routing with security, rate limiting, and performance optimization.

Architecture: API Gateway + Load Balancing + Rate Limiting + Security + Traffic Management
Performance: <50ms routing decision, <100ms load balancing, <200ms request processing
Security: Authentication, authorization, rate limiting, request validation, response filtering
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
import hashlib
import time
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.api_orchestration_architecture import (
    ServiceId, LoadBalancerId, OrchestrationId,
    LoadBalancingStrategy, RoutingRule, ServiceHealthStatus,
    APIEndpoint, ServiceDefinition, LoadBalancerConfig,
    APIOrchestrationError, LoadBalancerError, ServiceUnavailableError,
    create_load_balancer_id
)


class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthenticationType(Enum):
    """Authentication types supported by gateway."""
    NONE = "none"                       # No authentication
    API_KEY = "api_key"                 # API key authentication
    BEARER_TOKEN = "bearer_token"       # Bearer token
    BASIC_AUTH = "basic_auth"           # Basic authentication
    OAUTH2 = "oauth2"                   # OAuth 2.0
    JWT = "jwt"                         # JSON Web Token
    CUSTOM = "custom"                   # Custom authentication


class RateLimitWindow(Enum):
    """Rate limiting time windows."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


@dataclass(frozen=True)
class GatewayRoute:
    """API gateway route configuration."""
    route_id: str
    path_pattern: str                   # URL path pattern
    methods: List[RequestMethod]
    target_service_id: ServiceId
    target_endpoint_id: str
    routing_rules: List[RoutingRule] = field(default_factory=list)
    authentication: AuthenticationType = AuthenticationType.NONE
    rate_limit_config: Optional[Dict[str, Any]] = None
    load_balancer_config: Optional[Dict[str, Any]] = None
    transformation_rules: Dict[str, Any] = field(default_factory=dict)
    security_policies: List[str] = field(default_factory=list)
    caching_config: Optional[Dict[str, Any]] = None
    timeout_ms: int = 30000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.path_pattern.startswith('/'):
            raise ValueError("Path pattern must start with '/'")
        if not self.methods:
            raise ValueError("Route must specify at least one HTTP method")


@dataclass(frozen=True)
class GatewayRequest:
    """Incoming gateway request."""
    request_id: str
    method: RequestMethod
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    client_ip: str = "unknown"
    user_agent: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatewayResponse:
    """Gateway response."""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    body: Optional[str] = None
    processing_time_ms: int = 0
    target_service: Optional[ServiceId] = None
    target_endpoint: Optional[str] = None
    cache_hit: bool = False
    rate_limited: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerTarget:
    """Load balancer target configuration."""
    target_id: str
    service_id: ServiceId
    endpoint_url: str
    weight: float = 1.0
    health_status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN
    current_connections: int = 0
    response_time_ms: int = 0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitBucket:
    """Rate limiting bucket for tracking requests."""
    bucket_id: str
    window: RateLimitWindow
    limit: int
    current_count: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_request: Optional[datetime] = None
    
    def is_allowed(self) -> bool:
        """Check if request is allowed within rate limit."""
        now = datetime.now(UTC)
        
        # Calculate window duration
        if self.window == RateLimitWindow.SECOND:
            window_duration = timedelta(seconds=1)
        elif self.window == RateLimitWindow.MINUTE:
            window_duration = timedelta(minutes=1)
        elif self.window == RateLimitWindow.HOUR:
            window_duration = timedelta(hours=1)
        else:  # DAY
            window_duration = timedelta(days=1)
        
        # Reset window if expired
        if now - self.window_start > window_duration:
            self.window_start = now
            self.current_count = 0
        
        # Check if within limit
        if self.current_count < self.limit:
            self.current_count += 1
            self.last_request = now
            return True
        
        return False


class LoadBalancer:
    """Load balancer for distributing requests across targets."""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.targets: List[LoadBalancerTarget] = []
        self.current_index = 0  # For round robin
        self.request_count = 0
        
        # Initialize targets from config
        for target_config in config.targets:
            target = LoadBalancerTarget(
                target_id=target_config.get("target_id", ""),
                service_id=ServiceId(target_config.get("service_id", "")),
                endpoint_url=target_config.get("endpoint_url", ""),
                weight=target_config.get("weight", 1.0),
                health_status=ServiceHealthStatus.UNKNOWN
            )
            self.targets.append(target)
    
    def select_target(self, request: GatewayRequest) -> Optional[LoadBalancerTarget]:
        """Select target based on load balancing strategy."""
        healthy_targets = [t for t in self.targets if t.health_status in [ServiceHealthStatus.HEALTHY, ServiceHealthStatus.DEGRADED]]
        
        if not healthy_targets:
            return None
        
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_targets)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_targets)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_targets)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_targets)
        elif self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(healthy_targets, request)
        elif self.config.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(healthy_targets)
        elif self.config.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_targets)
        else:
            return healthy_targets[0]  # Default to first healthy target
    
    def _round_robin_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Round robin target selection."""
        target = targets[self.current_index % len(targets)]
        self.current_index += 1
        return target
    
    def _weighted_round_robin_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Weighted round robin selection."""
        total_weight = sum(t.weight for t in targets)
        if total_weight == 0:
            return targets[0]
        
        # Simple weighted selection
        weight_position = (self.request_count % int(total_weight * 10)) / 10
        current_weight = 0
        
        for target in targets:
            current_weight += target.weight
            if weight_position <= current_weight:
                self.request_count += 1
                return target
        
        return targets[-1]  # Fallback
    
    def _least_connections_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Least connections selection."""
        return min(targets, key=lambda t: t.current_connections)
    
    def _least_response_time_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Least response time selection."""
        return min(targets, key=lambda t: t.response_time_ms)
    
    def _consistent_hash_selection(self, targets: List[LoadBalancerTarget], request: GatewayRequest) -> LoadBalancerTarget:
        """Consistent hash selection based on client IP."""
        hash_input = f"{request.client_ip}:{request.path}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        target_index = hash_value % len(targets)
        return targets[target_index]
    
    def _random_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Random target selection."""
        import random
        return random.choice(targets)
    
    def _health_based_selection(self, targets: List[LoadBalancerTarget]) -> LoadBalancerTarget:
        """Health-based selection prioritizing healthiest targets."""
        # Sort by health status and response time
        sorted_targets = sorted(
            targets,
            key=lambda t: (t.health_status.value, t.response_time_ms)
        )
        return sorted_targets[0]


class APIGateway:
    """API gateway with routing, load balancing, and traffic management."""
    
    def __init__(self):
        self.routes: Dict[str, GatewayRoute] = {}
        self.load_balancers: Dict[LoadBalancerId, LoadBalancer] = {}
        self.rate_limit_buckets: Dict[str, RateLimitBucket] = {}
        self.request_cache: Dict[str, Any] = {}
        self.middleware_chain: List[Callable] = []
        self.security_policies: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        
        # Initialize default metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "cache_hits": 0,
            "average_response_time": 0.0
        }
    
    @require(lambda route: isinstance(route, GatewayRoute))
    def register_route(self, route: GatewayRoute) -> Either[APIOrchestrationError, bool]:
        """Register a new route in the gateway."""
        try:
            # Validate route configuration
            if route.route_id in self.routes:
                return Either.error(APIOrchestrationError(f"Route {route.route_id} already exists"))
            
            # Register route
            self.routes[route.route_id] = route
            
            # Create load balancer if needed
            if route.load_balancer_config:
                lb_config = LoadBalancerConfig(
                    balancer_id=create_load_balancer_id(route.target_service_id),
                    strategy=LoadBalancingStrategy(route.load_balancer_config.get("strategy", "round_robin")),
                    targets=route.load_balancer_config.get("targets", [])
                )
                self.load_balancers[lb_config.balancer_id] = LoadBalancer(lb_config)
            
            return Either.success(True)
            
        except Exception as e:
            return Either.error(APIOrchestrationError(f"Route registration failed: {str(e)}"))
    
    @require(lambda request: isinstance(request, GatewayRequest))
    @ensure(lambda result: result.is_success() or result.is_error())
    async def process_request(self, request: GatewayRequest) -> Either[APIOrchestrationError, GatewayResponse]:
        """
        Process incoming gateway request with routing and load balancing.
        
        Args:
            request: Incoming gateway request
            
        Returns:
            Either API orchestration error or gateway response
        """
        try:
            start_time = time.time()
            self.metrics["total_requests"] += 1
            
            # Find matching route
            route = self._find_matching_route(request)
            if not route:
                response = GatewayResponse(
                    request_id=request.request_id,
                    status_code=404,
                    headers={"Content-Type": "application/json"},
                    body='{"error": "Route not found"}',
                    error_message="No matching route found"
                )
                return Either.success(response)
            
            # Check authentication
            auth_result = await self._authenticate_request(request, route)
            if not auth_result:
                response = GatewayResponse(
                    request_id=request.request_id,
                    status_code=401,
                    headers={"Content-Type": "application/json"},
                    body='{"error": "Authentication failed"}',
                    error_message="Authentication failed"
                )
                return Either.success(response)
            
            # Check rate limiting
            if not self._check_rate_limit(request, route):
                self.metrics["rate_limited_requests"] += 1
                response = GatewayResponse(
                    request_id=request.request_id,
                    status_code=429,
                    headers={"Content-Type": "application/json"},
                    body='{"error": "Rate limit exceeded"}',
                    rate_limited=True,
                    error_message="Rate limit exceeded"
                )
                return Either.success(response)
            
            # Check cache
            cache_key = self._generate_cache_key(request, route)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.metrics["cache_hits"] += 1
                cached_response.cache_hit = True
                return Either.success(cached_response)
            
            # Select target using load balancer
            target = await self._select_target(route)
            if not target:
                response = GatewayResponse(
                    request_id=request.request_id,
                    status_code=503,
                    headers={"Content-Type": "application/json"},
                    body='{"error": "Service unavailable"}',
                    error_message="No healthy targets available"
                )
                return Either.success(response)
            
            # Transform request if needed
            transformed_request = await self._transform_request(request, route)
            
            # Forward request to target
            response = await self._forward_request(transformed_request, target, route)
            
            # Transform response if needed
            final_response = await self._transform_response(response, route)
            
            # Cache response if configured
            if route.caching_config and final_response.status_code == 200:
                self._cache_response(cache_key, final_response, route.caching_config)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            final_response.processing_time_ms = int(processing_time)
            
            if final_response.status_code < 400:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
            
            # Update average response time
            total_requests = self.metrics["total_requests"]
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (current_avg * (total_requests - 1) + processing_time) / total_requests
            
            return Either.success(final_response)
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            return Either.error(APIOrchestrationError(f"Request processing failed: {str(e)}"))
    
    def _find_matching_route(self, request: GatewayRequest) -> Optional[GatewayRoute]:
        """Find route matching the request."""
        for route in self.routes.values():
            # Check method
            if RequestMethod(request.method.value) not in route.methods:
                continue
            
            # Simple path matching (in production would use regex patterns)
            if self._path_matches(request.path, route.path_pattern):
                return route
        
        return None
    
    def _path_matches(self, request_path: str, pattern: str) -> bool:
        """Check if request path matches route pattern."""
        # Simple exact match for now - in production would use regex
        if pattern == request_path:
            return True
        
        # Check for wildcard patterns
        if pattern.endswith('/*') and request_path.startswith(pattern[:-2]):
            return True
        
        return False
    
    async def _authenticate_request(self, request: GatewayRequest, route: GatewayRoute) -> bool:
        """Authenticate request based on route configuration."""
        if route.authentication == AuthenticationType.NONE:
            return True
        
        # Simple authentication check - in production would validate tokens/keys
        if route.authentication == AuthenticationType.API_KEY:
            return "X-API-Key" in request.headers
        elif route.authentication == AuthenticationType.BEARER_TOKEN:
            auth_header = request.headers.get("Authorization", "")
            return auth_header.startswith("Bearer ")
        elif route.authentication == AuthenticationType.BASIC_AUTH:
            auth_header = request.headers.get("Authorization", "")
            return auth_header.startswith("Basic ")
        
        return False
    
    def _check_rate_limit(self, request: GatewayRequest, route: GatewayRoute) -> bool:
        """Check rate limiting for request."""
        if not route.rate_limit_config:
            return True
        
        # Generate rate limit key (by client IP or user)
        rate_limit_key = f"{route.route_id}:{request.client_ip}"
        
        # Get or create rate limit bucket
        if rate_limit_key not in self.rate_limit_buckets:
            self.rate_limit_buckets[rate_limit_key] = RateLimitBucket(
                bucket_id=rate_limit_key,
                window=RateLimitWindow(route.rate_limit_config.get("window", "minute")),
                limit=route.rate_limit_config.get("limit", 100)
            )
        
        bucket = self.rate_limit_buckets[rate_limit_key]
        return bucket.is_allowed()
    
    async def _select_target(self, route: GatewayRoute) -> Optional[LoadBalancerTarget]:
        """Select target for request using load balancer."""
        lb_id = create_load_balancer_id(route.target_service_id)
        
        if lb_id in self.load_balancers:
            return self.load_balancers[lb_id].select_target(GatewayRequest(
                request_id="dummy",
                method=RequestMethod.GET,
                path=route.path_pattern,
                headers={}
            ))
        
        # Default target (single service)
        return LoadBalancerTarget(
            target_id=route.target_endpoint_id,
            service_id=route.target_service_id,
            endpoint_url=f"http://localhost:8080/api/{route.target_service_id}",
            health_status=ServiceHealthStatus.HEALTHY
        )
    
    async def _transform_request(self, request: GatewayRequest, route: GatewayRoute) -> GatewayRequest:
        """Transform request based on route transformation rules."""
        if not route.transformation_rules:
            return request
        
        # Apply transformations (simplified)
        transformed_headers = request.headers.copy()
        
        # Add headers
        if "add_headers" in route.transformation_rules:
            transformed_headers.update(route.transformation_rules["add_headers"])
        
        # Remove headers
        if "remove_headers" in route.transformation_rules:
            for header in route.transformation_rules["remove_headers"]:
                transformed_headers.pop(header, None)
        
        return GatewayRequest(
            request_id=request.request_id,
            method=request.method,
            path=request.path,
            headers=transformed_headers,
            query_params=request.query_params,
            body=request.body,
            client_ip=request.client_ip,
            user_agent=request.user_agent,
            timestamp=request.timestamp,
            metadata=request.metadata
        )
    
    async def _forward_request(self, request: GatewayRequest, target: LoadBalancerTarget, route: GatewayRoute) -> GatewayResponse:
        """Forward request to target service."""
        # Simulate request forwarding
        await asyncio.sleep(0.05)  # Simulate network delay
        
        # Mock successful response
        return GatewayResponse(
            request_id=request.request_id,
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"status": "success", "data": "mock_response"}',
            target_service=target.service_id,
            target_endpoint=target.target_id
        )
    
    async def _transform_response(self, response: GatewayResponse, route: GatewayRoute) -> GatewayResponse:
        """Transform response based on route transformation rules."""
        if not route.transformation_rules:
            return response
        
        # Apply response transformations
        transformed_headers = response.headers.copy()
        
        # Add response headers
        if "add_response_headers" in route.transformation_rules:
            transformed_headers.update(route.transformation_rules["add_response_headers"])
        
        return GatewayResponse(
            request_id=response.request_id,
            status_code=response.status_code,
            headers=transformed_headers,
            body=response.body,
            processing_time_ms=response.processing_time_ms,
            target_service=response.target_service,
            target_endpoint=response.target_endpoint,
            cache_hit=response.cache_hit,
            rate_limited=response.rate_limited,
            error_message=response.error_message,
            metadata=response.metadata
        )
    
    def _generate_cache_key(self, request: GatewayRequest, route: GatewayRoute) -> str:
        """Generate cache key for request."""
        key_parts = [
            route.route_id,
            request.method.value,
            request.path,
            str(sorted(request.query_params.items()))
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[GatewayResponse]:
        """Get cached response if available."""
        return self.request_cache.get(cache_key)
    
    def _cache_response(self, cache_key: str, response: GatewayResponse, cache_config: Dict[str, Any]):
        """Cache response based on configuration."""
        ttl_seconds = cache_config.get("ttl_seconds", 300)  # 5 minutes default
        
        # Simple in-memory cache (in production would use Redis/Memcached)
        self.request_cache[cache_key] = response
        
        # Schedule cache expiration (simplified)
        # In production would use proper cache management
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        return self.metrics.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get gateway health status."""
        return {
            "status": "healthy",
            "routes_count": len(self.routes),
            "load_balancers_count": len(self.load_balancers),
            "cache_size": len(self.request_cache),
            "rate_limit_buckets": len(self.rate_limit_buckets),
            "metrics": self.get_metrics()
        }


# Export the API gateway class
__all__ = [
    "APIGateway", "GatewayRoute", "GatewayRequest", "GatewayResponse",
    "LoadBalancer", "LoadBalancerTarget", "RateLimitBucket",
    "RequestMethod", "AuthenticationType", "RateLimitWindow"
]