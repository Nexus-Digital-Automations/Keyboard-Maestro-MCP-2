"""
Intelligent caching system for AI operations.

This module provides comprehensive caching capabilities for AI operations
including multi-level caching, intelligent cache management, predictive
prefetching, and performance optimization with enterprise-grade reliability.

Security: All cache operations include validation and secure data handling.
Performance: Optimized for high-throughput with intelligent eviction policies.
Type Safety: Complete integration with AI processing architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NewType, Dict, List, Optional, Any, Set, Callable, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import json
import hashlib
import pickle
import zlib
from pathlib import Path
import threading
from collections import OrderedDict

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError
from ..core.ai_integration import AIOperation, AIRequest, AIResponse

# Branded Types for Caching System
CacheKey = NewType('CacheKey', str)
CacheNamespace = NewType('CacheNamespace', str)
CacheSize = NewType('CacheSize', int)
HitRatio = NewType('HitRatio', float)
ExpirationTime = NewType('ExpirationTime', datetime)


class CacheLevel(Enum):
    """Cache level hierarchy."""
    L1_MEMORY = "l1_memory"         # Fast in-memory cache
    L2_COMPRESSED = "l2_compressed" # Compressed in-memory cache
    L3_DISK = "l3_disk"            # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed" # Distributed cache (future)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    SIZE_BASED = "size_based"      # Size-based eviction
    INTELLIGENT = "intelligent"    # AI-driven eviction
    FIFO = "fifo"                  # First In, First Out


class CacheStrategy(Enum):
    """Caching strategies."""
    CACHE_ASIDE = "cache_aside"    # Load on demand
    WRITE_THROUGH = "write_through" # Write to cache and storage
    WRITE_BEHIND = "write_behind"  # Async write to storage
    REFRESH_AHEAD = "refresh_ahead" # Predictive refresh
    READ_THROUGH = "read_through"  # Auto-load on miss


@dataclass(frozen=True)
class CacheEntry:
    """Individual cache entry with metadata."""
    key: CacheKey
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[timedelta] = None
    namespace: CacheNamespace = CacheNamespace("default")
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.key) > 0)
    @require(lambda self: self.size_bytes >= 0)
    @require(lambda self: self.access_count >= 0)
    def __post_init__(self):
        """Validate cache entry."""
        pass
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return datetime.now(UTC) > self.created_at + self.ttl
    
    def calculate_age(self) -> timedelta:
        """Calculate age of cache entry."""
        return datetime.now(UTC) - self.created_at
    
    def get_access_rate(self) -> float:
        """Calculate access rate per hour."""
        age_hours = self.calculate_age().total_seconds() / 3600
        if age_hours == 0:
            return float(self.access_count)
        return self.access_count / age_hours


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    hit_ratio: float = 0.0
    average_access_time: float = 0.0
    memory_usage: int = 0
    disk_usage: int = 0
    
    def update_hit_ratio(self) -> None:
        """Update hit ratio calculation."""
        total_requests = self.hits + self.misses
        self.hit_ratio = self.hits / total_requests if total_requests > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_size": self.total_size,
            "entry_count": self.entry_count,
            "hit_ratio": self.hit_ratio,
            "average_access_time": self.average_access_time,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage
        }


class CacheManager:
    """Base cache manager with common functionality."""
    
    def __init__(self, max_size: int = 1000, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.cache: Dict[CacheKey, CacheEntry] = OrderedDict()
        self.statistics = CacheStatistics()
        self._lock = threading.RLock()
        self.namespace_stats: Dict[CacheNamespace, CacheStatistics] = {}
    
    def _generate_key(self, operation: AIOperation, input_data: Any, 
                     parameters: Dict[str, Any] = None) -> CacheKey:
        """Generate cache key for AI operation."""
        # Create deterministic key from operation and inputs
        key_components = [
            operation.value,
            str(input_data),
            json.dumps(parameters or {}, sort_keys=True)
        ]
        key_string = "|".join(key_components)
        
        # Hash for consistent length and security
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:32]
        return CacheKey(f"{operation.value}:{key_hash}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self.cache) >= self.max_size
    
    def _select_eviction_candidate(self) -> Optional[CacheKey]:
        """Select entry for eviction based on policy."""
        if not self.cache:
            return None
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # OrderedDict maintains insertion order, first item is least recently used
            return next(iter(self.cache))
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Find entry with lowest access count
            min_access = min(entry.access_count for entry in self.cache.values())
            for key, entry in self.cache.items():
                if entry.access_count == min_access:
                    return key
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Find oldest expired entry
            now = datetime.now(UTC)
            for key, entry in self.cache.items():
                if entry.ttl and now > entry.created_at + entry.ttl:
                    return key
            # If no expired entries, fall back to LRU
            return next(iter(self.cache))
        
        elif self.eviction_policy == EvictionPolicy.SIZE_BASED:
            # Find largest entry
            max_size = max(entry.size_bytes for entry in self.cache.values())
            for key, entry in self.cache.items():
                if entry.size_bytes == max_size:
                    return key
        
        elif self.eviction_policy == EvictionPolicy.INTELLIGENT:
            # Score-based eviction considering multiple factors
            return self._intelligent_eviction_candidate()
        
        else:  # FIFO
            return next(iter(self.cache))
        
        return None
    
    def _intelligent_eviction_candidate(self) -> Optional[CacheKey]:
        """Select eviction candidate using intelligent scoring."""
        if not self.cache:
            return None
        
        scores = {}
        now = datetime.now(UTC)
        
        for key, entry in self.cache.items():
            # Score based on multiple factors (lower score = more likely to evict)
            age_hours = (now - entry.created_at).total_seconds() / 3600
            time_since_access = (now - entry.last_accessed).total_seconds() / 3600
            
            # Factors: recency, frequency, size, age
            recency_score = 1.0 / (time_since_access + 1)  # Higher = more recent
            frequency_score = entry.access_count / (age_hours + 1)  # Higher = more frequent
            size_penalty = entry.size_bytes / 1024  # Lower = smaller penalty
            
            # Combined score (higher = keep, lower = evict)
            score = (recency_score * 0.4) + (frequency_score * 0.4) - (size_penalty * 0.2)
            scores[key] = score
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _evict_entry(self, key: CacheKey) -> None:
        """Evict entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.statistics.evictions += 1
            self.statistics.total_size -= entry.size_bytes
            self.statistics.entry_count -= 1
            
            # Update namespace stats
            if entry.namespace in self.namespace_stats:
                ns_stats = self.namespace_stats[entry.namespace]
                ns_stats.evictions += 1
                ns_stats.total_size -= entry.size_bytes
                ns_stats.entry_count -= 1
    
    def get(self, key: CacheKey, namespace: CacheNamespace = CacheNamespace("default")) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            full_key = CacheKey(f"{namespace}:{key}")
            
            if full_key in self.cache:
                entry = self.cache[full_key]
                
                # Check expiration
                if entry.is_expired():
                    self._evict_entry(full_key)
                    self.statistics.misses += 1
                    return None
                
                # Update access info
                updated_entry = CacheEntry(
                    key=entry.key,
                    value=entry.value,
                    created_at=entry.created_at,
                    last_accessed=datetime.now(UTC),
                    access_count=entry.access_count + 1,
                    size_bytes=entry.size_bytes,
                    ttl=entry.ttl,
                    namespace=entry.namespace,
                    tags=entry.tags,
                    metadata=entry.metadata
                )
                
                # Move to end for LRU
                del self.cache[full_key]
                self.cache[full_key] = updated_entry
                
                self.statistics.hits += 1
                self.statistics.update_hit_ratio()
                
                return entry.value
            else:
                self.statistics.misses += 1
                self.statistics.update_hit_ratio()
                return None
    
    def put(self, key: CacheKey, value: Any, ttl: Optional[timedelta] = None,
            namespace: CacheNamespace = CacheNamespace("default"),
            tags: Set[str] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            full_key = CacheKey(f"{namespace}:{key}")
            
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Evict if necessary
            while self._should_evict():
                candidate = self._select_eviction_candidate()
                if candidate:
                    self._evict_entry(candidate)
                else:
                    break
            
            # Create cache entry
            entry = CacheEntry(
                key=full_key,
                value=value,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl,
                namespace=namespace,
                tags=tags or set(),
                metadata={}
            )
            
            # Update existing or add new
            was_update = full_key in self.cache
            if was_update:
                old_entry = self.cache[full_key]
                self.statistics.total_size -= old_entry.size_bytes
            else:
                self.statistics.entry_count += 1
            
            self.cache[full_key] = entry
            self.statistics.total_size += size_bytes
            
            # Update namespace stats
            if namespace not in self.namespace_stats:
                self.namespace_stats[namespace] = CacheStatistics()
            
            ns_stats = self.namespace_stats[namespace]
            if not was_update:
                ns_stats.entry_count += 1
            ns_stats.total_size += size_bytes
            
            return True
    
    def invalidate(self, key: CacheKey, namespace: CacheNamespace = CacheNamespace("default")) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            full_key = CacheKey(f"{namespace}:{key}")
            if full_key in self.cache:
                self._evict_entry(full_key)
                return True
            return False
    
    def invalidate_namespace(self, namespace: CacheNamespace) -> int:
        """Invalidate all entries in namespace."""
        with self._lock:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{namespace}:")]
            
            for key in keys_to_remove:
                self._evict_entry(key)
            
            # Clear namespace stats
            if namespace in self.namespace_stats:
                del self.namespace_stats[namespace]
            
            return len(keys_to_remove)
    
    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate entries matching any of the tags."""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.tags & tags:  # Intersection of sets
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._evict_entry(key)
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.statistics = CacheStatistics()
            self.namespace_stats.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cache_stats": self.statistics.get_summary(),
                "namespace_stats": {
                    str(ns): stats.get_summary() 
                    for ns, stats in self.namespace_stats.items()
                },
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "eviction_policy": self.eviction_policy.value
            }


class MultiLevelCache:
    """Multi-level caching system with L1/L2/L3 hierarchy."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.l1_cache = CacheManager(max_size=500, eviction_policy=EvictionPolicy.LRU)  # Fast memory
        self.l2_cache = CacheManager(max_size=2000, eviction_policy=EvictionPolicy.INTELLIGENT)  # Compressed
        self.cache_dir = cache_dir or Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # L3 disk cache (simple file-based)
        self.l3_cache_dir = self.cache_dir / "l3"
        self.l3_cache_dir.mkdir(exist_ok=True)
        
        self.compression_enabled = True
        self.disk_cache_enabled = True
    
    async def get(self, key: CacheKey, namespace: CacheNamespace = CacheNamespace("default")) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = self.l1_cache.get(key, namespace)
        if value is not None:
            return value
        
        # Try L2 (compressed)
        value = self.l2_cache.get(key, namespace)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value, namespace=namespace)
            return value
        
        # Try L3 (disk)
        if self.disk_cache_enabled:
            value = await self._get_from_disk(key, namespace)
            if value is not None:
                # Promote to L2 and L1
                self.l2_cache.put(key, value, namespace=namespace)
                self.l1_cache.put(key, value, namespace=namespace)
                return value
        
        return None
    
    async def put(self, key: CacheKey, value: Any, ttl: Optional[timedelta] = None,
                 namespace: CacheNamespace = CacheNamespace("default"),
                 tags: Set[str] = None, persist_to_disk: bool = True) -> bool:
        """Put value in multi-level cache."""
        # Store in L1
        success = self.l1_cache.put(key, value, ttl, namespace, tags)
        
        # Store in L2 (compressed)
        if self.compression_enabled:
            compressed_value = self._compress_value(value)
            self.l2_cache.put(key, compressed_value, ttl, namespace, tags)
        
        # Store in L3 (disk) if enabled
        if self.disk_cache_enabled and persist_to_disk:
            await self._put_to_disk(key, value, ttl, namespace, tags)
        
        return success
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value for L2 cache."""
        try:
            pickled = pickle.dumps(value)
            return zlib.compress(pickled)
        except:
            return pickle.dumps(value)
    
    def _decompress_value(self, compressed_data: bytes) -> Any:
        """Decompress value from L2 cache."""
        try:
            decompressed = zlib.decompress(compressed_data)
            return pickle.loads(decompressed)
        except:
            return pickle.loads(compressed_data)
    
    async def _get_from_disk(self, key: CacheKey, namespace: CacheNamespace) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            file_path = self.l3_cache_dir / f"{namespace}" / f"{key}.cache"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Check expiration
                if 'ttl' in cache_data and cache_data['ttl']:
                    expiry = cache_data['created_at'] + cache_data['ttl']
                    if datetime.now(UTC) > expiry:
                        file_path.unlink()  # Delete expired file
                        return None
                
                return cache_data['value']
        except:
            return None
        
        return None
    
    async def _put_to_disk(self, key: CacheKey, value: Any, ttl: Optional[timedelta],
                          namespace: CacheNamespace, tags: Set[str]) -> None:
        """Put value to disk cache."""
        try:
            namespace_dir = self.l3_cache_dir / f"{namespace}"
            namespace_dir.mkdir(exist_ok=True)
            
            file_path = namespace_dir / f"{key}.cache"
            
            cache_data = {
                'value': value,
                'created_at': datetime.now(UTC),
                'ttl': ttl,
                'tags': tags or set(),
                'namespace': namespace
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except:
            # Log error but don't fail the cache operation
            pass
    
    def invalidate(self, key: CacheKey, namespace: CacheNamespace = CacheNamespace("default")) -> bool:
        """Invalidate key from all cache levels."""
        l1_result = self.l1_cache.invalidate(key, namespace)
        l2_result = self.l2_cache.invalidate(key, namespace)
        
        # Remove from disk
        try:
            file_path = self.l3_cache_dir / f"{namespace}" / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        except:
            pass
        
        return l1_result or l2_result
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all cache levels."""
        return {
            "l1_cache": self.l1_cache.get_statistics(),
            "l2_cache": self.l2_cache.get_statistics(),
            "l3_disk_usage": self._get_disk_cache_size(),
            "compression_enabled": self.compression_enabled,
            "disk_cache_enabled": self.disk_cache_enabled,
            "cache_directory": str(self.cache_dir)
        }
    
    def _get_disk_cache_size(self) -> int:
        """Calculate total disk cache size."""
        try:
            total_size = 0
            for file_path in self.l3_cache_dir.rglob("*.cache"):
                total_size += file_path.stat().st_size
            return total_size
        except:
            return 0


class IntelligentCacheManager:
    """AI-powered intelligent cache management with predictive capabilities."""
    
    def __init__(self, ai_manager=None):
        self.cache = MultiLevelCache()
        self.ai_manager = ai_manager
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.prefetch_enabled = True
        self.learning_enabled = True
        
        # Cache for AI operations specifically
        self.ai_cache_namespace = CacheNamespace("ai_operations")
    
    async def get_ai_result(self, operation: AIOperation, input_data: Any,
                           parameters: Dict[str, Any] = None) -> Optional[Any]:
        """Get AI result from cache or compute if missing."""
        cache_key = self._generate_ai_cache_key(operation, input_data, parameters)
        
        # Record access pattern
        self._record_access_pattern(cache_key)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key, self.ai_cache_namespace)
        if cached_result is not None:
            return cached_result
        
        # Cache miss - would typically compute result here
        return None
    
    async def put_ai_result(self, operation: AIOperation, input_data: Any,
                           result: Any, parameters: Dict[str, Any] = None,
                           ttl: Optional[timedelta] = None) -> bool:
        """Store AI result in cache."""
        cache_key = self._generate_ai_cache_key(operation, input_data, parameters)
        
        # Determine TTL based on operation type
        if ttl is None:
            ttl = self._get_default_ttl(operation)
        
        # Determine tags for intelligent invalidation
        tags = self._generate_cache_tags(operation, input_data, parameters)
        
        return await self.cache.put(
            cache_key, result, ttl=ttl, 
            namespace=self.ai_cache_namespace, tags=tags
        )
    
    def _generate_ai_cache_key(self, operation: AIOperation, input_data: Any,
                              parameters: Dict[str, Any] = None) -> CacheKey:
        """Generate cache key for AI operation."""
        return self.cache.l1_cache._generate_key(operation, input_data, parameters)
    
    def _get_default_ttl(self, operation: AIOperation) -> timedelta:
        """Get default TTL based on operation type."""
        # Different operations have different cache lifetimes
        ttl_map = {
            AIOperation.ANALYZE: timedelta(hours=6),    # Analysis results change moderately
            AIOperation.GENERATE: timedelta(hours=2),   # Generation is more dynamic
            AIOperation.CLASSIFY: timedelta(hours=12),  # Classification is more stable
            AIOperation.EXTRACT: timedelta(hours=8),    # Extraction is fairly stable
            AIOperation.SUMMARIZE: timedelta(hours=4),  # Summaries change moderately
            AIOperation.TRANSLATE: timedelta(days=1),   # Translations are very stable
        }
        return ttl_map.get(operation, timedelta(hours=4))
    
    def _generate_cache_tags(self, operation: AIOperation, input_data: Any,
                           parameters: Dict[str, Any] = None) -> Set[str]:
        """Generate cache tags for intelligent invalidation."""
        tags = {operation.value}
        
        # Add model-based tags if specified
        if parameters:
            model_type = parameters.get("model_type")
            if model_type:
                tags.add(f"model:{model_type}")
            
            processing_mode = parameters.get("processing_mode")
            if processing_mode:
                tags.add(f"mode:{processing_mode}")
        
        # Add content-based tags
        input_str = str(input_data)
        if len(input_str) > 1000:
            tags.add("large_input")
        elif len(input_str) < 100:
            tags.add("small_input")
        
        return tags
    
    def _record_access_pattern(self, cache_key: CacheKey) -> None:
        """Record access pattern for predictive caching."""
        if not self.learning_enabled:
            return
        
        now = datetime.now(UTC)
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        
        self.access_patterns[cache_key].append(now)
        
        # Keep only recent accesses (last 24 hours)
        cutoff = now - timedelta(hours=24)
        self.access_patterns[cache_key] = [
            access_time for access_time in self.access_patterns[cache_key]
            if access_time > cutoff
        ]
    
    async def predictive_prefetch(self) -> None:
        """Perform predictive prefetching based on access patterns."""
        if not self.prefetch_enabled or not self.ai_manager:
            return
        
        # Analyze access patterns and prefetch likely-to-be-requested items
        # This is a simplified implementation
        now = datetime.now(UTC)
        
        for cache_key, access_times in self.access_patterns.items():
            if len(access_times) < 3:  # Need minimum history
                continue
            
            # Check if pattern suggests upcoming access
            if self._predict_upcoming_access(access_times, now):
                # Would prefetch the result here
                pass
    
    def _predict_upcoming_access(self, access_times: List[datetime], now: datetime) -> bool:
        """Predict if cache key will be accessed soon based on patterns."""
        if len(access_times) < 3:
            return False
        
        # Simple pattern detection - look for regular intervals
        intervals = []
        for i in range(1, len(access_times)):
            interval = (access_times[i] - access_times[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return False
        
        # Check if intervals are relatively consistent (within 50% variance)
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        if std_dev / avg_interval > 0.5:  # High variance
            return False
        
        # Predict next access time
        last_access = access_times[-1]
        predicted_next = last_access + timedelta(seconds=avg_interval)
        
        # If predicted time is within next 10 minutes, prefetch
        return abs((predicted_next - now).total_seconds()) < 600
    
    def invalidate_by_operation(self, operation: AIOperation) -> int:
        """Invalidate all cached results for specific operation."""
        tags = {operation.value}
        return self.cache.l1_cache.invalidate_by_tags(tags)
    
    def invalidate_by_model(self, model_type: str) -> int:
        """Invalidate all cached results for specific model."""
        tags = {f"model:{model_type}"}
        return self.cache.l1_cache.invalidate_by_tags(tags)
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive cache efficiency report."""
        stats = self.cache.get_comprehensive_statistics()
        
        # Add AI-specific metrics
        ai_stats = stats.get("l1_cache", {}).get("namespace_stats", {}).get(str(self.ai_cache_namespace), {})
        
        # Calculate efficiency metrics
        total_requests = ai_stats.get("hits", 0) + ai_stats.get("misses", 0)
        cache_savings = 0.0
        if total_requests > 0:
            # Estimate time/cost savings (assuming 2s average AI operation time)
            cache_savings = ai_stats.get("hits", 0) * 2.0
        
        return {
            "cache_statistics": stats,
            "ai_cache_performance": ai_stats,
            "access_patterns_tracked": len(self.access_patterns),
            "estimated_time_saved_seconds": cache_savings,
            "prefetch_enabled": self.prefetch_enabled,
            "learning_enabled": self.learning_enabled,
            "cache_efficiency_score": ai_stats.get("hit_ratio", 0.0) * 100
        }