"""
Advanced Behavioral Pattern Analysis with Privacy Protection.

This module implements sophisticated behavioral pattern analysis, user workflow recognition,
and intelligent pattern extraction while maintaining strict privacy protection and
comprehensive security validation throughout the analysis process.

Security: Privacy-first design with configurable anonymization and data protection.
Performance: Optimized pattern recognition with intelligent caching and filtering.
Type Safety: Complete branded type system with contract-driven validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta, UTC
from collections import defaultdict, Counter
import asyncio
import statistics
import hashlib

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import PrivacyLevel, UserBehaviorPattern
from src.core.errors import IntelligenceError
from src.intelligence.intelligence_types import AnalysisScope
from src.intelligence.data_anonymizer import DataAnonymizer
from src.intelligence.pattern_validator import PatternValidator

logger = get_logger(__name__)


class BehaviorAnalyzer:
    """Advanced behavioral pattern analysis with comprehensive privacy protection."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.HIGH):
        self.privacy_level = privacy_level
        self.pattern_cache: Dict[str, UserBehaviorPattern] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        self.anonymizer = DataAnonymizer()
        self.pattern_validator = PatternValidator()
        
        # Analysis configuration
        self.min_pattern_frequency = 3
        self.min_confidence_threshold = 0.5
        self.max_patterns_per_analysis = 100
        
    async def initialize(self) -> Either[IntelligenceError, None]:
        """Initialize behavior analyzer with privacy-compliant settings."""
        try:
            # Initialize anonymizer with privacy level
            await self.anonymizer.initialize(self.privacy_level)
            
            # Configure analysis parameters based on privacy level
            self._configure_privacy_settings()
            
            logger.info(f"Behavior analyzer initialized with privacy level: {self.privacy_level.value}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Behavior analyzer initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))
    
    @require(lambda self, time_period: time_period in ["1d", "7d", "30d", "90d", "all"])
    async def analyze_user_behavior(
        self,
        time_period: str = "30d",
        analysis_scope: AnalysisScope = AnalysisScope.USER_BEHAVIOR,
        privacy_level: Optional[PrivacyLevel] = None
    ) -> Either[IntelligenceError, List[UserBehaviorPattern]]:
        """
        Analyze user behavior patterns with comprehensive privacy protection.
        
        Performs sophisticated pattern recognition on user behavioral data while
        maintaining strict privacy compliance and security validation throughout
        the analysis process.
        
        Args:
            time_period: Time window for pattern analysis
            analysis_scope: Scope of behavioral analysis
            privacy_level: Privacy protection level override
            
        Returns:
            Either error or list of validated behavior patterns
            
        Security:
            - Complete data anonymization based on privacy level
            - Secure pattern extraction with no sensitive data retention
            - Privacy-compliant behavioral data processing
        """
        try:
            effective_privacy_level = privacy_level or self.privacy_level
            
            # Collect behavioral data with privacy filtering
            raw_data_result = await self._collect_behavioral_data(time_period, analysis_scope)
            if raw_data_result.is_left():
                return raw_data_result
            
            raw_data = raw_data_result.get_right()
            
            # Apply privacy protection and anonymization
            anonymized_data = await self.anonymizer.anonymize_behavior_data(
                raw_data, effective_privacy_level
            )
            
            # Extract behavioral patterns using advanced algorithms
            patterns = await self._extract_behavior_patterns(anonymized_data, analysis_scope)
            
            # Validate patterns for security and privacy compliance
            validated_patterns = []
            for pattern in patterns:
                if self.pattern_validator.is_valid_for_analysis(pattern, effective_privacy_level):
                    validated_patterns.append(pattern)
            
            # Filter patterns by relevance and confidence
            filtered_patterns = self._filter_patterns_by_relevance(validated_patterns)
            
            # Cache results for performance optimization
            self._cache_analysis_results(filtered_patterns, time_period, analysis_scope)
            
            logger.info(f"Analyzed {len(filtered_patterns)} behavioral patterns for scope: {analysis_scope.value}")
            return Either.right(filtered_patterns)
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {str(e)}")
            return Either.left(IntelligenceError.behavior_analysis_failed(str(e)))
    
    async def _collect_behavioral_data(
        self, 
        time_period: str, 
        analysis_scope: AnalysisScope
    ) -> Either[IntelligenceError, List[Dict[str, Any]]]:
        """Collect behavioral data from system logs with privacy protection."""
        try:
            # Calculate time window
            time_deltas = {
                "1d": timedelta(days=1),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30),
                "90d": timedelta(days=90)
            }
            
            if time_period != "all" and time_period not in time_deltas:
                return Either.left(IntelligenceError.behavior_analysis_failed(
                    f"Invalid time period: {time_period}"
                ))
            
            cutoff_time = None if time_period == "all" else datetime.now(UTC) - time_deltas[time_period]
            
            # Collect data based on analysis scope
            behavioral_data = []
            
            if analysis_scope in [AnalysisScope.USER_BEHAVIOR, AnalysisScope.USAGE]:
                tool_usage_data = await self._get_tool_usage_data(cutoff_time)
                behavioral_data.extend(tool_usage_data)
            
            if analysis_scope in [AnalysisScope.AUTOMATION_PATTERNS, AnalysisScope.WORKFLOW]:
                automation_data = await self._get_automation_sequence_data(cutoff_time)
                behavioral_data.extend(automation_data)
            
            if analysis_scope == AnalysisScope.PERFORMANCE:
                performance_data = await self._get_performance_behavioral_data(cutoff_time)
                behavioral_data.extend(performance_data)
            
            if analysis_scope == AnalysisScope.ERROR_PATTERNS:
                error_data = await self._get_error_pattern_data(cutoff_time)
                behavioral_data.extend(error_data)
            
            return Either.right(behavioral_data)
            
        except Exception as e:
            logger.error(f"Behavioral data collection failed: {str(e)}")
            return Either.left(IntelligenceError.behavior_analysis_failed(str(e)))
    
    async def _extract_behavior_patterns(
        self, 
        data: List[Dict[str, Any]], 
        scope: AnalysisScope
    ) -> List[UserBehaviorPattern]:
        """Extract meaningful behavior patterns from anonymized data."""
        try:
            patterns = []
            
            # Group data by user and action sequences
            user_sequences = self._group_data_by_user_sequences(data)
            
            for user_id, sequences in user_sequences.items():
                user_patterns = await self._analyze_user_sequences(user_id, sequences, scope)
                patterns.extend(user_patterns)
            
            # Apply pattern recognition algorithms
            refined_patterns = await self._apply_pattern_recognition_algorithms(patterns)
            
            return refined_patterns
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {str(e)}")
            return []
    
    def _group_data_by_user_sequences(self, data: List[Dict[str, Any]]) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Group behavioral data by user and sequential action patterns."""
        user_data = defaultdict(list)
        
        # Group by user_id (anonymized)
        for item in data:
            user_id = item.get('user_id', 'anonymous')
            user_data[user_id].append(item)
        
        # Create sequences from temporal data
        user_sequences = {}
        for user_id, user_items in user_data.items():
            # Sort by timestamp
            sorted_items = sorted(user_items, key=lambda x: x.get('timestamp', datetime.min))
            
            # Group into sequences based on time gaps
            sequences = []
            current_sequence = []
            last_timestamp = None
            
            for item in sorted_items:
                timestamp = item.get('timestamp', datetime.now(UTC))
                
                # Start new sequence if gap > 30 minutes
                if (last_timestamp and 
                    (timestamp - last_timestamp).total_seconds() > 1800):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = [item]
                else:
                    current_sequence.append(item)
                
                last_timestamp = timestamp
            
            if current_sequence:
                sequences.append(current_sequence)
            
            user_sequences[user_id] = sequences
        
        return user_sequences
    
    async def _analyze_user_sequences(
        self, 
        user_id: str, 
        sequences: List[List[Dict[str, Any]]], 
        scope: AnalysisScope
    ) -> List[UserBehaviorPattern]:
        """Analyze user action sequences to identify behavioral patterns."""
        patterns = []
        
        # Analyze sequence patterns
        sequence_patterns = self._identify_sequence_patterns(sequences)
        
        for pattern_data in sequence_patterns:
            # Create behavior pattern with privacy protection
            try:
                pattern = await self._create_behavior_pattern(
                    user_id, pattern_data, scope
                )
                if pattern:
                    patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Failed to create pattern for user {user_id}: {str(e)}")
                continue
        
        return patterns
    
    def _identify_sequence_patterns(self, sequences: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify repeating patterns within action sequences."""
        pattern_candidates = []
        
        # Extract action sequences
        action_sequences = []
        for sequence in sequences:
            actions = [item.get('action', 'unknown') for item in sequence]
            if len(actions) >= 2:  # Minimum sequence length
                action_sequences.append((actions, sequence))
        
        # Find common subsequences
        subsequence_counts = defaultdict(list)
        
        for actions, sequence_data in action_sequences:
            # Generate subsequences of length 2-5
            for length in range(2, min(6, len(actions) + 1)):
                for i in range(len(actions) - length + 1):
                    subseq = tuple(actions[i:i + length])
                    subsequence_counts[subseq].append({
                        'sequence_data': sequence_data[i:i + length],
                        'full_sequence': sequence_data
                    })
        
        # Convert frequent subsequences to patterns
        for subseq, occurrences in subsequence_counts.items():
            if len(occurrences) >= self.min_pattern_frequency:
                pattern_data = self._create_pattern_data_from_subsequence(subseq, occurrences)
                pattern_candidates.append(pattern_data)
        
        return pattern_candidates
    
    def _create_pattern_data_from_subsequence(
        self, 
        subsequence: Tuple[str, ...], 
        occurrences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create pattern data structure from identified subsequence."""
        # Calculate pattern metrics
        frequency = len(occurrences)
        
        # Extract timing and success information
        durations = []
        success_rates = []
        tools_used = set()
        time_contexts = []
        
        for occurrence in occurrences:
            sequence_data = occurrence['sequence_data']
            
            # Calculate duration
            if len(sequence_data) >= 2:
                start_time = sequence_data[0].get('timestamp', datetime.now(UTC))
                end_time = sequence_data[-1].get('timestamp', datetime.now(UTC))
                duration = (end_time - start_time).total_seconds()
                durations.append(max(1.0, duration))  # Minimum 1 second
            
            # Extract success information
            successes = [item.get('success', True) for item in sequence_data]
            success_rate = sum(successes) / len(successes) if successes else 1.0
            success_rates.append(success_rate)
            
            # Extract tools and context
            for item in sequence_data:
                if 'tool_name' in item:
                    tools_used.add(item['tool_name'])
                
                timestamp = item.get('timestamp', datetime.now(UTC))
                time_contexts.append(timestamp.hour)
        
        # Calculate aggregate metrics
        avg_duration = statistics.mean(durations) if durations else 1.0
        avg_success_rate = statistics.mean(success_rates) if success_rates else 1.0
        
        # Calculate confidence based on consistency
        confidence = min(1.0, frequency / 10.0)  # Higher frequency = higher confidence
        if durations:
            duration_variance = statistics.stdev(durations) if len(durations) > 1 else 0
            consistency_factor = max(0.5, 1.0 - (duration_variance / avg_duration))
            confidence *= consistency_factor
        
        return {
            'pattern_type': f"sequence_{len(subsequence)}_step",
            'action_sequence': list(subsequence),
            'frequency': frequency,
            'average_duration': avg_duration,
            'success_rate': avg_success_rate,
            'confidence': confidence,
            'tools_used': list(tools_used),
            'time_contexts': time_contexts,
            'occurrences': occurrences
        }
    
    async def _create_behavior_pattern(
        self, 
        user_id: str, 
        pattern_data: Dict[str, Any], 
        scope: AnalysisScope
    ) -> Optional[UserBehaviorPattern]:
        """Create validated behavior pattern from pattern data."""
        try:
            # Generate pattern ID with privacy protection
            pattern_content = f"{pattern_data['pattern_type']}_{pattern_data['action_sequence']}"
            pattern_id = self._generate_secure_pattern_id(user_id, pattern_content)
            
            # Extract context tags
            context_tags = set()
            context_tags.add(f"scope:{scope.value}")
            context_tags.add(f"type:{pattern_data['pattern_type']}")
            
            for tool in pattern_data.get('tools_used', []):
                context_tags.add(f"tool:{tool}")
            
            # Create behavior pattern
            pattern = UserBehaviorPattern(
                pattern_id=pattern_id,
                user_id=self._anonymize_user_id(user_id),
                action_sequence=pattern_data['action_sequence'],
                frequency=pattern_data['frequency'],
                success_rate=pattern_data['success_rate'],
                average_completion_time=pattern_data['average_duration'],
                context_tags=context_tags,
                last_observed=datetime.now(UTC),
                confidence_score=pattern_data['confidence']
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Failed to create behavior pattern: {str(e)}")
            return None
    
    def _filter_patterns_by_relevance(self, patterns: List[UserBehaviorPattern]) -> List[UserBehaviorPattern]:
        """Filter patterns by relevance, confidence, and utility."""
        filtered = []
        
        for pattern in patterns:
            # Apply relevance filters
            if (pattern.frequency >= self.min_pattern_frequency and 
                pattern.confidence_score >= self.min_confidence_threshold and
                pattern.success_rate >= 0.5):  # Minimum 50% success rate
                filtered.append(pattern)
        
        # Sort by importance (frequency * confidence * success_rate)
        filtered.sort(
            key=lambda p: p.frequency * p.confidence_score * p.success_rate,
            reverse=True
        )
        
        # Limit to max patterns for performance
        return filtered[:self.max_patterns_per_analysis]
    
    def _configure_privacy_settings(self) -> None:
        """Configure analysis parameters based on privacy level."""
        if self.privacy_level == PrivacyLevel.MAXIMUM:
            self.min_pattern_frequency = 5  # Higher threshold for strict privacy
            self.min_confidence_threshold = 0.7
            self.max_patterns_per_analysis = 50
        elif self.privacy_level == PrivacyLevel.BALANCED:
            self.min_pattern_frequency = 3
            self.min_confidence_threshold = 0.5
            self.max_patterns_per_analysis = 100
        else:  # PERMISSIVE
            self.min_pattern_frequency = 2
            self.min_confidence_threshold = 0.3
            self.max_patterns_per_analysis = 200
    
    def _generate_secure_pattern_id(self, user_id: str, pattern_content: str) -> str:
        """Generate secure, anonymized pattern ID."""
        content_hash = hashlib.sha256(f"{user_id}_{pattern_content}".encode()).hexdigest()
        return f"pattern_{content_hash[:16]}"
    
    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID based on privacy level."""
        if self.privacy_level == PrivacyLevel.MAXIMUM:
            return hashlib.sha256(user_id.encode()).hexdigest()[:16]
        elif self.privacy_level == PrivacyLevel.BALANCED:
            return f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
        else:
            return user_id  # Keep original for permissive level
    
    def _cache_analysis_results(
        self, 
        patterns: List[UserBehaviorPattern], 
        time_period: str, 
        scope: AnalysisScope
    ) -> None:
        """Cache analysis results for performance optimization."""
        cache_key = f"{time_period}_{scope.value}_{self.privacy_level.value}"
        self.pattern_cache[cache_key] = {
            'patterns': patterns,
            'timestamp': datetime.now(UTC),
            'count': len(patterns)
        }
        
        # Limit cache size
        if len(self.pattern_cache) > 20:
            oldest_key = min(self.pattern_cache.keys(), 
                           key=lambda k: self.pattern_cache[k]['timestamp'])
            del self.pattern_cache[oldest_key]
    
    # Placeholder methods for data collection
    async def _get_tool_usage_data(self, cutoff_time: Optional[datetime]) -> List[Dict[str, Any]]:
        """Get tool usage data for behavioral analysis."""
        # This would interface with actual system logs
        return []
    
    async def _get_automation_sequence_data(self, cutoff_time: Optional[datetime]) -> List[Dict[str, Any]]:
        """Get automation sequence data for pattern analysis."""
        return []
    
    async def _get_performance_behavioral_data(self, cutoff_time: Optional[datetime]) -> List[Dict[str, Any]]:
        """Get performance-related behavioral data."""
        return []
    
    async def _get_error_pattern_data(self, cutoff_time: Optional[datetime]) -> List[Dict[str, Any]]:
        """Get error pattern data for analysis."""
        return []
    
    async def _apply_pattern_recognition_algorithms(self, patterns: List[UserBehaviorPattern]) -> List[UserBehaviorPattern]:
        """Apply advanced pattern recognition algorithms."""
        # Placeholder for advanced pattern recognition
        return patterns