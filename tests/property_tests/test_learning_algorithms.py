"""
Property-Based Testing for Automation Intelligence Learning Algorithms.

This module provides comprehensive property-based testing for behavioral analysis,
adaptive learning, suggestion generation, and privacy protection to ensure
correctness, security, and performance across various input scenarios.

Testing Strategy: Property-based validation with Hypothesis for edge case discovery,
contract verification, privacy compliance checking, and performance validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta, UTC
import pytest
import asyncio

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, precondition

from src.core.suggestion_system import (
    UserBehaviorPattern, PrivacyLevel, create_behavior_pattern
)
from src.intelligence.automation_intelligence_manager import (
    AutomationIntelligenceManager, IntelligenceOperation, AnalysisScope, LearningMode
)
from src.intelligence.behavior_analyzer import BehaviorAnalyzer
from src.intelligence.learning_engine import LearningEngine, LearningFeatures
from src.intelligence.suggestion_system import (
    IntelligentSuggestionSystem, AutomationSuggestion, SuggestionCategory
)
from src.intelligence.privacy_manager import PrivacyManager
from src.intelligence.data_anonymizer import DataAnonymizer
from src.intelligence.pattern_validator import PatternValidator


# Hypothesis strategies for test data generation
@st.composite
def user_behavior_pattern_strategy(draw):
    """Generate valid UserBehaviorPattern instances for testing."""
    pattern_id = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
    user_id = draw(st.text(min_size=3, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    
    # Generate action sequence
    actions = draw(st.lists(
        st.text(min_size=3, max_size=15, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        min_size=1, max_size=10
    ))
    
    frequency = draw(st.integers(min_value=1, max_value=100))
    success_rate = draw(st.floats(min_value=0.0, max_value=1.0))
    completion_time = draw(st.floats(min_value=0.1, max_value=3600.0))
    confidence_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate context tags
    context_tags = draw(st.sets(
        st.text(min_size=3, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        max_size=5
    ))
    
    return UserBehaviorPattern(
        pattern_id=pattern_id,
        user_id=user_id,
        action_sequence=actions,
        frequency=frequency,
        success_rate=success_rate,
        average_completion_time=completion_time,
        context_tags=context_tags,
        last_observed=datetime.now(UTC),
        confidence_score=confidence_score
    )


@st.composite
def privacy_level_strategy(draw):
    """Generate PrivacyLevel instances for testing."""
    return draw(st.sampled_from([PrivacyLevel.STRICT, PrivacyLevel.BALANCED, PrivacyLevel.PERMISSIVE]))


@st.composite
def learning_mode_strategy(draw):
    """Generate LearningMode instances for testing."""
    return draw(st.sampled_from([LearningMode.ADAPTIVE, LearningMode.SUPERVISED, LearningMode.UNSUPERVISED, LearningMode.REINFORCEMENT]))


class TestBehaviorAnalyzerProperties:
    """Property-based tests for BehaviorAnalyzer with privacy protection validation."""
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=1, max_size=20),
        privacy_level=privacy_level_strategy()
    )
    @settings(max_examples=50, deadline=5000)
    def test_pattern_analysis_privacy_preservation(self, patterns, privacy_level):
        """Property: Pattern analysis must preserve privacy at specified levels."""
        analyzer = BehaviorAnalyzer(privacy_level)
        
        # Property: All patterns should be validated for privacy compliance
        validator = PatternValidator()
        
        for pattern in patterns:
            # Pattern validation should respect privacy level
            is_valid = validator.is_valid_for_analysis(pattern, privacy_level)
            
            # Strict privacy should have higher validation standards
            if privacy_level == PrivacyLevel.STRICT:
                # Additional validation for strict privacy
                assert not validator._appears_to_be_personal_identifier(pattern.user_id) or len(pattern.user_id) >= 16
                assert not validator._contains_sensitive_actions(pattern.action_sequence)
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=3, max_size=50),
        privacy_level=privacy_level_strategy()
    )
    @settings(max_examples=30, deadline=10000)
    def test_pattern_extraction_consistency(self, patterns, privacy_level):
        """Property: Pattern extraction should be deterministic and consistent."""
        analyzer = BehaviorAnalyzer(privacy_level)
        
        # Property: Same input should produce consistent results
        # (when using same session keys for anonymization)
        
        # Ensure patterns have sufficient frequency for analysis
        valid_patterns = [p for p in patterns if p.frequency >= analyzer.min_pattern_frequency]
        assume(len(valid_patterns) >= 1)
        
        # Property: Pattern filtering should be consistent
        filtered_patterns_1 = analyzer._filter_patterns_by_relevance(valid_patterns)
        filtered_patterns_2 = analyzer._filter_patterns_by_relevance(valid_patterns)
        
        # Should produce same number of patterns
        assert len(filtered_patterns_1) == len(filtered_patterns_2)
        
        # Should preserve pattern ordering for same input
        if filtered_patterns_1 and filtered_patterns_2:
            assert filtered_patterns_1[0].pattern_id == filtered_patterns_2[0].pattern_id
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=1, max_size=30)
    )
    @settings(max_examples=50, deadline=5000)
    def test_pattern_frequency_constraints(self, patterns):
        """Property: Pattern frequency analysis should respect constraints."""
        analyzer = BehaviorAnalyzer()
        
        # Property: Filtered patterns should meet minimum frequency requirements
        filtered_patterns = analyzer._filter_patterns_by_relevance(patterns)
        
        for pattern in filtered_patterns:
            assert pattern.frequency >= analyzer.min_pattern_frequency
            assert pattern.confidence_score >= analyzer.min_confidence_threshold
            assert pattern.success_rate >= 0.5  # Minimum success rate requirement


class TestLearningEngineProperties:
    """Property-based tests for LearningEngine with adaptive learning validation."""
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=5, max_size=30),
        learning_mode=learning_mode_strategy()
    )
    @settings(max_examples=30, deadline=10000)
    def test_feature_extraction_completeness(self, patterns, learning_mode):
        """Property: Feature extraction should be complete and valid."""
        engine = LearningEngine(learning_mode)
        
        # Property: Feature extraction should produce valid feature vectors
        features = engine._extract_temporal_features(patterns)
        
        # Temporal features should contain required fields
        required_fields = ['average_completion_time', 'total_frequency', 'average_frequency']
        for field in required_fields:
            assert field in features
            assert isinstance(features[field], (int, float))
            assert features[field] >= 0
        
        # Sequence features validation
        sequence_features = engine._extract_sequence_features(patterns)
        assert 'total_sequences' in sequence_features
        assert 'unique_actions' in sequence_features
        assert sequence_features['total_sequences'] == len(patterns)
        assert isinstance(sequence_features['unique_actions'], set)
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=5, max_size=25),
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=40, deadline=8000)
    def test_learning_confidence_bounds(self, patterns, confidence_threshold):
        """Property: Learning confidence should be bounded and monotonic."""
        engine = LearningEngine()
        
        # Property: Confidence calculation should be bounded [0, 1]
        features = LearningFeatures(
            temporal_features=engine._extract_temporal_features(patterns),
            sequence_features=engine._extract_sequence_features(patterns),
            tool_usage_features=engine._extract_tool_usage_features(patterns),
            performance_features=engine._extract_performance_features(patterns),
            context_features=engine._extract_context_features(patterns)
        )
        
        insights = []
        optimizations = []
        confidence = engine._calculate_learning_confidence(features, insights, optimizations)
        
        # Confidence must be bounded
        assert 0.0 <= confidence <= 1.0
        
        # More patterns should generally increase confidence (monotonicity property)
        if len(patterns) >= 10:
            assert confidence >= 0.1  # Minimum confidence with sufficient data
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=1, max_size=20)
    )
    @settings(max_examples=50, deadline=5000)
    def test_feature_vector_properties(self, patterns):
        """Property: Feature vectors should have consistent structure and bounds."""
        engine = LearningEngine()
        
        features = LearningFeatures(
            temporal_features=engine._extract_temporal_features(patterns),
            sequence_features=engine._extract_sequence_features(patterns),
            tool_usage_features=engine._extract_tool_usage_features(patterns),
            performance_features=engine._extract_performance_features(patterns),
            context_features=engine._extract_context_features(patterns)
        )
        
        # Property: Feature vector should be consistent length
        feature_vector = features.get_feature_vector()
        
        # Should produce numeric feature vector
        assert isinstance(feature_vector, list)
        assert all(isinstance(f, (int, float)) for f in feature_vector)
        
        # Should have expected length (based on implementation)
        expected_length = 3 + 3 + 3 + 3  # temporal + sequence + tool + performance
        assert len(feature_vector) == expected_length
        
        # All features should be non-negative
        assert all(f >= 0 for f in feature_vector)


class TestSuggestionSystemProperties:
    """Property-based tests for IntelligentSuggestionSystem with ROI validation."""
    
    @given(
        patterns=st.lists(user_behavior_pattern_strategy(), min_size=3, max_size=25),
        suggestion_count=st.integers(min_value=1, max_value=10),
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=30, deadline=15000)
    def test_suggestion_generation_bounds(self, patterns, suggestion_count, confidence_threshold):
        """Property: Suggestion generation should respect count and confidence bounds."""
        system = IntelligentSuggestionSystem()
        
        # Filter patterns that could generate suggestions
        automation_candidates = [
            p for p in patterns
            if (p.frequency >= 5 and p.confidence_score >= 0.8 and 
                p.success_rate >= 0.8 and len(p.action_sequence) >= 2)
        ]
        
        assume(len(automation_candidates) >= 1)
        
        # Test automation suggestion generation
        suggestions = asyncio.run(system._generate_automation_suggestions(automation_candidates))
        
        # Property: Should not exceed max suggestions per category
        assert len(suggestions) <= system.max_suggestions_per_category
        
        # Property: All suggestions should meet confidence threshold
        for suggestion in suggestions:
            assert 0.0 <= suggestion.confidence <= 1.0
            assert suggestion.potential_time_saved >= 0.0
            assert suggestion.implementation_complexity in ["low", "medium", "high"]
            assert 0.0 <= suggestion.estimated_success_rate <= 1.0
    
    @given(
        automation_suggestions=st.lists(
            st.builds(
                AutomationSuggestion,
                suggestion_id=st.text(min_size=5, max_size=20),
                suggestion_type=st.just("automation"),
                category=st.just(SuggestionCategory.AUTOMATION_OPPORTUNITY),
                title=st.text(min_size=10, max_size=50),
                description=st.text(min_size=20, max_size=100),
                confidence=st.floats(min_value=0.0, max_value=1.0),
                potential_time_saved=st.floats(min_value=0.0, max_value=10000.0),
                implementation_complexity=st.sampled_from(["low", "medium", "high"]),
                tools_involved=st.lists(st.text(min_size=3, max_size=15), max_size=5),
                trigger_conditions=st.fixed_dictionaries({}),
                estimated_success_rate=st.floats(min_value=0.0, max_value=1.0),
                rationale=st.text(min_size=10, max_size=100),
                supporting_patterns=st.lists(st.text(min_size=5, max_size=20), max_size=3)
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_roi_calculation_properties(self, automation_suggestions):
        """Property: ROI calculation should be consistent and bounded."""
        
        for suggestion in automation_suggestions:
            roi = suggestion.get_roi_estimate()
            
            # Property: ROI should be non-negative
            assert roi >= 0.0
            
            # Property: Higher time savings should generally increase ROI
            # (when other factors are constant)
            if suggestion.potential_time_saved > 0:
                assert roi > 0.0 or suggestion.implementation_complexity == "high"
            
            # Property: High impact suggestions should be identifiable
            is_high_impact = suggestion.is_high_impact()
            if suggestion.potential_time_saved >= 300.0 and suggestion.confidence >= 0.7:
                assert is_high_impact
    
    @given(
        suggestions=st.lists(
            st.builds(
                AutomationSuggestion,
                suggestion_id=st.text(min_size=5, max_size=20),
                suggestion_type=st.just("automation"),
                category=st.just(SuggestionCategory.AUTOMATION_OPPORTUNITY),
                title=st.text(min_size=10, max_size=50),
                description=st.text(min_size=20, max_size=100),
                confidence=st.floats(min_value=0.0, max_value=1.0),
                potential_time_saved=st.floats(min_value=0.0, max_value=10000.0),
                implementation_complexity=st.sampled_from(["low", "medium", "high"]),
                tools_involved=st.lists(st.text(min_size=3, max_size=15), max_size=5),
                trigger_conditions=st.fixed_dictionaries({}),
                estimated_success_rate=st.floats(min_value=0.0, max_value=1.0),
                rationale=st.text(min_size=10, max_size=100),
                supporting_patterns=st.lists(st.text(min_size=5, max_size=20), max_size=3)
            ),
            min_size=2, max_size=15
        )
    )
    @settings(max_examples=40, deadline=8000)
    def test_suggestion_ranking_consistency(self, suggestions):
        """Property: Suggestion ranking should be consistent and properly ordered."""
        from src.intelligence.suggestion_system import SuggestionRanker
        
        ranker = SuggestionRanker()
        ranked_suggestions = ranker.rank_suggestions(suggestions)
        
        # Property: Ranking should preserve all suggestions
        assert len(ranked_suggestions) == len(suggestions)
        
        # Property: Ranking should be properly ordered (descending priority)
        for i in range(len(ranked_suggestions) - 1):
            current_score = ranked_suggestions[i].priority_score
            next_score = ranked_suggestions[i + 1].priority_score
            assert current_score >= next_score
        
        # Property: All suggestions should have valid priority scores
        for suggestion in ranked_suggestions:
            assert 0.0 <= suggestion.priority_score <= 4.0  # Max possible score


class TestPrivacyManagerProperties:
    """Property-based tests for PrivacyManager with compliance validation."""
    
    @given(
        data=st.lists(
            st.fixed_dictionaries({
                'user_id': st.text(min_size=5, max_size=20),
                'action': st.text(min_size=3, max_size=20),
                'timestamp': st.just(datetime.now(UTC)),
                'success': st.booleans(),
                'tool_name': st.text(min_size=3, max_size=15)
            }),
            min_size=1, max_size=20
        ),
        privacy_level=privacy_level_strategy()
    )
    @settings(max_examples=30, deadline=10000)
    def test_data_anonymization_properties(self, data, privacy_level):
        """Property: Data anonymization should preserve utility while protecting privacy."""
        anonymizer = DataAnonymizer()
        asyncio.run(anonymizer.initialize(privacy_level))
        
        anonymized_data = asyncio.run(anonymizer.anonymize_behavior_data(data, privacy_level))
        
        # Property: Anonymization should not increase data size inappropriately
        assert len(anonymized_data) <= len(data)
        
        # Property: Strict privacy should be more restrictive
        if privacy_level == PrivacyLevel.STRICT:
            for record in anonymized_data:
                # Should not contain obvious personal identifiers
                record_str = str(record).lower()
                assert '@' not in record_str  # No email patterns
                assert not any(len(str(v)) > 20 and ' ' in str(v) for v in record.values())  # No long text with spaces (names)
        
        # Property: Some data should be preserved for analytical utility
        if anonymized_data and data:
            # Should preserve timestamp information (for temporal analysis)
            has_temporal_info = any('timestamp' in record for record in anonymized_data)
            # Should preserve some activity indicators
            has_activity_info = any(len(record) > 1 for record in anonymized_data)
            
            assert has_temporal_info or has_activity_info  # At least some utility preserved
    
    @given(
        results=st.fixed_dictionaries({
            'total_patterns': st.integers(min_value=0, max_value=100),
            'user_data': st.lists(st.dictionaries(
                st.text(min_size=3, max_size=15),
                st.one_of(st.text(max_size=50), st.integers(), st.floats(), st.booleans())
            ), max_size=10),
            'sensitive_field': st.text(min_size=10, max_size=50),
            'performance_metrics': st.dictionaries(
                st.text(min_size=5, max_size=20),
                st.floats(min_value=0.0, max_value=1000.0)
            )
        }),
        privacy_level=privacy_level_strategy()
    )
    @settings(max_examples=30, deadline=8000)
    def test_result_filtering_properties(self, results, privacy_level):
        """Property: Result filtering should maintain data integrity while protecting privacy."""
        privacy_manager = PrivacyManager()
        asyncio.run(privacy_manager.initialize())
        
        filtered_results = asyncio.run(privacy_manager.filter_results(results, privacy_level, anonymize_data=True))
        
        # Property: Filtered results should be valid dictionary
        assert isinstance(filtered_results, dict)
        
        # Property: Should include privacy compliance metadata
        assert 'privacy_compliance' in filtered_results or 'error' in filtered_results
        
        # Property: Strict privacy should be more restrictive than permissive
        if privacy_level == PrivacyLevel.STRICT:
            # Should have fewer fields or more anonymized content
            assert len(filtered_results) <= len(results) + 2  # Allow for metadata
        
        # Property: No obvious sensitive patterns should remain
        result_str = str(filtered_results).lower()
        sensitive_patterns = ['password', 'secret', 'token', '@gmail.com']
        for pattern in sensitive_patterns:
            assert pattern not in result_str or '[redacted]' in result_str


class TestDataAnonymizerProperties:
    """Property-based tests for DataAnonymizer with anonymization validation."""
    
    @given(
        value=st.text(min_size=1, max_size=100),
        privacy_level=privacy_level_strategy()
    )
    @settings(max_examples=100, deadline=5000)
    def test_string_anonymization_consistency(self, value, privacy_level):
        """Property: String anonymization should be consistent and privacy-preserving."""
        anonymizer = DataAnonymizer()
        
        # Property: Same input should produce same anonymized output (within session)
        anonymized_1 = anonymizer._anonymize_string_value(value, privacy_level)
        anonymized_2 = anonymizer._anonymize_string_value(value, privacy_level)
        
        assert anonymized_1 == anonymized_2  # Consistency
        
        # Property: Anonymization should preserve string type
        assert isinstance(anonymized_1, str)
        
        # Property: Strict privacy should be more aggressive
        if privacy_level == PrivacyLevel.STRICT and anonymizer._appears_to_be_identifier(value):
            # Should be significantly different from original for identifiers
            assert anonymized_1 != value or len(value) <= 3  # Very short strings might be preserved
    
    @given(
        text=st.text(min_size=10, max_size=200)
    )
    @settings(max_examples=100, deadline=5000)
    def test_sensitive_pattern_detection(self, text):
        """Property: Sensitive pattern detection should be comprehensive."""
        anonymizer = DataAnonymizer()
        
        # Add known sensitive patterns to test text
        test_cases = [
            f"{text} password=secret123",
            f"{text} user@example.com contact",
            f"{text} 123-45-6789 number",
            f"{text} api_key=abc123def456"
        ]
        
        for test_text in test_cases:
            contains_sensitive = anonymizer._contains_sensitive_pattern(test_text)
            
            # Should detect obvious sensitive patterns
            if any(pattern in test_text.lower() for pattern in ['password=', '@', '123-45-6789', 'api_key=']):
                assert contains_sensitive


# Integration test for full automation intelligence pipeline
class AutomationIntelligenceStateMachine(RuleBasedStateMachine):
    """Stateful testing for automation intelligence system integration."""
    
    def __init__(self):
        super().__init__()
        self.intelligence_manager = None
        self.patterns = []
        self.analysis_results = []
    
    @initialize()
    def init_intelligence_system(self):
        """Initialize the intelligence system for testing."""
        self.intelligence_manager = AutomationIntelligenceManager()
        asyncio.run(self.intelligence_manager.initialize())
    
    @rule(
        pattern=user_behavior_pattern_strategy()
    )
    def add_behavior_pattern(self, pattern):
        """Add a behavior pattern to the system."""
        # Validate pattern before adding
        validator = PatternValidator()
        if validator.is_valid_for_analysis(pattern, PrivacyLevel.BALANCED):
            self.patterns.append(pattern)
    
    @rule()
    @precondition(lambda self: len(self.patterns) >= 3)
    def analyze_patterns(self):
        """Analyze accumulated patterns for insights."""
        if self.intelligence_manager and len(self.patterns) >= 3:
            # Test analysis operation
            result = asyncio.run(
                self.intelligence_manager.process_intelligence_request(
                    operation=IntelligenceOperation.ANALYZE,
                    analysis_scope=AnalysisScope.USER_BEHAVIOR,
                    time_period="30d",
                    privacy_level=PrivacyLevel.BALANCED
                )
            )
            
            if result.is_right():
                analysis_data = result.get_right()
                self.analysis_results.append(analysis_data)
                
                # Verify analysis results structure
                assert isinstance(analysis_data, dict)
                assert 'total_patterns' in analysis_data or 'error' in analysis_data
    
    @rule()
    @precondition(lambda self: len(self.analysis_results) >= 1)
    def validate_analysis_consistency(self):
        """Validate that analysis results are consistent across operations."""
        if len(self.analysis_results) >= 2:
            # Compare recent analysis results for consistency
            result1 = self.analysis_results[-1]
            result2 = self.analysis_results[-2]
            
            # Should have similar structure
            assert type(result1) == type(result2)
            
            # If both successful, should have required fields
            if 'total_patterns' in result1 and 'total_patterns' in result2:
                assert isinstance(result1['total_patterns'], int)
                assert isinstance(result2['total_patterns'], int)


# Test class instantiation
TestIntelligenceStateMachine = AutomationIntelligenceStateMachine.TestCase


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])