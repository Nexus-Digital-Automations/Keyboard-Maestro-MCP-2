"""
Pattern Validation for Behavioral Intelligence with Security Compliance.

This module provides comprehensive validation for behavioral patterns, ensuring
privacy compliance, security validation, and analytical utility while maintaining
strict quality standards for machine learning applications.

Security: Multi-level validation with privacy compliance checking.
Performance: Efficient validation with minimal processing overhead.
Type Safety: Contract-driven validation for all pattern operations.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
import re
from datetime import datetime, timedelta
from collections import defaultdict

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import PrivacyLevel, UserBehaviorPattern
from src.core.errors import IntelligenceError

logger = get_logger(__name__)


class PatternValidator:
    """Comprehensive pattern validation with privacy and security compliance."""
    
    def __init__(self):
        self.validation_rules: Dict[PrivacyLevel, Dict[str, Any]] = {}
        self.security_patterns: List[str] = []
        self.quality_thresholds: Dict[str, float] = {}
        self._configure_validation_rules()
    
    @require(lambda self, pattern: isinstance(pattern, UserBehaviorPattern))
    def is_valid_for_analysis(
        self,
        pattern: UserBehaviorPattern,
        privacy_level: PrivacyLevel
    ) -> bool:
        """
        Validate behavioral pattern for analysis with privacy compliance.
        
        Performs comprehensive validation including privacy compliance,
        security boundaries, quality thresholds, and analytical utility
        assessment for machine learning applications.
        
        Args:
            pattern: Behavioral pattern for validation
            privacy_level: Privacy protection level requirements
            
        Returns:
            True if pattern meets all validation criteria
            
        Security:
            - Privacy level compliance verification
            - Sensitive data detection and filtering
            - Security boundary validation
        """
        try:
            # Privacy compliance validation
            if not self._validate_privacy_compliance(pattern, privacy_level):
                logger.debug(f"Pattern {pattern.pattern_id} failed privacy compliance")
                return False
            
            # Security validation
            if not self._validate_security_boundaries(pattern):
                logger.debug(f"Pattern {pattern.pattern_id} failed security validation")
                return False
            
            # Quality threshold validation
            if not self._validate_quality_thresholds(pattern):
                logger.debug(f"Pattern {pattern.pattern_id} failed quality thresholds")
                return False
            
            # Analytical utility validation
            if not self._validate_analytical_utility(pattern):
                logger.debug(f"Pattern {pattern.pattern_id} failed analytical utility")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Pattern validation error for {pattern.pattern_id}: {str(e)}")
            return False
    
    @require(lambda self, pattern: isinstance(pattern, UserBehaviorPattern))
    def is_valid_for_learning(
        self,
        pattern: UserBehaviorPattern,
        learning_type: str = "general"
    ) -> bool:
        """
        Validate behavioral pattern for machine learning applications.
        
        Ensures pattern meets quality standards for adaptive learning,
        has sufficient data points for statistical reliability, and
        contains actionable information for automation improvement.
        
        Args:
            pattern: Behavioral pattern for learning validation
            learning_type: Type of learning application
            
        Returns:
            True if pattern is suitable for machine learning
        """
        try:
            # Minimum data requirements for learning
            if pattern.frequency < 3:
                return False
            
            # Confidence threshold for learning
            if pattern.confidence_score < 0.6:
                return False
            
            # Success rate threshold for actionable patterns
            if pattern.success_rate < 0.7:
                return False
            
            # Pattern must have meaningful action sequence
            if len(pattern.action_sequence) == 0:
                return False
            
            # Learning-specific validation
            if learning_type == "automation":
                return self._validate_automation_learning_criteria(pattern)
            elif learning_type == "optimization":
                return self._validate_optimization_learning_criteria(pattern)
            else:
                return True  # General learning requirements already met
            
        except Exception as e:
            logger.warning(f"Learning validation error for {pattern.pattern_id}: {str(e)}")
            return False
    
    def _validate_privacy_compliance(
        self,
        pattern: UserBehaviorPattern,
        privacy_level: PrivacyLevel
    ) -> bool:
        """Validate pattern compliance with privacy level requirements."""
        rules = self.validation_rules.get(privacy_level, {})
        
        # Check for privacy level specific restrictions
        if privacy_level == PrivacyLevel.MAXIMUM:
            # Strict privacy: user_id must be anonymized
            if self._appears_to_be_personal_identifier(pattern.user_id):
                return False
            
            # Action sequences must not contain sensitive information
            if self._contains_sensitive_actions(pattern.action_sequence):
                return False
            
            # Context tags must be sanitized
            if self._contains_sensitive_context_tags(pattern.context_tags):
                return False
        
        elif privacy_level == PrivacyLevel.BALANCED:
            # Balanced privacy: moderate restrictions
            if self._contains_highly_sensitive_data(pattern):
                return False
        
        # Permissive level has minimal restrictions
        return True
    
    def _validate_security_boundaries(self, pattern: UserBehaviorPattern) -> bool:
        """Validate pattern meets security boundary requirements."""
        # Check for dangerous action sequences
        dangerous_actions = {
            'delete_all', 'format_disk', 'rm_rf', 'sudo_password',
            'admin_login', 'root_access', 'system_modify'
        }
        
        for action in pattern.action_sequence:
            if any(dangerous in action.lower() for dangerous in dangerous_actions):
                return False
        
        # Check for security-sensitive context tags
        security_tags = {'admin', 'root', 'sudo', 'password', 'secret'}
        if any(tag in pattern.context_tags for tag in security_tags):
            return False
        
        # Validate pattern doesn't expose system vulnerabilities
        if self._exposes_security_vulnerabilities(pattern):
            return False
        
        return True
    
    def _validate_quality_thresholds(self, pattern: UserBehaviorPattern) -> bool:
        """Validate pattern meets minimum quality thresholds."""
        thresholds = self.quality_thresholds
        
        # Frequency threshold
        if pattern.frequency < thresholds.get('min_frequency', 1):
            return False
        
        # Success rate threshold
        if pattern.success_rate < thresholds.get('min_success_rate', 0.5):
            return False
        
        # Confidence threshold
        if pattern.confidence_score < thresholds.get('min_confidence', 0.5):
            return False
        
        # Completion time reasonableness (not too fast or too slow)
        min_time = thresholds.get('min_completion_time', 0.1)
        max_time = thresholds.get('max_completion_time', 3600.0)  # 1 hour
        if not (min_time <= pattern.average_completion_time <= max_time):
            return False
        
        return True
    
    def _validate_analytical_utility(self, pattern: UserBehaviorPattern) -> bool:
        """Validate pattern has sufficient analytical utility."""
        # Pattern must have meaningful action sequence
        if not pattern.action_sequence or len(pattern.action_sequence) == 0:
            return False
        
        # Pattern must have context information
        if len(pattern.context_tags) == 0:
            return False
        
        # Pattern must be recent enough to be relevant
        if not pattern.is_recent(days=90):  # 90 days recency threshold
            return False
        
        # Pattern should have reasonable reliability score
        if pattern.get_reliability_score() < 0.3:
            return False
        
        return True
    
    def _validate_automation_learning_criteria(self, pattern: UserBehaviorPattern) -> bool:
        """Validate pattern meets criteria for automation learning."""
        # High frequency patterns are better for automation
        if pattern.frequency < 5:
            return False
        
        # Consistent timing is important for automation
        if pattern.get_efficiency_score() < 0.6:
            return False
        
        # Must have actionable sequence
        if len(pattern.action_sequence) < 2:
            return False
        
        return True
    
    def _validate_optimization_learning_criteria(self, pattern: UserBehaviorPattern) -> bool:
        """Validate pattern meets criteria for optimization learning."""
        # Need performance data for optimization
        if pattern.average_completion_time <= 0:
            return False
        
        # Need success/failure data for optimization
        if pattern.success_rate >= 1.0:  # Perfect success gives no optimization insight
            return False
        
        # Need sufficient frequency for optimization analysis
        if pattern.frequency < 3:
            return False
        
        return True
    
    def _appears_to_be_personal_identifier(self, user_id: str) -> bool:
        """Check if user_id appears to contain personal information."""
        # Check for email-like patterns
        if '@' in user_id and '.' in user_id:
            return True
        
        # Check for real names (contains spaces and multiple words)
        words = user_id.split()
        if len(words) >= 2 and all(word.isalpha() for word in words):
            return True
        
        # Check for phone number patterns
        if re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', user_id):
            return True
        
        # If it's a hash or UUID-like, it's probably anonymized
        if len(user_id) >= 16 and all(c in '0123456789abcdefABCDEF-' for c in user_id):
            return False
        
        # Check for patterns that look like real names or personal info
        # Only consider it personal if it actually looks like personal data
        personal_patterns = [
            (r'^[A-Z][a-z]+[A-Z][a-z]+', 0),  # CamelCase names like JohnSmith (case sensitive)
            (r'^[a-z]+\d{2,4}$', re.IGNORECASE),          # username123 patterns
            (r'.*user.*', re.IGNORECASE),                 # Contains "user"
            (r'.*admin.*', re.IGNORECASE),                # Contains "admin" 
            (r'.*test.*', re.IGNORECASE)                  # Contains "test"
        ]
        
        for pattern, flags in personal_patterns:
            if re.match(pattern, user_id, flags):
                return True
        
        # Short random strings like "aaa", "xyz" are probably not personal
        # Only consider very short IDs personal if they have specific patterns
        return False
    
    def _contains_sensitive_actions(self, action_sequence: List[str]) -> bool:
        """Check if action sequence contains sensitive information."""
        sensitive_keywords = {
            'password', 'login', 'secret', 'token', 'key', 'auth',
            'credit', 'ssn', 'social', 'private', 'confidential'
        }
        
        for action in action_sequence:
            action_lower = action.lower()
            # Use word boundaries to avoid false positives like 'keya' containing 'key'
            for keyword in sensitive_keywords:
                # Check for keyword as standalone word or with common separators
                patterns = [
                    f'\\b{keyword}\\b',          # Word boundary match
                    f'{keyword}_',               # keyword_something
                    f'_{keyword}',               # something_keyword  
                    f'{keyword}:',               # keyword:value
                    f'{keyword}=',               # keyword=value
                    f'^{keyword}$'               # Exact match
                ]
                
                for pattern in patterns:
                    if re.search(pattern, action_lower):
                        return True
        
        return False
    
    def _contains_sensitive_context_tags(self, context_tags: Set[str]) -> bool:
        """Check if context tags contain sensitive information."""
        sensitive_tag_patterns = {
            'user:', 'password:', 'secret:', 'token:', 'key:',
            'email:', 'phone:', 'ssn:', 'credit:'
        }
        
        for tag in context_tags:
            tag_lower = tag.lower()
            if any(pattern in tag_lower for pattern in sensitive_tag_patterns):
                return True
        
        return False
    
    def _contains_highly_sensitive_data(self, pattern: UserBehaviorPattern) -> bool:
        """Check for highly sensitive data that should never be in patterns."""
        # Convert pattern to string for comprehensive checking
        pattern_str = f"{pattern.action_sequence} {pattern.context_tags}".lower()
        
        # Highly sensitive patterns
        highly_sensitive = [
            r'password\s*[:=]\s*\w+',
            r'secret\s*[:=]\s*\w+',
            r'token\s*[:=]\s*\w+',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]*\d{4}[\s-]*\d{4}[\s-]*\d{4}\b',  # Credit card
        ]
        
        for pattern_regex in highly_sensitive:
            if re.search(pattern_regex, pattern_str):
                return True
        
        return False
    
    def _exposes_security_vulnerabilities(self, pattern: UserBehaviorPattern) -> bool:
        """Check if pattern exposes security vulnerabilities."""
        # Check for patterns that could be exploited
        vulnerable_patterns = [
            'admin_bypass', 'security_disable', 'firewall_off',
            'sudo_nopasswd', 'root_shell', 'system_override'
        ]
        
        pattern_content = ' '.join(pattern.action_sequence).lower()
        for vulnerable in vulnerable_patterns:
            if vulnerable in pattern_content:
                return True
        
        return False
    
    def _configure_validation_rules(self) -> None:
        """Configure validation rules for different privacy levels."""
        self.validation_rules = {
            PrivacyLevel.MAXIMUM: {
                'require_anonymized_ids': True,
                'filter_sensitive_actions': True,
                'filter_sensitive_context': True,
                'max_retention_days': 7
            },
            PrivacyLevel.BALANCED: {
                'require_anonymized_ids': False,
                'filter_sensitive_actions': True,
                'filter_sensitive_context': False,
                'max_retention_days': 30
            },
            PrivacyLevel.PERMISSIVE: {
                'require_anonymized_ids': False,
                'filter_sensitive_actions': False,
                'filter_sensitive_context': False,
                'max_retention_days': 90
            }
        }
        
        self.quality_thresholds = {
            'min_frequency': 1,
            'min_success_rate': 0.5,
            'min_confidence': 0.5,
            'min_completion_time': 0.1,
            'max_completion_time': 3600.0
        }
    
    def get_validation_summary(self, patterns: List[UserBehaviorPattern]) -> Dict[str, Any]:
        """Get validation summary for a list of patterns."""
        total_patterns = len(patterns)
        if total_patterns == 0:
            return {'total': 0, 'valid': 0, 'invalid': 0, 'validation_rate': 0.0}
        
        valid_count = 0
        invalid_reasons = defaultdict(int)
        
        for pattern in patterns:
            if self.is_valid_for_analysis(pattern, PrivacyLevel.BALANCED):
                valid_count += 1
            else:
                # Determine why it's invalid for reporting
                if pattern.frequency < self.quality_thresholds['min_frequency']:
                    invalid_reasons['low_frequency'] += 1
                elif pattern.success_rate < self.quality_thresholds['min_success_rate']:
                    invalid_reasons['low_success_rate'] += 1
                elif pattern.confidence_score < self.quality_thresholds['min_confidence']:
                    invalid_reasons['low_confidence'] += 1
                else:
                    invalid_reasons['other'] += 1
        
        return {
            'total': total_patterns,
            'valid': valid_count,
            'invalid': total_patterns - valid_count,
            'validation_rate': valid_count / total_patterns,
            'invalid_reasons': dict(invalid_reasons)
        }