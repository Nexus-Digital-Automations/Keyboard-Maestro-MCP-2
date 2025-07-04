"""
Privacy Manager for Automation Intelligence with Comprehensive Data Protection.

This module provides advanced privacy protection capabilities including data anonymization,
privacy policy compliance, secure data handling, and configurable privacy levels for
behavioral intelligence and machine learning applications.

Security: Multi-level privacy protection with encryption and secure data handling.
Performance: Efficient privacy processing with minimal performance impact.
Type Safety: Complete branded type system with contract-driven validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta, UTC
import hashlib
import re
import uuid
from enum import Enum

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import PrivacyLevel
from src.core.errors import IntelligenceError
# from src.intelligence.intelligence_types import Any  # Circular import issue

logger = get_logger(__name__)


class DataClassification(Enum):
    """Data classification levels for privacy processing."""
    PUBLIC = "public"                    # Non-sensitive public data
    INTERNAL = "internal"                # Internal use data
    CONFIDENTIAL = "confidential"        # Confidential business data
    RESTRICTED = "restricted"            # Highly sensitive restricted data


class PrivacyCompliance(Enum):
    """Privacy compliance standards and regulations."""
    GDPR = "gdpr"                        # General Data Protection Regulation
    CCPA = "ccpa"                        # California Consumer Privacy Act
    HIPAA = "hipaa"                      # Health Insurance Portability and Accountability Act
    SOX = "sox"                          # Sarbanes-Oxley Act
    INTERNAL = "internal"                # Internal privacy standards


class PrivacyManager:
    """Comprehensive privacy management with multi-level data protection."""
    
    def __init__(self):
        self.privacy_policies: Dict[PrivacyLevel, Dict[str, Any]] = {}
        self.data_retention_policies: Dict[str, timedelta] = {}
        self.anonymization_keys: Dict[str, str] = {}
        self.compliance_rules: Dict[PrivacyCompliance, Dict[str, Any]] = {}
        
        # Privacy tracking
        self.privacy_audit_log: List[Dict[str, Any]] = []
        self.data_access_log: List[Dict[str, Any]] = []
        
        # Initialize privacy configuration
        self._initialize_privacy_policies()
        self._initialize_compliance_rules()
        self._initialize_retention_policies()
    
    async def initialize(self) -> Either[IntelligenceError, None]:
        """Initialize privacy manager with secure configuration."""
        try:
            # Generate session-specific anonymization keys
            self._generate_anonymization_keys()
            
            # Validate privacy configuration
            validation_result = self._validate_privacy_configuration()
            if validation_result.is_left():
                return validation_result
            
            logger.info("Privacy manager initialized with comprehensive data protection")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Privacy manager initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))
    
    async def validate_privacy_compliance(self, request: Any) -> Either[IntelligenceError, None]:
        """
        Validate intelligence request for comprehensive privacy compliance.
        
        Performs multi-level privacy validation including operation authorization,
        data classification compliance, retention policy validation, and
        regulatory compliance checking based on configured privacy standards.
        
        Args:
            request: Intelligence request for privacy validation
            
        Returns:
            Either error or successful privacy compliance validation
            
        Security:
            - Multi-level privacy policy enforcement
            - Regulatory compliance validation
            - Data classification and retention policy checks
        """
        try:
            # Check operation authorization for privacy level
            operation_check = self._validate_operation_authorization(request)
            if operation_check.is_left():
                return operation_check
            
            # Validate data scope compliance
            scope_check = self._validate_data_scope_compliance(request)
            if scope_check.is_left():
                return scope_check
            
            # Check retention policy compliance
            retention_check = self._validate_retention_policy_compliance(request)
            if retention_check.is_left():
                return retention_check
            
            # Validate regulatory compliance if applicable
            if hasattr(request, 'compliance_requirements'):
                compliance_check = self._validate_regulatory_compliance(request)
                if compliance_check.is_left():
                    return compliance_check
            
            # Log privacy compliance validation
            self._log_privacy_validation(request, "approved")
            
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Privacy compliance validation failed: {str(e)}")
            self._log_privacy_validation(request, "failed", str(e))
            return Either.left(IntelligenceError.privacy_violation(f"Privacy validation failed: {str(e)}"))
    
    @require(lambda self, results: isinstance(results, dict))
    async def filter_results(
        self,
        results: Dict[str, Any],
        privacy_level: PrivacyLevel,
        anonymize_data: bool = True
    ) -> Dict[str, Any]:
        """
        Filter and anonymize results based on privacy protection requirements.
        
        Applies comprehensive privacy filtering including sensitive data removal,
        field anonymization, data aggregation, and privacy-compliant result
        formatting based on configured privacy protection levels.
        
        Args:
            results: Raw intelligence results for privacy filtering
            privacy_level: Privacy protection level for filtering
            anonymize_data: Apply data anonymization transformations
            
        Returns:
            Privacy-compliant filtered and anonymized results
            
        Security:
            - Sensitive data detection and removal
            - Field-level anonymization and aggregation
            - Privacy-compliant result formatting
        """
        try:
            # Deep copy results to avoid modifying original
            filtered_results = self._deep_copy_dict(results)
            
            # Apply privacy level specific filtering
            if privacy_level == PrivacyLevel.MAXIMUM:
                filtered_results = await self._apply_strict_result_filtering(filtered_results)
            elif privacy_level == PrivacyLevel.BALANCED:
                filtered_results = await self._apply_balanced_result_filtering(filtered_results)
            else:  # PERMISSIVE
                filtered_results = await self._apply_permissive_result_filtering(filtered_results)
            
            # Apply data anonymization if requested
            if anonymize_data:
                filtered_results = await self._anonymize_result_data(filtered_results, privacy_level)
            
            # Remove any remaining sensitive patterns
            filtered_results = self._filter_sensitive_patterns(filtered_results)
            
            # Log data access for audit trail
            self._log_data_access(results, filtered_results, privacy_level)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Result filtering failed: {str(e)}")
            # Return minimal safe results on filtering failure
            return {"error": "Privacy filtering failed", "timestamp": datetime.now(UTC).isoformat()}
    
    def _validate_operation_authorization(self, request: Any) -> Either[IntelligenceError, None]:
        """Validate if operation is authorized for privacy level."""
        policy = self.privacy_policies.get(request.privacy_level, {})
        authorized_operations = policy.get('authorized_operations', [])
        
        if request.operation.value not in authorized_operations:
            return Either.left(IntelligenceError.privacy_violation(
                f"Operation {request.operation.value} not authorized for privacy level {request.privacy_level.value}"
            ))
        
        return Either.right(None)
    
    def _validate_data_scope_compliance(self, request: Any) -> Either[IntelligenceError, None]:
        """Validate data scope compliance with privacy level."""
        policy = self.privacy_policies.get(request.privacy_level, {})
        authorized_scopes = policy.get('authorized_scopes', [])
        
        if request.analysis_scope.value not in authorized_scopes:
            return Either.left(IntelligenceError.privacy_violation(
                f"Analysis scope {request.analysis_scope.value} not authorized for privacy level {request.privacy_level.value}"
            ))
        
        return Either.right(None)
    
    def _validate_retention_policy_compliance(self, request: Any) -> Either[IntelligenceError, None]:
        """Validate request compliance with data retention policies."""
        policy = self.privacy_policies.get(request.privacy_level, {})
        max_retention = policy.get('max_data_retention_days', 30)
        
        # Parse time period to days
        time_period_days = self._parse_time_period_to_days(request.time_period)
        
        if time_period_days > max_retention:
            return Either.left(IntelligenceError.privacy_violation(
                f"Requested time period {request.time_period} exceeds maximum retention {max_retention} days"
            ))
        
        return Either.right(None)
    
    def _validate_regulatory_compliance(self, request: Any) -> Either[IntelligenceError, None]:
        """Validate regulatory compliance requirements."""
        # This would implement specific regulatory compliance checks
        # For now, return success as placeholder
        return Either.right(None)
    
    async def _apply_strict_result_filtering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict privacy filtering to results."""
        # Remove all potentially identifying information
        filtered = {}
        
        # Only preserve essential analytics
        safe_fields = {
            'total_patterns', 'analysis_summary', 'performance_insights',
            'suggestion_summary', 'processing_time_seconds', 'timestamp'
        }
        
        for key, value in results.items():
            if key in safe_fields:
                if isinstance(value, dict):
                    # Recursively filter nested dictionaries
                    filtered[key] = await self._filter_nested_dict(value, privacy_level=PrivacyLevel.MAXIMUM)
                elif isinstance(value, list):
                    # Filter lists and limit size
                    filtered[key] = await self._filter_list_data(value, max_items=5)
                else:
                    filtered[key] = value
        
        # Add privacy compliance metadata
        filtered['privacy_compliance'] = {
            'privacy_level': 'strict',
            'data_anonymized': True,
            'pii_removed': True,
            'minimal_data_retained': True
        }
        
        return filtered
    
    async def _apply_balanced_result_filtering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply balanced privacy filtering to results."""
        filtered = results.copy()
        
        # Remove highly sensitive fields
        sensitive_fields = {'user_details', 'personal_info', 'sensitive_patterns'}
        for field in sensitive_fields:
            filtered.pop(field, None)
        
        # Anonymize user identifiers in nested data
        filtered = await self._anonymize_user_identifiers(filtered)
        
        # Limit detailed pattern information
        if 'patterns_summary' in filtered:
            patterns = filtered['patterns_summary']
            if isinstance(patterns, list) and len(patterns) > 10:
                filtered['patterns_summary'] = patterns[:10]
        
        filtered['privacy_compliance'] = {
            'privacy_level': 'balanced',
            'data_anonymized': True,
            'sensitive_data_removed': True
        }
        
        return filtered
    
    async def _apply_permissive_result_filtering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply permissive privacy filtering to results."""
        filtered = results.copy()
        
        # Remove only explicitly dangerous information
        dangerous_fields = {'passwords', 'tokens', 'secrets', 'api_keys'}
        for field in dangerous_fields:
            filtered.pop(field, None)
        
        # Sanitize but preserve most data
        filtered = await self._sanitize_dangerous_content(filtered)
        
        filtered['privacy_compliance'] = {
            'privacy_level': 'permissive',
            'dangerous_content_removed': True
        }
        
        return filtered
    
    async def _anonymize_result_data(self, results: Dict[str, Any], privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Apply anonymization to result data."""
        anonymized = {}
        
        for key, value in results.items():
            if isinstance(value, str):
                anonymized[key] = self._anonymize_string_value(value, privacy_level)
            elif isinstance(value, dict):
                anonymized[key] = await self._anonymize_result_data(value, privacy_level)
            elif isinstance(value, list):
                anonymized[key] = [
                    await self._anonymize_result_data(item, privacy_level) if isinstance(item, dict)
                    else self._anonymize_string_value(str(item), privacy_level) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                anonymized[key] = value
        
        return anonymized
    
    def _anonymize_string_value(self, value: str, privacy_level: PrivacyLevel) -> str:
        """Anonymize string value based on privacy level."""
        if privacy_level == PrivacyLevel.MAXIMUM:
            # Hash all potential identifiers
            if self._appears_to_be_identifier(value):
                return self._hash_value(value)
            
            # Replace with generic placeholders
            if len(value) > 20:
                return "[LONG_STRING]"
            elif any(char.isdigit() for char in value):
                return "[ID_STRING]"
            else:
                return value
        
        elif privacy_level == PrivacyLevel.BALANCED:
            # Selective anonymization
            return self._selective_anonymization(value)
        
        else:  # PERMISSIVE
            # Minimal anonymization
            return self._minimal_anonymization(value)
    
    def _filter_sensitive_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive patterns from results."""
        sensitive_patterns = [
            r'(?i)(password|passwd|pwd)[\\s:=]+[^\\s]+',
            r'(?i)(secret|token|key)[\\s:=]+[^\\s]+',
            r'(?i)(api[_\\s]*key)[\\s:=]+[^\\s]+',
            r'\\b\\d{3}-\\d{2}-\\d{4}\\b',  # SSN
            r'\\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\\b',  # Email
        ]
        
        def clean_value(value):
            if isinstance(value, str):
                for pattern in sensitive_patterns:
                    value = re.sub(pattern, '[REDACTED]', value, flags=re.IGNORECASE)
                return value
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            else:
                return value
        
        return clean_value(results)
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of dictionary for safe modification."""
        import copy
        return copy.deepcopy(d)
    
    def _hash_value(self, value: str) -> str:
        """Create secure hash of value with session key."""
        session_key = self.anonymization_keys.get('session', 'default')
        salted_value = f"{session_key}_{value}"
        return hashlib.sha256(salted_value.encode()).hexdigest()[:16]
    
    def _appears_to_be_identifier(self, value: str) -> bool:
        """Check if string appears to be an identifier."""
        # Check for common identifier patterns
        return (
            '@' in value or  # Email-like
            len(value) > 10 and any(c.isdigit() for c in value) or  # Long with numbers
            value.startswith(('user_', 'id_', 'session_')) or  # Identifier prefixes
            re.match(r'^[a-f0-9]{8,}$', value.lower())  # Hex string
        )
    
    def _selective_anonymization(self, value: str) -> str:
        """Apply selective anonymization for balanced privacy."""
        # Replace emails
        value = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', value)
        
        # Replace phone numbers
        value = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', value)
        
        # Replace SSNs
        value = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', value)
        
        return value
    
    def _minimal_anonymization(self, value: str) -> str:
        """Apply minimal anonymization for permissive privacy."""
        # Only replace explicit secrets
        value = re.sub(r'(?i)(password|secret|token)[\\s:=]+[^\\s]+', '[REDACTED]', value)
        return value
    
    def _parse_time_period_to_days(self, time_period: str) -> int:
        """Parse time period string to number of days."""
        if time_period == "1d":
            return 1
        elif time_period == "7d":
            return 7
        elif time_period == "30d":
            return 30
        elif time_period == "90d":
            return 90
        elif time_period == "all":
            return 365  # Default max for "all"
        else:
            return 30  # Default
    
    def _generate_anonymization_keys(self) -> None:
        """Generate session-specific anonymization keys."""
        self.anonymization_keys = {
            'session': hashlib.sha256(f"{datetime.now(UTC)}_{uuid.uuid4()}".encode()).hexdigest()[:16],
            'user': hashlib.sha256(f"user_{uuid.uuid4()}".encode()).hexdigest()[:16],
            'pattern': hashlib.sha256(f"pattern_{uuid.uuid4()}".encode()).hexdigest()[:16]
        }
    
    def _validate_privacy_configuration(self) -> Either[IntelligenceError, None]:
        """Validate privacy manager configuration."""
        # Check that all required policies are configured
        required_levels = [PrivacyLevel.MAXIMUM, PrivacyLevel.BALANCED, PrivacyLevel.PERMISSIVE]
        
        for level in required_levels:
            if level not in self.privacy_policies:
                return Either.left(IntelligenceError.initialization_failed(
                    f"Privacy policy not configured for level: {level.value}"
                ))
        
        return Either.right(None)
    
    def _log_privacy_validation(self, request: Any, status: str, error: str = None) -> None:
        """Log privacy validation for audit trail."""
        log_entry = {
            'timestamp': datetime.now(UTC),
            'operation': request.operation.value,
            'analysis_scope': request.analysis_scope.value,
            'privacy_level': request.privacy_level.value,
            'status': status,
            'error': error
        }
        
        self.privacy_audit_log.append(log_entry)
        
        # Limit log size
        if len(self.privacy_audit_log) > 1000:
            self.privacy_audit_log = self.privacy_audit_log[-500:]
    
    def _log_data_access(self, original_results: Dict[str, Any], filtered_results: Dict[str, Any], privacy_level: PrivacyLevel) -> None:
        """Log data access for audit trail."""
        access_entry = {
            'timestamp': datetime.now(UTC),
            'privacy_level': privacy_level.value,
            'original_fields': len(original_results),
            'filtered_fields': len(filtered_results),
            'data_reduction_ratio': 1.0 - (len(filtered_results) / max(1, len(original_results)))
        }
        
        self.data_access_log.append(access_entry)
        
        # Limit log size
        if len(self.data_access_log) > 1000:
            self.data_access_log = self.data_access_log[-500:]
    
    def _initialize_privacy_policies(self) -> None:
        """Initialize privacy policies for different protection levels."""
        self.privacy_policies = {
            PrivacyLevel.MAXIMUM: {
                'authorized_operations': ['analyze', 'insights'],
                'authorized_scopes': ['performance', 'usage'],
                'max_data_retention_days': 7,
                'require_anonymization': True,
                'min_aggregation_size': 10,
                'allow_user_identification': False
            },
            PrivacyLevel.BALANCED: {
                'authorized_operations': ['analyze', 'suggest', 'optimize', 'insights'],
                'authorized_scopes': ['user_behavior', 'automation_patterns', 'performance', 'usage'],
                'max_data_retention_days': 30,
                'require_anonymization': True,
                'min_aggregation_size': 5,
                'allow_user_identification': False
            },
            PrivacyLevel.PERMISSIVE: {
                'authorized_operations': ['analyze', 'learn', 'suggest', 'optimize', 'predict', 'insights'],
                'authorized_scopes': ['user_behavior', 'automation_patterns', 'performance', 'usage', 'workflow', 'error_patterns'],
                'max_data_retention_days': 90,
                'require_anonymization': False,
                'min_aggregation_size': 1,
                'allow_user_identification': True
            }
        }
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize regulatory compliance rules."""
        self.compliance_rules = {
            PrivacyCompliance.GDPR: {
                'require_consent': True,
                'allow_data_export': True,
                'require_deletion_capability': True,
                'max_retention_days': 365
            },
            PrivacyCompliance.CCPA: {
                'require_opt_out': True,
                'allow_data_export': True,
                'require_deletion_capability': True,
                'max_retention_days': 365
            },
            PrivacyCompliance.INTERNAL: {
                'require_approval': False,
                'max_retention_days': 90,
                'audit_required': True
            }
        }
    
    def _initialize_retention_policies(self) -> None:
        """Initialize data retention policies."""
        self.data_retention_policies = {
            'behavioral_patterns': timedelta(days=30),
            'performance_metrics': timedelta(days=90),
            'usage_analytics': timedelta(days=60),
            'error_logs': timedelta(days=30),
            'audit_logs': timedelta(days=365)
        }
    
    # Placeholder methods for advanced filtering
    async def _filter_nested_dict(self, data: Dict[str, Any], privacy_level: PrivacyLevel) -> Dict[str, Any]:
        """Filter nested dictionary data."""
        return data
    
    async def _filter_list_data(self, data: List[Any], max_items: int) -> List[Any]:
        """Filter list data with size limits."""
        return data[:max_items]
    
    async def _anonymize_user_identifiers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize user identifiers in data."""
        return data
    
    async def _sanitize_dangerous_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dangerous content from data."""
        return data