"""
Privacy-Preserving Data Anonymization for Behavioral Intelligence.

This module provides comprehensive data anonymization capabilities with configurable
privacy protection levels, ensuring secure behavioral data processing while
maintaining analytical utility for machine learning and pattern recognition.

Security: Multiple anonymization techniques with privacy level compliance.
Performance: Efficient anonymization with minimal processing overhead.
Type Safety: Contract-driven validation for all anonymization operations.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
import hashlib
import re
from datetime import datetime, UTC
import uuid

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.logging import get_logger
from src.core.suggestion_system import PrivacyLevel
from src.core.errors import IntelligenceError

logger = get_logger(__name__)


class DataAnonymizer:
    """Comprehensive data anonymization with configurable privacy protection."""
    
    def __init__(self):
        self.anonymization_rules: Dict[PrivacyLevel, Dict[str, Any]] = {}
        self.sensitive_patterns: List[str] = []
        self.field_mappings: Dict[str, str] = {}
        self._salt = self._generate_session_salt()
        
        # Initialize patterns immediately for testing and basic usage
        self._configure_anonymization_rules()
        self._configure_sensitive_patterns()
    
    async def initialize(self, privacy_level: PrivacyLevel) -> Either[IntelligenceError, None]:
        """Initialize anonymizer with privacy-specific configuration."""
        try:
            self._configure_anonymization_rules()
            self._configure_sensitive_patterns()
            
            logger.info(f"Data anonymizer initialized for privacy level: {privacy_level.value}")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Data anonymizer initialization failed: {str(e)}")
            return Either.left(IntelligenceError.initialization_failed(str(e)))
    
    @require(lambda self, data: isinstance(data, list))
    async def anonymize_behavior_data(
        self,
        data: List[Dict[str, Any]],
        privacy_level: PrivacyLevel
    ) -> List[Dict[str, Any]]:
        """
        Anonymize behavioral data based on privacy protection level.
        
        Applies comprehensive anonymization techniques including field removal,
        data hashing, pattern obfuscation, and temporal anonymization while
        preserving analytical utility for behavioral pattern recognition.
        
        Args:
            data: Raw behavioral data for anonymization
            privacy_level: Privacy protection level configuration
            
        Returns:
            Anonymized behavioral data maintaining analytical utility
            
        Security:
            - Multi-level anonymization based on privacy requirements
            - Sensitive data detection and secure removal
            - Cryptographic hashing for identifier protection
        """
        try:
            anonymized_data = []
            rules = self.anonymization_rules.get(privacy_level, {})
            
            for record in data:
                anonymized_record = await self._anonymize_single_record(record, privacy_level, rules)
                if anonymized_record:  # Only include valid anonymized records
                    anonymized_data.append(anonymized_record)
            
            logger.debug(f"Anonymized {len(data)} records to {len(anonymized_data)} records "
                        f"at privacy level: {privacy_level.value}")
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Behavioral data anonymization failed: {str(e)}")
            return []  # Return empty list on anonymization failure for safety
    
    async def _anonymize_single_record(
        self,
        record: Dict[str, Any],
        privacy_level: PrivacyLevel,
        rules: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Anonymize a single behavioral data record."""
        try:
            anonymized = record.copy()
            
            # Apply privacy level specific anonymization
            if privacy_level == PrivacyLevel.MAXIMUM:
                anonymized = await self._apply_strict_anonymization(anonymized)
            elif privacy_level == PrivacyLevel.BALANCED:
                anonymized = await self._apply_balanced_anonymization(anonymized)
            else:  # PERMISSIVE
                anonymized = await self._apply_permissive_anonymization(anonymized)
            
            # Apply universal sensitive data filtering
            anonymized = self._filter_sensitive_data(anonymized)
            
            # Validate anonymized record
            if self._is_valid_anonymized_record(anonymized):
                return anonymized
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Single record anonymization failed: {str(e)}")
            return None
    
    async def _apply_strict_anonymization(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict anonymization - maximum privacy protection."""
        anonymized = {}
        
        # Remove all potentially identifying fields
        identifying_fields = {
            'user_id', 'session_id', 'device_id', 'ip_address', 'username',
            'email', 'phone', 'location', 'device_name', 'mac_address'
        }
        
        for key, value in record.items():
            if key.lower() in identifying_fields:
                continue  # Remove identifying fields entirely
            
            # Hash sensitive string values
            if key in ['tool_name', 'action', 'file_path']:
                anonymized[key] = self._hash_value(str(value))
            
            # Preserve essential analytics data with anonymization for timestamps
            elif key in ['timestamp', 'success', 'execution_time', 'error_count']:
                if key == 'timestamp' and isinstance(value, datetime):
                    # Anonymize timestamp to remove precise timing information
                    anonymized[key] = value.strftime('%Y-%m-%d')  # Date only, no time
                else:
                    anonymized[key] = value
            
            # Anonymize other string fields
            elif isinstance(value, str) and not self._is_safe_value(value):
                anonymized[key] = self._hash_value(value)
            
            # Preserve numeric and boolean values
            elif isinstance(value, (int, float, bool)):
                anonymized[key] = value
        
        # Add anonymized identifiers for pattern correlation
        anonymized['anonymous_session'] = self._generate_anonymous_id('session')
        anonymized['anonymous_user'] = self._generate_anonymous_id('user')
        
        return anonymized
    
    async def _apply_balanced_anonymization(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply balanced anonymization - moderate privacy with analytical utility."""
        anonymized = record.copy()
        
        # Hash direct identifiers but keep some context
        if 'user_id' in anonymized:
            anonymized['user_id'] = self._hash_value(anonymized['user_id'])
        
        if 'session_id' in anonymized:
            anonymized['session_id'] = self._hash_value(anonymized['session_id'])
        
        # Remove highly sensitive fields
        sensitive_fields = {'email', 'phone', 'ip_address', 'mac_address'}
        for field in sensitive_fields:
            anonymized.pop(field, None)
        
        # Partially anonymize file paths (keep structure, remove personal info)
        if 'file_path' in anonymized:
            anonymized['file_path'] = self._anonymize_file_path(anonymized['file_path'])
        
        # Keep tool names and actions for pattern analysis but sanitize
        for field in ['tool_name', 'action']:
            if field in anonymized:
                anonymized[field] = self._sanitize_for_analysis(anonymized[field])
        
        return anonymized
    
    async def _apply_permissive_anonymization(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply permissive anonymization - minimal privacy for enhanced learning."""
        anonymized = record.copy()
        
        # Remove only explicitly sensitive fields
        highly_sensitive = {'password', 'secret', 'token', 'api_key', 'credit_card'}
        for field in list(anonymized.keys()):
            if any(sensitive in field.lower() for sensitive in highly_sensitive):
                del anonymized[field]
        
        # Hash only direct personal identifiers
        if 'email' in anonymized:
            anonymized['email'] = self._hash_value(anonymized['email'])
        
        if 'phone' in anonymized:
            anonymized['phone'] = self._hash_value(anonymized['phone'])
        
        # Keep most data for comprehensive analysis
        return anonymized
    
    def _filter_sensitive_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply universal sensitive data filtering regardless of privacy level."""
        filtered = {}
        
        for key, value in record.items():
            # Check if value contains sensitive patterns
            if isinstance(value, str) and self._contains_sensitive_pattern(value):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def _contains_sensitive_pattern(self, text: str) -> bool:
        """Check if text contains sensitive patterns."""
        text_lower = text.lower()
        
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _hash_value(self, value: str) -> str:
        """Create secure hash of value with session salt."""
        salted_value = f"{self._salt}_{value}"
        return hashlib.sha256(salted_value.encode()).hexdigest()[:16]
    
    def _generate_anonymous_id(self, prefix: str) -> str:
        """Generate anonymous identifier for correlation."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _generate_session_salt(self) -> str:
        """Generate session-specific salt for hashing."""
        return hashlib.sha256(f"{datetime.now(UTC).isoformat()}_{uuid.uuid4()}".encode()).hexdigest()[:16]
    
    def _is_safe_value(self, value: str) -> bool:
        """Check if string value is safe to preserve without anonymization."""
        safe_values = {
            'true', 'false', 'success', 'error', 'complete', 'pending',
            'start', 'end', 'click', 'type', 'scroll', 'open', 'close'
        }
        return value.lower() in safe_values
    
    def _anonymize_file_path(self, file_path: str) -> str:
        """Anonymize file path while preserving structural information."""
        # Replace user-specific parts while keeping file structure
        anonymized = re.sub(r'/Users/[^/]+', '/Users/[USER]', file_path)
        anonymized = re.sub(r'\\Users\\[^\\]+', '\\Users\\[USER]', anonymized)
        
        # Replace personal document names
        anonymized = re.sub(r'/Documents/[^/]+', '/Documents/[DOC]', anonymized)
        anonymized = re.sub(r'\\Documents\\[^\\]+', '\\Documents\\[DOC]', anonymized)
        
        return anonymized
    
    def _sanitize_for_analysis(self, value: str) -> str:
        """Sanitize value for analytical utility while removing sensitive info."""
        # Remove potential personal information but keep analytical value
        sanitized = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', value)
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', sanitized)
        
        return sanitized
    
    def _anonymize_string_value(self, value: str, privacy_level: PrivacyLevel) -> str:
        """Anonymize a string value based on privacy level."""
        if privacy_level == PrivacyLevel.MAXIMUM:
            # Hash all non-safe values for maximum privacy
            if self._is_safe_value(value):
                return value
            else:
                return self._hash_value(value)
        elif privacy_level == PrivacyLevel.BALANCED:
            # Apply moderate anonymization
            if self._contains_sensitive_patterns(value):
                return self._hash_value(value)
            else:
                return value
        else:  # PERMISSIVE
            # Minimal anonymization - only highly sensitive patterns
            if any(pattern in value.lower() for pattern in ['password', 'secret', 'token']):
                return self._hash_value(value)
            else:
                return value
    
    def _contains_sensitive_patterns(self, value: str) -> bool:
        """Check if value contains sensitive patterns."""
        sensitive_keywords = [
            'password', 'secret', 'token', 'key', 'auth',
            'email', 'phone', 'ssn', 'social security',
            'credit card', 'private', 'confidential'
        ]
        value_lower = value.lower()
        return any(keyword in value_lower for keyword in sensitive_keywords)
    
    def _appears_to_be_identifier(self, value: str) -> bool:
        """Check if value appears to be a personal identifier."""
        # Check for email-like patterns
        if '@' in value and '.' in value:
            return True
        
        # Check for real names (contains spaces and multiple words)
        words = value.split()
        if len(words) >= 2 and all(word.isalpha() for word in words):
            return True
        
        # Check for phone number patterns
        if re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', value):
            return True
        
        # Check for ID-like patterns (long alphanumeric strings)
        if len(value) > 8 and value.isalnum():
            return True
        
        # Check for username patterns
        if any(pattern in value.lower() for pattern in ['user', 'admin', 'test']):
            return True
        
        return False
    
    def _is_valid_anonymized_record(self, record: Dict[str, Any]) -> bool:
        """Validate that anonymized record meets privacy requirements."""
        # Check that essential fields are present for analysis
        required_fields = {'timestamp'}
        
        # Must have at least timestamp for temporal analysis
        if not any(field in record for field in required_fields):
            return False
        
        # Check that no obvious sensitive data remains
        record_str = str(record).lower()
        dangerous_patterns = ['password', 'secret', '@gmail.com', 'ssn:', 'credit_card']
        
        for pattern in dangerous_patterns:
            if pattern in record_str:
                logger.warning(f"Potential sensitive data found in anonymized record: {pattern}")
                return False
        
        return True
    
    def _configure_anonymization_rules(self) -> None:
        """Configure anonymization rules for different privacy levels."""
        self.anonymization_rules = {
            PrivacyLevel.MAXIMUM: {
                'remove_fields': [
                    'user_id', 'session_id', 'device_id', 'ip_address', 'username',
                    'email', 'phone', 'location', 'device_name', 'mac_address'
                ],
                'hash_fields': ['tool_name', 'action', 'file_path'],
                'preserve_fields': ['timestamp', 'success', 'execution_time', 'error_count']
            },
            PrivacyLevel.BALANCED: {
                'hash_fields': ['user_id', 'session_id', 'email', 'phone'],
                'remove_fields': ['ip_address', 'mac_address'],
                'sanitize_fields': ['tool_name', 'action', 'file_path']
            },
            PrivacyLevel.PERMISSIVE: {
                'remove_fields': ['password', 'secret', 'token', 'api_key'],
                'hash_fields': ['email', 'phone'],
                'preserve_most': True
            }
        }
    
    def _configure_sensitive_patterns(self) -> None:
        """Configure patterns for sensitive data detection."""
        self.sensitive_patterns = [
            r'(?i)(password|passwd|pwd)[\s:=]+[^\s]+',
            r'(?i)(secret|token|key)[\s:=]+[^\s]+',
            r'(?i)(api[_\s]*key)[\s:=]+[^\s]+',
            r'(?i)(credit[_\s]*card|ssn|social[_\s]*security)',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
            r'\b\d{4}[\s-]*\d{4}[\s-]*\d{4}[\s-]*\d{4}\b',  # Credit card format
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # Email format
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone format
        ]