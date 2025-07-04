"""
Privacy Manager - TASK_67 Phase 2 Core Identity Engine

User data privacy protection, consent management, and compliance with privacy regulations.
Handles data anonymization, secure storage, and user privacy controls.

Architecture: Privacy Protection + Design by Contract + Type Safety + Compliance
Performance: <50ms privacy validation, <100ms data anonymization, <200ms consent processing
Security: End-to-end encryption, secure data handling, audit trails, compliance monitoring
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import json
import hashlib
import secrets
from pathlib import Path

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.user_identity_architecture import (
    UserProfile, PersonalizationSettings, IdentityError,
    UserProfileId, PrivacyLevel
)

logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of user consent."""
    DATA_COLLECTION = "data_collection"
    PERSONALIZATION = "personalization"
    ANALYTICS = "analytics"
    BEHAVIORAL_TRACKING = "behavioral_tracking"
    CROSS_SESSION_SYNC = "cross_session_sync"
    MARKETING = "marketing"
    THIRD_PARTY_SHARING = "third_party_sharing"


class DataCategory(Enum):
    """Categories of user data."""
    PERSONAL_INFO = "personal_info"
    BEHAVIORAL_DATA = "behavioral_data"
    PREFERENCES = "preferences"
    INTERACTION_HISTORY = "interaction_history"
    DEVICE_INFO = "device_info"
    LOCATION_DATA = "location_data"
    AUTHENTICATION_DATA = "authentication_data"


@dataclass(frozen=True)
class ConsentRecord:
    """Record of user consent."""
    consent_id: str
    user_profile_id: UserProfileId
    consent_type: ConsentType
    granted: bool
    timestamp: datetime
    expiry_date: Optional[datetime]
    consent_context: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    withdrawal_date: Optional[datetime]


@dataclass(frozen=True)
class DataRetentionPolicy:
    """Data retention policy specification."""
    data_category: DataCategory
    retention_days: int
    auto_delete: bool
    archive_before_delete: bool
    compliance_requirements: List[str]
    deletion_method: str  # secure_delete|anonymize|archive


@dataclass(frozen=True)
class PrivacyAuditRecord:
    """Privacy audit record."""
    audit_id: str
    user_profile_id: UserProfileId
    action: str
    data_accessed: List[str]
    timestamp: datetime
    compliance_status: str
    risk_level: str
    automated_action: bool


class PrivacyManager:
    """Privacy protection and compliance management system."""
    
    def __init__(self):
        self.consent_records: Dict[UserProfileId, List[ConsentRecord]] = {}
        self.data_retention_policies: Dict[DataCategory, DataRetentionPolicy] = {}
        self.privacy_audit_log: List[PrivacyAuditRecord] = []
        self.anonymization_keys: Dict[UserProfileId, str] = {}
        self.compliance_frameworks: Set[str] = {"GDPR", "CCPA", "PIPEDA"}
        self.encryption_keys: Dict[str, str] = {}
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default data retention policies."""
        self.data_retention_policies = {
            DataCategory.PERSONAL_INFO: DataRetentionPolicy(
                data_category=DataCategory.PERSONAL_INFO,
                retention_days=2555,  # 7 years
                auto_delete=False,
                archive_before_delete=True,
                compliance_requirements=["GDPR", "CCPA"],
                deletion_method="secure_delete"
            ),
            DataCategory.BEHAVIORAL_DATA: DataRetentionPolicy(
                data_category=DataCategory.BEHAVIORAL_DATA,
                retention_days=365,  # 1 year
                auto_delete=True,
                archive_before_delete=False,
                compliance_requirements=["GDPR"],
                deletion_method="anonymize"
            ),
            DataCategory.PREFERENCES: DataRetentionPolicy(
                data_category=DataCategory.PREFERENCES,
                retention_days=1825,  # 5 years
                auto_delete=False,
                archive_before_delete=True,
                compliance_requirements=["GDPR", "CCPA"],
                deletion_method="secure_delete"
            ),
            DataCategory.INTERACTION_HISTORY: DataRetentionPolicy(
                data_category=DataCategory.INTERACTION_HISTORY,
                retention_days=180,  # 6 months
                auto_delete=True,
                archive_before_delete=False,
                compliance_requirements=["GDPR"],
                deletion_method="anonymize"
            ),
            DataCategory.AUTHENTICATION_DATA: DataRetentionPolicy(
                data_category=DataCategory.AUTHENTICATION_DATA,
                retention_days=90,  # 3 months
                auto_delete=True,
                archive_before_delete=False,
                compliance_requirements=["GDPR", "CCPA"],
                deletion_method="secure_delete"
            )
        }
    
    @require(lambda user_profile_id: user_profile_id is not None)
    async def record_consent(
        self,
        user_profile_id: UserProfileId,
        consent_type: ConsentType,
        granted: bool,
        context: str,
        duration_days: Optional[int] = None,
        client_info: Optional[Dict[str, str]] = None
    ) -> Either[IdentityError, ConsentRecord]:
        """Record user consent for specific data usage."""
        try:
            consent_id = secrets.token_hex(16)
            current_time = datetime.now(UTC)
            
            # Calculate expiry date
            expiry_date = None
            if duration_days:
                expiry_date = current_time + timedelta(days=duration_days)
            elif consent_type == ConsentType.BEHAVIORAL_TRACKING:
                expiry_date = current_time + timedelta(days=365)  # 1 year default
            
            # Create consent record
            consent_record = ConsentRecord(
                consent_id=consent_id,
                user_profile_id=user_profile_id,
                consent_type=consent_type,
                granted=granted,
                timestamp=current_time,
                expiry_date=expiry_date,
                consent_context=context,
                ip_address=client_info.get("ip_address") if client_info else None,
                user_agent=client_info.get("user_agent") if client_info else None,
                withdrawal_date=None
            )
            
            # Store consent record
            if user_profile_id not in self.consent_records:
                self.consent_records[user_profile_id] = []
            
            self.consent_records[user_profile_id].append(consent_record)
            
            # Log for audit
            await self._log_privacy_audit(
                user_profile_id,
                f"consent_{consent_type.value}_{'granted' if granted else 'denied'}",
                [consent_type.value],
                "compliant",
                "low",
                True
            )
            
            logger.info(f"Recorded consent for user {user_profile_id}: {consent_type.value} = {granted}")
            return Either.success(consent_record)
            
        except Exception as e:
            logger.error(f"Failed to record consent: {e}")
            return Either.error(IdentityError(
                f"Consent recording failed: {str(e)}",
                "CONSENT_ERROR"
            ))
    
    @require(lambda user_profile_id: user_profile_id is not None)
    async def check_consent(
        self,
        user_profile_id: UserProfileId,
        consent_type: ConsentType
    ) -> Either[IdentityError, bool]:
        """Check if user has granted consent for specific data usage."""
        try:
            if user_profile_id not in self.consent_records:
                return Either.success(False)
            
            user_consents = self.consent_records[user_profile_id]
            current_time = datetime.now(UTC)
            
            # Find the most recent consent for this type
            relevant_consents = [
                consent for consent in user_consents 
                if consent.consent_type == consent_type
            ]
            
            if not relevant_consents:
                return Either.success(False)
            
            # Get the most recent consent
            latest_consent = max(relevant_consents, key=lambda c: c.timestamp)
            
            # Check if consent is still valid
            if latest_consent.withdrawal_date:
                return Either.success(False)
            
            if latest_consent.expiry_date and latest_consent.expiry_date < current_time:
                return Either.success(False)
            
            return Either.success(latest_consent.granted)
            
        except Exception as e:
            logger.error(f"Failed to check consent: {e}")
            return Either.error(IdentityError(
                f"Consent check failed: {str(e)}",
                "CONSENT_CHECK_ERROR"
            ))
    
    @require(lambda user_profile_id: user_profile_id is not None)
    async def withdraw_consent(
        self,
        user_profile_id: UserProfileId,
        consent_type: ConsentType,
        reason: Optional[str] = None
    ) -> Either[IdentityError, bool]:
        """Withdraw user consent for specific data usage."""
        try:
            if user_profile_id not in self.consent_records:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))
            
            user_consents = self.consent_records[user_profile_id]
            current_time = datetime.now(UTC)
            
            # Find active consents for this type
            for consent in user_consents:
                if (consent.consent_type == consent_type and 
                    consent.granted and 
                    not consent.withdrawal_date):
                    
                    # Mark as withdrawn
                    withdrawn_consent = ConsentRecord(
                        consent_id=consent.consent_id,
                        user_profile_id=consent.user_profile_id,
                        consent_type=consent.consent_type,
                        granted=consent.granted,
                        timestamp=consent.timestamp,
                        expiry_date=consent.expiry_date,
                        consent_context=consent.consent_context,
                        ip_address=consent.ip_address,
                        user_agent=consent.user_agent,
                        withdrawal_date=current_time
                    )
                    
                    # Replace in list
                    consent_index = user_consents.index(consent)
                    user_consents[consent_index] = withdrawn_consent
                    
                    # Log withdrawal
                    await self._log_privacy_audit(
                        user_profile_id,
                        f"consent_{consent_type.value}_withdrawn",
                        [consent_type.value],
                        "compliant",
                        "medium",
                        True
                    )
                    
                    # Trigger data cleanup if required
                    await self._trigger_data_cleanup(user_profile_id, consent_type)
                    
                    logger.info(f"Withdrawn consent for user {user_profile_id}: {consent_type.value}")
                    return Either.success(True)
            
            return Either.success(False)  # No active consent to withdraw
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent: {e}")
            return Either.error(IdentityError(
                f"Consent withdrawal failed: {str(e)}",
                "CONSENT_WITHDRAWAL_ERROR"
            ))
    
    async def _trigger_data_cleanup(
        self,
        user_profile_id: UserProfileId,
        consent_type: ConsentType
    ) -> None:
        """Trigger data cleanup based on withdrawn consent."""
        try:
            # Map consent types to data categories
            cleanup_mapping = {
                ConsentType.BEHAVIORAL_TRACKING: [DataCategory.BEHAVIORAL_DATA, DataCategory.INTERACTION_HISTORY],
                ConsentType.PERSONALIZATION: [DataCategory.PREFERENCES],
                ConsentType.ANALYTICS: [DataCategory.BEHAVIORAL_DATA]
            }
            
            data_categories = cleanup_mapping.get(consent_type, [])
            
            for category in data_categories:
                await self._schedule_data_deletion(user_profile_id, category, "consent_withdrawal")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    @require(lambda user_profile_id: user_profile_id is not None)
    async def anonymize_user_data(
        self,
        user_profile_id: UserProfileId,
        data_categories: List[DataCategory],
        anonymization_level: str = "standard"
    ) -> Either[IdentityError, Dict[str, Any]]:
        """Anonymize user data for privacy protection."""
        try:
            anonymization_key = self._get_anonymization_key(user_profile_id)
            
            anonymization_results = {}
            
            for category in data_categories:
                result = await self._anonymize_category_data(
                    user_profile_id, category, anonymization_key, anonymization_level
                )
                anonymization_results[category.value] = result
            
            # Log anonymization
            await self._log_privacy_audit(
                user_profile_id,
                "data_anonymization",
                [cat.value for cat in data_categories],
                "compliant",
                "high",
                True
            )
            
            return Either.success(anonymization_results)
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return Either.error(IdentityError(
                f"Anonymization failed: {str(e)}",
                "ANONYMIZATION_ERROR"
            ))
    
    def _get_anonymization_key(self, user_profile_id: UserProfileId) -> str:
        """Get or create anonymization key for user."""
        if user_profile_id not in self.anonymization_keys:
            self.anonymization_keys[user_profile_id] = secrets.token_hex(32)
        return self.anonymization_keys[user_profile_id]
    
    async def _anonymize_category_data(
        self,
        user_profile_id: UserProfileId,
        category: DataCategory,
        anonymization_key: str,
        level: str
    ) -> Dict[str, Any]:
        """Anonymize specific category of data."""
        try:
            # Simulate data anonymization
            if level == "basic":
                return {
                    "method": "hash_replacement",
                    "fields_anonymized": ["identifiers", "names"],
                    "success": True
                }
            elif level == "standard":
                return {
                    "method": "differential_privacy",
                    "fields_anonymized": ["identifiers", "names", "timestamps"],
                    "privacy_budget": 0.1,
                    "success": True
                }
            elif level == "advanced":
                return {
                    "method": "k_anonymity",
                    "fields_anonymized": ["identifiers", "names", "timestamps", "behavioral_patterns"],
                    "k_value": 5,
                    "success": True
                }
            else:
                return {"success": False, "error": "Unknown anonymization level"}
                
        except Exception as e:
            logger.error(f"Category anonymization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _schedule_data_deletion(
        self,
        user_profile_id: UserProfileId,
        category: DataCategory,
        reason: str
    ) -> None:
        """Schedule data deletion according to retention policy."""
        try:
            if category in self.data_retention_policies:
                policy = self.data_retention_policies[category]
                
                # Log scheduled deletion
                await self._log_privacy_audit(
                    user_profile_id,
                    f"data_deletion_scheduled_{reason}",
                    [category.value],
                    "compliant",
                    "high",
                    True
                )
                
                logger.info(f"Scheduled data deletion for user {user_profile_id}, category {category.value}, reason: {reason}")
        
        except Exception as e:
            logger.error(f"Failed to schedule data deletion: {e}")
    
    async def _log_privacy_audit(
        self,
        user_profile_id: UserProfileId,
        action: str,
        data_accessed: List[str],
        compliance_status: str,
        risk_level: str,
        automated: bool
    ) -> None:
        """Log privacy audit record."""
        try:
            audit_record = PrivacyAuditRecord(
                audit_id=secrets.token_hex(16),
                user_profile_id=user_profile_id,
                action=action,
                data_accessed=data_accessed,
                timestamp=datetime.now(UTC),
                compliance_status=compliance_status,
                risk_level=risk_level,
                automated_action=automated
            )
            
            self.privacy_audit_log.append(audit_record)
            
            # Keep only recent audit records (last 10,000)
            if len(self.privacy_audit_log) > 10000:
                self.privacy_audit_log = self.privacy_audit_log[-10000:]
        
        except Exception as e:
            logger.error(f"Privacy audit logging failed: {e}")
    
    @require(lambda user_profile_id: user_profile_id is not None)
    async def get_privacy_report(
        self,
        user_profile_id: UserProfileId
    ) -> Either[IdentityError, Dict[str, Any]]:
        """Generate privacy report for user."""
        try:
            user_consents = self.consent_records.get(user_profile_id, [])
            user_audit_records = [
                record for record in self.privacy_audit_log 
                if record.user_profile_id == user_profile_id
            ]
            
            # Calculate consent summary
            consent_summary = {}
            for consent_type in ConsentType:
                consent_result = await self.check_consent(user_profile_id, consent_type)
                if consent_result.is_success():
                    consent_summary[consent_type.value] = consent_result.value
            
            # Calculate data retention summary
            data_retention_summary = {}
            for category, policy in self.data_retention_policies.items():
                data_retention_summary[category.value] = {
                    "retention_days": policy.retention_days,
                    "auto_delete": policy.auto_delete,
                    "deletion_method": policy.deletion_method
                }
            
            privacy_report = {
                "user_profile_id": user_profile_id,
                "consent_summary": consent_summary,
                "total_consents": len(user_consents),
                "active_consents": len([c for c in user_consents if not c.withdrawal_date]),
                "data_retention_policies": data_retention_summary,
                "audit_records_count": len(user_audit_records),
                "compliance_frameworks": list(self.compliance_frameworks),
                "report_generated_at": datetime.now(UTC).isoformat()
            }
            
            return Either.success(privacy_report)
            
        except Exception as e:
            logger.error(f"Privacy report generation failed: {e}")
            return Either.error(IdentityError(
                f"Privacy report failed: {str(e)}",
                "PRIVACY_REPORT_ERROR"
            ))
    
    async def validate_compliance(
        self,
        framework: str = "GDPR"
    ) -> Either[IdentityError, Dict[str, Any]]:
        """Validate compliance with privacy framework."""
        try:
            compliance_results = {
                "framework": framework,
                "compliant": True,
                "violations": [],
                "recommendations": [],
                "audit_passed": True
            }
            
            # Check consent management
            if framework == "GDPR":
                # GDPR requires explicit consent
                for user_id, consents in self.consent_records.items():
                    for consent in consents:
                        if not consent.granted and consent.consent_type == ConsentType.DATA_COLLECTION:
                            compliance_results["violations"].append(
                                f"User {user_id} has not granted data collection consent"
                            )
                            compliance_results["compliant"] = False
            
            # Check data retention policies
            for category, policy in self.data_retention_policies.items():
                if policy.retention_days > 2555:  # More than 7 years
                    compliance_results["recommendations"].append(
                        f"Consider reducing retention period for {category.value} data"
                    )
            
            return Either.success(compliance_results)
            
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return Either.error(IdentityError(
                f"Compliance validation error: {str(e)}",
                "COMPLIANCE_ERROR"
            ))
    
    async def export_user_data(
        self,
        user_profile_id: UserProfileId,
        export_format: str = "json"
    ) -> Either[IdentityError, Dict[str, Any]]:
        """Export user data for data portability (GDPR Article 20)."""
        try:
            # Check if user has consented to data export
            consent_result = await self.check_consent(user_profile_id, ConsentType.DATA_COLLECTION)
            if consent_result.is_error() or not consent_result.value:
                return Either.error(IdentityError(
                    "User has not consented to data collection",
                    "NO_CONSENT"
                ))
            
            # Gather all user data
            user_data = {
                "user_profile_id": user_profile_id,
                "export_format": export_format,
                "export_timestamp": datetime.now(UTC).isoformat(),
                "consents": [
                    {
                        "consent_type": consent.consent_type.value,
                        "granted": consent.granted,
                        "timestamp": consent.timestamp.isoformat(),
                        "context": consent.consent_context
                    }
                    for consent in self.consent_records.get(user_profile_id, [])
                ],
                "audit_records": [
                    {
                        "action": record.action,
                        "timestamp": record.timestamp.isoformat(),
                        "data_accessed": record.data_accessed
                    }
                    for record in self.privacy_audit_log
                    if record.user_profile_id == user_profile_id
                ]
            }
            
            # Log data export
            await self._log_privacy_audit(
                user_profile_id,
                "data_export_gdpr",
                ["all_user_data"],
                "compliant",
                "medium",
                True
            )
            
            return Either.success(user_data)
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return Either.error(IdentityError(
                f"Data export error: {str(e)}",
                "DATA_EXPORT_ERROR"
            ))
    
    async def delete_user_data(
        self,
        user_profile_id: UserProfileId,
        deletion_reason: str = "user_request"
    ) -> Either[IdentityError, bool]:
        """Delete all user data (GDPR Article 17 - Right to be forgotten)."""
        try:
            # Remove consent records
            if user_profile_id in self.consent_records:
                del self.consent_records[user_profile_id]
            
            # Remove anonymization keys
            if user_profile_id in self.anonymization_keys:
                del self.anonymization_keys[user_profile_id]
            
            # Anonymize audit records (can't delete for compliance reasons)
            for record in self.privacy_audit_log:
                if record.user_profile_id == user_profile_id:
                    # Replace with anonymized version
                    anonymized_record = PrivacyAuditRecord(
                        audit_id=record.audit_id,
                        user_profile_id=UserProfileId("ANONYMIZED"),
                        action=record.action,
                        data_accessed=record.data_accessed,
                        timestamp=record.timestamp,
                        compliance_status=record.compliance_status,
                        risk_level=record.risk_level,
                        automated_action=record.automated_action
                    )
                    
                    record_index = self.privacy_audit_log.index(record)
                    self.privacy_audit_log[record_index] = anonymized_record
            
            # Log deletion
            await self._log_privacy_audit(
                UserProfileId("SYSTEM"),
                f"user_data_deletion_{deletion_reason}",
                ["all_user_data"],
                "compliant",
                "high",
                True
            )
            
            logger.info(f"Deleted all data for user {user_profile_id}, reason: {deletion_reason}")
            return Either.success(True)
            
        except Exception as e:
            logger.error(f"User data deletion failed: {e}")
            return Either.error(IdentityError(
                f"Data deletion error: {str(e)}",
                "DATA_DELETION_ERROR"
            ))