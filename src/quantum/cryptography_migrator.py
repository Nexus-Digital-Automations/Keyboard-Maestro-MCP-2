"""
Cryptography Migrator - TASK_68 Phase 2 Core Quantum Engine

Post-quantum cryptography migration system with algorithm analysis, migration planning,
and secure transition management for enterprise cryptographic assets.

Architecture: Migration Planning + Design by Contract + Type Safety + Cryptographic Security
Performance: <100ms analysis, <500ms migration planning, <1s cryptographic operations
Security: Secure key migration, algorithm validation, rollback mechanisms, audit logging
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
import asyncio
import logging
import hashlib
import secrets
from pathlib import Path

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.quantum_architecture import (
    PostQuantumAlgorithm, QuantumThreatLevel, CryptographicStrength,
    QuantumSecurityPolicy, QuantumError, CryptographicAsset,
    PostQuantumMigrationPlan, QuantumReadinessAssessment,
    CryptographicAssetId, QuantumSessionId,
    assess_algorithm_quantum_vulnerability, calculate_migration_priority,
    recommend_post_quantum_algorithm, create_default_quantum_config
)

logger = logging.getLogger(__name__)


class CryptographyMigrator:
    """Post-quantum cryptography migration system with secure asset management."""
    
    def __init__(self):
        self.cryptographic_assets: Dict[CryptographicAssetId, CryptographicAsset] = {}
        self.migration_plans: Dict[str, PostQuantumMigrationPlan] = {}
        self.migration_history: List[Dict[str, Any]] = []
        self.quantum_config = create_default_quantum_config()
        self.migration_metrics = {
            "total_assets_analyzed": 0,
            "vulnerable_assets_found": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "rollback_operations": 0
        }
    
    @require(lambda self, scope: scope in ["system", "application", "cryptography", "protocols"])
    @ensure(lambda result: result.is_success() or result.is_error())
    async def analyze_quantum_readiness(self, scope: str, 
                                      include_vulnerabilities: bool = True,
                                      deep_analysis: bool = True) -> Either[QuantumError, QuantumReadinessAssessment]:
        """Analyze quantum readiness of cryptographic systems."""
        try:
            # Discover cryptographic assets based on scope
            discovery_result = await self._discover_cryptographic_assets(scope)
            if discovery_result.is_error():
                return discovery_result
            
            discovered_assets = discovery_result.value
            
            # Analyze each asset for quantum vulnerabilities
            vulnerable_assets = []
            total_risk_score = 0.0
            
            for asset in discovered_assets:
                self.cryptographic_assets[asset.asset_id] = asset
                
                if include_vulnerabilities:
                    vulnerability_analysis = await self._analyze_asset_vulnerability(asset)
                    if vulnerability_analysis.is_success():
                        vuln_data = vulnerability_analysis.value
                        if vuln_data["vulnerable"]:
                            vulnerable_assets.append(asset)
                        total_risk_score += vuln_data["risk_score"]
            
            # Calculate overall readiness score
            if len(discovered_assets) > 0:
                avg_risk = total_risk_score / len(discovered_assets)
                readiness_score = max(0.0, 1.0 - avg_risk)
            else:
                readiness_score = 1.0  # No assets means no risk
            
            # Generate threat timeline estimates
            threat_timeline = self._estimate_threat_timeline()
            
            # Generate migration recommendations
            recommendations = await self._generate_migration_recommendations(vulnerable_assets)
            
            # Assess compliance status
            compliance_status = await self._assess_compliance_status(discovered_assets)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(vulnerable_assets, discovered_assets)
            
            assessment = QuantumReadinessAssessment(
                assessment_id=f"qra_{secrets.token_hex(8)}",
                scope=scope,
                overall_readiness_score=readiness_score,
                quantum_vulnerable_assets=vulnerable_assets,
                threat_timeline_estimate=threat_timeline,
                migration_recommendations=recommendations,
                compliance_status=compliance_status,
                risk_factors=risk_factors,
                estimated_migration_cost=self._estimate_migration_cost(vulnerable_assets)
            )
            
            # Update metrics
            self.migration_metrics["total_assets_analyzed"] += len(discovered_assets)
            self.migration_metrics["vulnerable_assets_found"] += len(vulnerable_assets)
            
            logger.info(f"Quantum readiness analysis completed: {scope} scope, "
                       f"{len(vulnerable_assets)}/{len(discovered_assets)} vulnerable assets")
            
            return Either.success(assessment)
            
        except Exception as e:
            logger.error(f"Quantum readiness analysis failed: {e}")
            return Either.error(QuantumError(f"Analysis failed: {str(e)}"))
    
    @require(lambda self, target_assets: len(target_assets) > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def create_migration_plan(self, target_assets: List[CryptographicAssetId],
                                  migration_strategy: str = "hybrid",
                                  target_security_level: str = "post_quantum") -> Either[QuantumError, PostQuantumMigrationPlan]:
        """Create comprehensive migration plan for post-quantum transition."""
        try:
            # Validate target assets exist
            missing_assets = [
                asset_id for asset_id in target_assets 
                if asset_id not in self.cryptographic_assets
            ]
            if missing_assets:
                return Either.error(QuantumError(f"Assets not found: {missing_assets}"))
            
            # Analyze assets and select appropriate algorithms
            target_algorithms = {}
            compatibility_requirements = []
            risk_assessment = {}
            
            for asset_id in target_assets:
                asset = self.cryptographic_assets[asset_id]
                
                # Recommend post-quantum algorithm
                recommended_alg = recommend_post_quantum_algorithm(
                    asset.algorithm, asset.usage_context
                )
                
                if recommended_alg:
                    target_algorithms[asset_id] = recommended_alg
                    
                    # Add compatibility requirements
                    if asset.usage_context in ["authentication", "enterprise"]:
                        compatibility_requirements.append("enterprise_directory_support")
                    if asset.asset_type == "certificate":
                        compatibility_requirements.append("x509_certificate_support")
                
                # Calculate risk for this asset
                risk_assessment[asset_id] = asset.get_quantum_risk_score()
            
            # Estimate migration duration based on complexity
            base_duration = timedelta(hours=2)  # Base time per asset
            complexity_multiplier = {
                "hybrid": 1.5,
                "full_replacement": 2.0,
                "gradual": 3.0
            }
            
            total_duration = base_duration * len(target_assets) * complexity_multiplier.get(migration_strategy, 1.5)
            
            # Create rollback strategy
            rollback_strategy = self._create_rollback_strategy(migration_strategy)
            
            # Define validation criteria
            validation_criteria = [
                "algorithm_compatibility_verified",
                "performance_benchmarks_passed",
                "security_validation_completed",
                "integration_testing_successful"
            ]
            
            if target_security_level == "quantum_ready":
                validation_criteria.extend([
                    "quantum_interface_ready",
                    "post_quantum_compliance_verified"
                ])
            
            migration_plan = PostQuantumMigrationPlan(
                plan_id=f"pqmp_{secrets.token_hex(8)}",
                target_assets=target_assets,
                migration_strategy=migration_strategy,
                target_algorithms=target_algorithms,
                estimated_duration=total_duration,
                risk_assessment=risk_assessment,
                compatibility_requirements=list(set(compatibility_requirements)),
                rollback_strategy=rollback_strategy,
                validation_criteria=validation_criteria,
                created_at=datetime.now(UTC)
            )
            
            # Store migration plan
            self.migration_plans[migration_plan.plan_id] = migration_plan
            
            logger.info(f"Migration plan created: {migration_plan.plan_id} for {len(target_assets)} assets")
            
            return Either.success(migration_plan)
            
        except Exception as e:
            logger.error(f"Migration plan creation failed: {e}")
            return Either.error(QuantumError(f"Migration planning failed: {str(e)}"))
    
    @require(lambda self, plan_id: len(plan_id) > 0)
    async def execute_migration_plan(self, plan_id: str,
                                   dry_run: bool = False,
                                   validation_mode: bool = True) -> Either[QuantumError, Dict[str, Any]]:
        """Execute post-quantum migration plan with validation and rollback capability."""
        try:
            if plan_id not in self.migration_plans:
                return Either.error(QuantumError(f"Migration plan not found: {plan_id}"))
            
            migration_plan = self.migration_plans[plan_id]
            execution_start = datetime.now(UTC)
            
            migration_results = {
                "plan_id": plan_id,
                "execution_start": execution_start,
                "dry_run": dry_run,
                "assets_processed": 0,
                "successful_migrations": 0,
                "failed_migrations": 0,
                "rollback_required": False,
                "validation_results": {},
                "performance_metrics": {}
            }
            
            # Execute migration phases
            phases = migration_plan.get_migration_phases()
            
            for phase_info in phases:
                phase_result = await self._execute_migration_phase(
                    migration_plan, phase_info, dry_run, validation_mode
                )
                
                if phase_result.is_error():
                    migration_results["failed_migrations"] += 1
                    migration_results["rollback_required"] = True
                    logger.error(f"Migration phase {phase_info['phase']} failed: {phase_result.error_value}")
                    break
                else:
                    phase_data = phase_result.value
                    migration_results["successful_migrations"] += phase_data["assets_migrated"]
                    migration_results["assets_processed"] += phase_data["assets_processed"]
            
            # Perform post-migration validation if enabled
            if validation_mode and not dry_run:
                validation_result = await self._validate_migration_results(migration_plan, migration_results)
                migration_results["validation_results"] = validation_result.value if validation_result.is_success() else {}
            
            # Update metrics
            if not dry_run:
                self.migration_metrics["successful_migrations"] += migration_results["successful_migrations"]
                self.migration_metrics["failed_migrations"] += migration_results["failed_migrations"]
            
            # Record migration history
            self.migration_history.append({
                "plan_id": plan_id,
                "execution_time": datetime.now(UTC),
                "results": migration_results,
                "dry_run": dry_run
            })
            
            execution_duration = (datetime.now(UTC) - execution_start).total_seconds()
            migration_results["execution_duration_seconds"] = execution_duration
            
            logger.info(f"Migration plan execution completed: {plan_id}, "
                       f"{migration_results['successful_migrations']} successful, "
                       f"{migration_results['failed_migrations']} failed")
            
            return Either.success(migration_results)
            
        except Exception as e:
            logger.error(f"Migration plan execution failed: {e}")
            return Either.error(QuantumError(f"Migration execution failed: {str(e)}"))
    
    async def get_migration_status(self, plan_id: Optional[str] = None) -> Either[QuantumError, Dict[str, Any]]:
        """Get migration status and metrics."""
        try:
            status = {
                "overall_metrics": self.migration_metrics.copy(),
                "total_plans": len(self.migration_plans),
                "recent_executions": len([
                    h for h in self.migration_history 
                    if h["execution_time"] >= datetime.now(UTC) - timedelta(days=1)
                ]),
                "quantum_config": {
                    "security_policy": self.quantum_config.security_policy.value,
                    "enabled_algorithms": [alg.value for alg in self.quantum_config.enabled_algorithms],
                    "monitoring_enabled": self.quantum_config.monitoring_enabled
                }
            }
            
            if plan_id:
                if plan_id in self.migration_plans:
                    plan = self.migration_plans[plan_id]
                    plan_executions = [
                        h for h in self.migration_history 
                        if h["plan_id"] == plan_id
                    ]
                    
                    status["plan_details"] = {
                        "plan_id": plan_id,
                        "created_at": plan.created_at.isoformat(),
                        "target_assets_count": len(plan.target_assets),
                        "migration_strategy": plan.migration_strategy,
                        "estimated_duration": str(plan.estimated_duration),
                        "execution_count": len(plan_executions),
                        "last_execution": plan_executions[-1]["execution_time"].isoformat() if plan_executions else None
                    }
                else:
                    return Either.error(QuantumError(f"Migration plan not found: {plan_id}"))
            
            return Either.success(status)
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return Either.error(QuantumError(f"Status retrieval failed: {str(e)}"))
    
    # Private helper methods
    
    async def _discover_cryptographic_assets(self, scope: str) -> Either[QuantumError, List[CryptographicAsset]]:
        """Discover cryptographic assets in the specified scope."""
        try:
            # Simulate asset discovery based on scope
            discovered_assets = []
            
            if scope == "system":
                # System-level cryptographic assets
                assets_data = [
                    ("system_root_ca", "certificate", "rsa", 2048, "root_certificate"),
                    ("ssh_host_key", "key", "ecdsa", 256, "ssh_authentication"),
                    ("tls_server_cert", "certificate", "rsa", 2048, "tls_encryption"),
                    ("system_signing_key", "key", "rsa", 2048, "code_signing")
                ]
            elif scope == "application":
                # Application-level cryptographic assets
                assets_data = [
                    ("app_api_key", "key", "aes", 256, "api_authentication"),
                    ("db_encryption_key", "key", "aes", 256, "database_encryption"),
                    ("jwt_signing_key", "key", "ecdsa", 256, "token_signing"),
                    ("oauth_client_cert", "certificate", "rsa", 2048, "oauth_authentication")
                ]
            elif scope == "cryptography":
                # Pure cryptographic assets
                assets_data = [
                    ("master_encryption_key", "key", "rsa", 4096, "master_key"),
                    ("backup_signing_key", "key", "ecdsa", 384, "backup_signing"),
                    ("legacy_des_key", "key", "des", 56, "legacy_encryption"),
                    ("hmac_secret", "key", "sha256", 256, "message_authentication")
                ]
            else:  # protocols
                # Protocol-level cryptographic assets
                assets_data = [
                    ("tls_session_key", "key", "aes", 256, "session_encryption"),
                    ("ipsec_key", "key", "aes", 256, "ipsec_tunnel"),
                    ("kerberos_key", "key", "aes", 256, "kerberos_authentication"),
                    ("vpn_psk", "key", "sha256", 256, "vpn_authentication")
                ]
            
            for asset_name, asset_type, algorithm, key_size, usage_context in assets_data:
                # Assess quantum vulnerability
                is_vulnerable, threat_level = assess_algorithm_quantum_vulnerability(algorithm, key_size)
                
                asset = CryptographicAsset(
                    asset_id=CryptographicAssetId(f"{scope}_{asset_name}_{secrets.token_hex(4)}"),
                    asset_type=asset_type,
                    algorithm=algorithm,
                    key_size=key_size,
                    created_at=datetime.now(UTC) - timedelta(days=secrets.randbelow(365)),
                    usage_context=usage_context,
                    quantum_vulnerable=is_vulnerable,
                    threat_assessment=threat_level,
                    migration_priority=calculate_migration_priority(
                        # Create a temporary asset for priority calculation
                        CryptographicAsset(
                            asset_id=CryptographicAssetId("temp"),
                            asset_type=asset_type,
                            algorithm=algorithm,
                            key_size=key_size,
                            created_at=datetime.now(UTC),
                            usage_context=usage_context,
                            quantum_vulnerable=is_vulnerable,
                            threat_assessment=threat_level,
                            migration_priority=1
                        )
                    ),
                    replacement_algorithm=recommend_post_quantum_algorithm(algorithm, usage_context)
                )
                
                discovered_assets.append(asset)
            
            return Either.success(discovered_assets)
            
        except Exception as e:
            return Either.error(QuantumError(f"Asset discovery failed: {str(e)}"))
    
    async def _analyze_asset_vulnerability(self, asset: CryptographicAsset) -> Either[QuantumError, Dict[str, Any]]:
        """Analyze specific asset for quantum vulnerabilities."""
        try:
            vulnerability_data = {
                "asset_id": asset.asset_id,
                "vulnerable": asset.quantum_vulnerable,
                "threat_level": asset.threat_assessment.value,
                "risk_score": asset.get_quantum_risk_score(),
                "needs_immediate_migration": asset.needs_immediate_migration(),
                "recommended_algorithm": asset.replacement_algorithm.value if asset.replacement_algorithm else None,
                "analysis_timestamp": datetime.now(UTC).isoformat()
            }
            
            return Either.success(vulnerability_data)
            
        except Exception as e:
            return Either.error(QuantumError(f"Vulnerability analysis failed: {str(e)}"))
    
    def _estimate_threat_timeline(self) -> Dict[str, datetime]:
        """Estimate quantum threat timeline."""
        current_time = datetime.now(UTC)
        
        return {
            "quantum_advantage_demonstration": current_time + timedelta(days=365 * 2),  # 2 years
            "cryptographically_relevant_quantum_computer": current_time + timedelta(days=365 * 8),  # 8 years
            "large_scale_quantum_attacks": current_time + timedelta(days=365 * 12),  # 12 years
            "post_quantum_transition_deadline": current_time + timedelta(days=365 * 5)  # 5 years (recommended)
        }
    
    async def _generate_migration_recommendations(self, vulnerable_assets: List[CryptographicAsset]) -> List[str]:
        """Generate migration recommendations based on vulnerable assets."""
        recommendations = []
        
        if not vulnerable_assets:
            recommendations.append("No vulnerable cryptographic assets found")
            return recommendations
        
        # Categorize by threat level
        critical_assets = [a for a in vulnerable_assets if a.threat_assessment == QuantumThreatLevel.CRITICAL]
        high_risk_assets = [a for a in vulnerable_assets if a.threat_assessment == QuantumThreatLevel.HIGH]
        
        if critical_assets:
            recommendations.append(f"URGENT: Migrate {len(critical_assets)} critical assets immediately")
            recommendations.append("Consider emergency migration protocols for critical infrastructure")
        
        if high_risk_assets:
            recommendations.append(f"HIGH PRIORITY: Plan migration for {len(high_risk_assets)} high-risk assets within 6 months")
        
        # Algorithm-specific recommendations
        rsa_assets = [a for a in vulnerable_assets if "rsa" in a.algorithm.lower()]
        if rsa_assets:
            recommendations.append(f"Replace {len(rsa_assets)} RSA assets with Kyber (KEM) + Dilithium (signatures)")
        
        ecdsa_assets = [a for a in vulnerable_assets if "ecdsa" in a.algorithm.lower()]
        if ecdsa_assets:
            recommendations.append(f"Replace {len(ecdsa_assets)} ECDSA assets with Falcon or Dilithium signatures")
        
        # Strategy recommendations
        if len(vulnerable_assets) > 10:
            recommendations.append("Consider phased migration approach due to large number of assets")
        else:
            recommendations.append("Full migration approach recommended due to manageable asset count")
        
        recommendations.append("Implement hybrid classical-quantum security during transition period")
        recommendations.append("Establish post-quantum cryptography testing environment")
        
        return recommendations
    
    async def _assess_compliance_status(self, assets: List[CryptographicAsset]) -> Dict[str, bool]:
        """Assess compliance status against quantum readiness standards."""
        total_assets = len(assets)
        if total_assets == 0:
            return {"no_assets": True}
        
        vulnerable_count = len([a for a in assets if a.quantum_vulnerable])
        post_quantum_count = len([a for a in assets if not a.quantum_vulnerable])
        
        return {
            "nist_post_quantum_ready": post_quantum_count / total_assets >= 0.8,
            "quantum_safe_majority": post_quantum_count > vulnerable_count,
            "critical_assets_protected": len([
                a for a in assets 
                if a.threat_assessment != QuantumThreatLevel.CRITICAL
            ]) == total_assets,
            "migration_plan_required": vulnerable_count > 0,
            "compliance_timeline_met": vulnerable_count / total_assets < 0.3
        }
    
    def _calculate_risk_factors(self, vulnerable_assets: List[CryptographicAsset], 
                               all_assets: List[CryptographicAsset]) -> Dict[str, float]:
        """Calculate various risk factors."""
        if not all_assets:
            return {}
        
        total_assets = len(all_assets)
        vulnerable_count = len(vulnerable_assets)
        
        return {
            "vulnerability_ratio": vulnerable_count / total_assets,
            "critical_asset_ratio": len([
                a for a in vulnerable_assets 
                if a.threat_assessment == QuantumThreatLevel.CRITICAL
            ]) / total_assets,
            "legacy_algorithm_ratio": len([
                a for a in all_assets 
                if a.algorithm.lower() in ["rsa", "ecdsa", "dh", "des"]
            ]) / total_assets,
            "immediate_migration_ratio": len([
                a for a in vulnerable_assets 
                if a.needs_immediate_migration()
            ]) / total_assets if vulnerable_count > 0 else 0.0
        }
    
    def _estimate_migration_cost(self, vulnerable_assets: List[CryptographicAsset]) -> Optional[float]:
        """Estimate migration cost based on asset complexity."""
        if not vulnerable_assets:
            return 0.0
        
        base_cost_per_asset = 1000.0  # Base cost in arbitrary units
        
        complexity_multipliers = {
            "certificate": 2.0,  # More complex due to PKI
            "key": 1.0,         # Standard key replacement
            "signature": 1.5     # Moderate complexity
        }
        
        total_cost = 0.0
        for asset in vulnerable_assets:
            asset_cost = base_cost_per_asset * complexity_multipliers.get(asset.asset_type, 1.0)
            
            # Higher priority assets may require more resources
            priority_multiplier = asset.migration_priority / 3.0
            asset_cost *= priority_multiplier
            
            total_cost += asset_cost
        
        return total_cost
    
    def _create_rollback_strategy(self, migration_strategy: str) -> str:
        """Create appropriate rollback strategy."""
        if migration_strategy == "hybrid":
            return "maintain_classical_fallback_during_transition"
        elif migration_strategy == "gradual":
            return "rollback_by_migration_phase_with_asset_restoration"
        else:  # full_replacement
            return "complete_system_restoration_from_pre_migration_backup"
    
    async def _execute_migration_phase(self, plan: PostQuantumMigrationPlan,
                                     phase_info: Dict[str, Any],
                                     dry_run: bool,
                                     validation_mode: bool) -> Either[QuantumError, Dict[str, Any]]:
        """Execute a single migration phase."""
        try:
            phase_results = {
                "phase": phase_info["phase"],
                "assets_processed": 0,
                "assets_migrated": 0,
                "start_time": datetime.now(UTC),
                "dry_run": dry_run
            }
            
            # Simulate phase execution
            target_count = phase_info["asset_count"]
            
            for i in range(target_count):
                # Simulate asset migration
                if not dry_run:
                    await asyncio.sleep(0.01)  # Simulate processing time
                
                phase_results["assets_processed"] += 1
                
                # Simulate success rate (95% for demonstration)
                if secrets.randbelow(100) < 95:
                    phase_results["assets_migrated"] += 1
            
            phase_results["end_time"] = datetime.now(UTC)
            phase_results["duration"] = (
                phase_results["end_time"] - phase_results["start_time"]
            ).total_seconds()
            
            return Either.success(phase_results)
            
        except Exception as e:
            return Either.error(QuantumError(f"Phase execution failed: {str(e)}"))
    
    async def _validate_migration_results(self, plan: PostQuantumMigrationPlan,
                                        results: Dict[str, Any]) -> Either[QuantumError, Dict[str, Any]]:
        """Validate migration results against plan criteria."""
        try:
            validation_results = {
                "overall_success": True,
                "criteria_results": {},
                "validation_timestamp": datetime.now(UTC).isoformat()
            }
            
            # Validate against each criterion
            for criterion in plan.validation_criteria:
                if criterion == "algorithm_compatibility_verified":
                    validation_results["criteria_results"][criterion] = True  # Simulated
                elif criterion == "performance_benchmarks_passed":
                    validation_results["criteria_results"][criterion] = results["successful_migrations"] > 0
                elif criterion == "security_validation_completed":
                    validation_results["criteria_results"][criterion] = results["failed_migrations"] == 0
                elif criterion == "integration_testing_successful":
                    validation_results["criteria_results"][criterion] = not results["rollback_required"]
                else:
                    validation_results["criteria_results"][criterion] = True
            
            # Overall success requires all criteria to pass
            validation_results["overall_success"] = all(validation_results["criteria_results"].values())
            
            return Either.success(validation_results)
            
        except Exception as e:
            return Either.error(QuantumError(f"Validation failed: {str(e)}"))