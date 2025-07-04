"""
Algorithm Analyzer - TASK_68 Phase 2 Core Quantum Engine

Cryptographic vulnerability assessment and quantum algorithm analysis with comprehensive
security evaluation, threat modeling, and migration recommendations.

Architecture: Security Analysis + Design by Contract + Type Safety + Threat Assessment
Performance: <50ms algorithm analysis, <100ms vulnerability assessment, <200ms report generation  
Security: Comprehensive threat modeling, quantum vulnerability assessment, secure analysis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
import asyncio
import logging
import hashlib
import secrets
import re
from pathlib import Path
from dataclasses import dataclass, field

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.quantum_architecture import (
    PostQuantumAlgorithm, QuantumThreatLevel, CryptographicStrength,
    QuantumError, CryptographicAsset, CryptographicAssetId,
    assess_algorithm_quantum_vulnerability, recommend_post_quantum_algorithm,
    calculate_migration_priority
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgorithmAnalysis:
    """Comprehensive algorithm analysis results."""
    algorithm_name: str
    algorithm_type: str  # symmetric|asymmetric|hash|signature|kem
    key_size: int
    quantum_vulnerable: bool
    threat_level: QuantumThreatLevel
    strength_category: CryptographicStrength
    vulnerability_details: Dict[str, Any]
    security_margin: float  # Years before quantum threat
    replacement_recommendations: List[PostQuantumAlgorithm]
    analysis_confidence: float  # 0.0 to 1.0
    
    @require(lambda self: 0.0 <= self.analysis_confidence <= 1.0)
    @require(lambda self: self.security_margin >= 0.0)
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class VulnerabilityAssessment:
    """Quantum vulnerability assessment results."""
    assessment_id: str
    target_system: str
    scope: str  # system|application|protocol|cryptography
    total_algorithms_analyzed: int
    vulnerable_algorithms: List[AlgorithmAnalysis]
    secure_algorithms: List[AlgorithmAnalysis]
    overall_risk_score: float  # 0.0 to 1.0
    critical_vulnerabilities: int
    high_risk_vulnerabilities: int
    migration_urgency: str  # immediate|high|medium|low
    threat_timeline: Dict[str, datetime]
    recommendations: List[str]
    
    @require(lambda self: 0.0 <= self.overall_risk_score <= 1.0)
    @require(lambda self: self.total_algorithms_analyzed >= 0)
    def __post_init__(self):
        pass


class AlgorithmAnalyzer:
    """Cryptographic algorithm analyzer with quantum vulnerability assessment."""
    
    def __init__(self):
        self.analysis_cache: Dict[str, AlgorithmAnalysis] = {}
        self.vulnerability_assessments: Dict[str, VulnerabilityAssessment] = {}
        self.algorithm_patterns: Dict[str, Dict[str, Any]] = {}
        self.threat_intelligence: Dict[str, Any] = {}
        self.analysis_metrics = {
            "total_analyses": 0,
            "vulnerable_found": 0,
            "secure_algorithms": 0,
            "assessments_completed": 0,
            "cache_hits": 0
        }
        
        # Initialize algorithm patterns and threat intelligence
        self._initialize_algorithm_patterns()
        self._initialize_threat_intelligence()
    
    @require(lambda self, algorithm_name: len(algorithm_name) > 0)
    @require(lambda self, key_size: key_size > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def analyze_algorithm(self, algorithm_name: str, key_size: int,
                              usage_context: str = "general",
                              detailed_analysis: bool = True) -> Either[QuantumError, AlgorithmAnalysis]:
        """Analyze cryptographic algorithm for quantum vulnerabilities."""
        try:
            # Check cache first
            cache_key = f"{algorithm_name}_{key_size}_{usage_context}"
            if cache_key in self.analysis_cache:
                self.analysis_metrics["cache_hits"] += 1
                return Either.success(self.analysis_cache[cache_key])
            
            # Perform algorithm analysis
            algorithm_type = self._classify_algorithm_type(algorithm_name)
            is_vulnerable, threat_level = assess_algorithm_quantum_vulnerability(algorithm_name, key_size)
            
            # Determine cryptographic strength
            strength_category = self._assess_cryptographic_strength(algorithm_name, is_vulnerable)
            
            # Calculate vulnerability details
            vulnerability_details = await self._analyze_vulnerability_details(
                algorithm_name, key_size, algorithm_type, detailed_analysis
            )
            
            # Estimate security margin
            security_margin = self._estimate_security_margin(algorithm_name, key_size, threat_level)
            
            # Get replacement recommendations
            replacement_recommendations = self._get_replacement_recommendations(
                algorithm_name, usage_context, threat_level
            )
            
            # Calculate analysis confidence
            confidence = self._calculate_analysis_confidence(algorithm_name, algorithm_type)
            
            # Create analysis result
            analysis = AlgorithmAnalysis(
                algorithm_name=algorithm_name,
                algorithm_type=algorithm_type,
                key_size=key_size,
                quantum_vulnerable=is_vulnerable,
                threat_level=threat_level,
                strength_category=strength_category,
                vulnerability_details=vulnerability_details,
                security_margin=security_margin,
                replacement_recommendations=replacement_recommendations,
                analysis_confidence=confidence
            )
            
            # Cache result
            self.analysis_cache[cache_key] = analysis
            
            # Update metrics
            self.analysis_metrics["total_analyses"] += 1
            if is_vulnerable:
                self.analysis_metrics["vulnerable_found"] += 1
            else:
                self.analysis_metrics["secure_algorithms"] += 1
            
            logger.info(f"Algorithm analysis completed: {algorithm_name} ({key_size} bits) - Vulnerable: {is_vulnerable}")
            
            return Either.success(analysis)
            
        except Exception as e:
            logger.error(f"Algorithm analysis failed: {e}")
            return Either.error(QuantumError(f"Analysis failed: {str(e)}"))
    
    @require(lambda self, algorithms: len(algorithms) > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def assess_system_vulnerabilities(self, algorithms: List[Dict[str, Any]],
                                          system_name: str = "target_system",
                                          scope: str = "system") -> Either[QuantumError, VulnerabilityAssessment]:
        """Perform comprehensive vulnerability assessment on cryptographic algorithms."""
        try:
            assessment_id = f"va_{secrets.token_hex(8)}"
            assessment_start = datetime.now(UTC)
            
            vulnerable_algorithms = []
            secure_algorithms = []
            total_risk_score = 0.0
            critical_count = 0
            high_risk_count = 0
            
            # Analyze each algorithm
            for alg_info in algorithms:
                algorithm_name = alg_info["name"]
                key_size = alg_info.get("key_size", 256)
                usage_context = alg_info.get("usage_context", "general")
                
                analysis_result = await self.analyze_algorithm(algorithm_name, key_size, usage_context)
                
                if analysis_result.is_success():
                    analysis = analysis_result.value
                    
                    if analysis.quantum_vulnerable:
                        vulnerable_algorithms.append(analysis)
                        
                        # Count severity levels
                        if analysis.threat_level == QuantumThreatLevel.CRITICAL:
                            critical_count += 1
                        elif analysis.threat_level == QuantumThreatLevel.HIGH:
                            high_risk_count += 1
                        
                        # Add to risk score
                        risk_weight = self._get_threat_level_weight(analysis.threat_level)
                        total_risk_score += risk_weight
                    else:
                        secure_algorithms.append(analysis)
            
            # Calculate overall risk score
            total_algorithms = len(algorithms)
            overall_risk_score = min(1.0, total_risk_score / total_algorithms) if total_algorithms > 0 else 0.0
            
            # Determine migration urgency
            migration_urgency = self._determine_migration_urgency(critical_count, high_risk_count, overall_risk_score)
            
            # Generate threat timeline
            threat_timeline = self._generate_threat_timeline()
            
            # Generate recommendations
            recommendations = await self._generate_assessment_recommendations(
                vulnerable_algorithms, secure_algorithms, overall_risk_score
            )
            
            # Create vulnerability assessment
            assessment = VulnerabilityAssessment(
                assessment_id=assessment_id,
                target_system=system_name,
                scope=scope,
                total_algorithms_analyzed=total_algorithms,
                vulnerable_algorithms=vulnerable_algorithms,
                secure_algorithms=secure_algorithms,
                overall_risk_score=overall_risk_score,
                critical_vulnerabilities=critical_count,
                high_risk_vulnerabilities=high_risk_count,
                migration_urgency=migration_urgency,
                threat_timeline=threat_timeline,
                recommendations=recommendations
            )
            
            # Store assessment
            self.vulnerability_assessments[assessment_id] = assessment
            self.analysis_metrics["assessments_completed"] += 1
            
            logger.info(f"Vulnerability assessment completed: {assessment_id} - "
                       f"{len(vulnerable_algorithms)}/{total_algorithms} vulnerable algorithms")
            
            return Either.success(assessment)
            
        except Exception as e:
            logger.error(f"Vulnerability assessment failed: {e}")
            return Either.error(QuantumError(f"Assessment failed: {str(e)}"))
    
    @require(lambda self, algorithm_patterns: len(algorithm_patterns) > 0)
    async def discover_algorithms_in_code(self, code_content: str,
                                        file_path: str = "unknown") -> Either[QuantumError, List[Dict[str, Any]]]:
        """Discover cryptographic algorithms in source code."""
        try:
            discovered_algorithms = []
            
            # Search for algorithm patterns
            for pattern_name, pattern_info in self.algorithm_patterns.items():
                regex_patterns = pattern_info["patterns"]
                algorithm_type = pattern_info["type"]
                default_key_size = pattern_info.get("default_key_size", 256)
                
                for regex_pattern in regex_patterns:
                    matches = re.finditer(regex_pattern, code_content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Extract key size if present in match groups
                        key_size = default_key_size
                        if match.groups():
                            try:
                                # Try to extract key size from first group
                                key_size = int(match.group(1))
                            except (ValueError, IndexError):
                                pass
                        
                        discovered_algorithms.append({
                            "name": pattern_name,
                            "algorithm_type": algorithm_type,
                            "key_size": key_size,
                            "location": {
                                "file": file_path,
                                "line": code_content[:match.start()].count('\n') + 1,
                                "context": match.group(0)
                            },
                            "usage_context": self._infer_usage_context(match.group(0), code_content)
                        })
            
            logger.info(f"Algorithm discovery completed: {len(discovered_algorithms)} algorithms found in {file_path}")
            
            return Either.success(discovered_algorithms)
            
        except Exception as e:
            logger.error(f"Algorithm discovery failed: {e}")
            return Either.error(QuantumError(f"Discovery failed: {str(e)}"))
    
    async def generate_migration_roadmap(self, assessment_id: str) -> Either[QuantumError, Dict[str, Any]]:
        """Generate quantum migration roadmap based on vulnerability assessment."""
        try:
            if assessment_id not in self.vulnerability_assessments:
                return Either.error(QuantumError(f"Assessment not found: {assessment_id}"))
            
            assessment = self.vulnerability_assessments[assessment_id]
            
            # Prioritize vulnerable algorithms by threat level and usage
            prioritized_algorithms = sorted(
                assessment.vulnerable_algorithms,
                key=lambda a: (
                    self._get_threat_level_weight(a.threat_level),
                    -a.security_margin,
                    a.key_size  # Smaller keys = higher priority
                ),
                reverse=True
            )
            
            # Create migration phases
            migration_phases = []
            
            # Phase 1: Critical and immediate threats
            critical_algorithms = [
                a for a in prioritized_algorithms
                if a.threat_level in [QuantumThreatLevel.CRITICAL, QuantumThreatLevel.HIGH]
                and a.security_margin < 2.0
            ]
            
            if critical_algorithms:
                migration_phases.append({
                    "phase": 1,
                    "name": "Immediate Security Upgrade",
                    "duration_months": 3,
                    "algorithms": critical_algorithms,
                    "priority": "critical",
                    "estimated_effort": "high",
                    "success_criteria": "All critical vulnerabilities mitigated"
                })
            
            # Phase 2: High-risk algorithms
            high_risk_algorithms = [
                a for a in prioritized_algorithms
                if a.threat_level == QuantumThreatLevel.HIGH
                and a.security_margin >= 2.0
                and a not in critical_algorithms
            ]
            
            if high_risk_algorithms:
                migration_phases.append({
                    "phase": 2,
                    "name": "Strategic Security Modernization",
                    "duration_months": 6,
                    "algorithms": high_risk_algorithms,
                    "priority": "high",
                    "estimated_effort": "medium",
                    "success_criteria": "Major quantum vulnerabilities addressed"
                })
            
            # Phase 3: Medium-risk algorithms
            medium_risk_algorithms = [
                a for a in prioritized_algorithms
                if a.threat_level == QuantumThreatLevel.MEDIUM
            ]
            
            if medium_risk_algorithms:
                migration_phases.append({
                    "phase": 3,
                    "name": "Comprehensive Security Enhancement",
                    "duration_months": 12,
                    "algorithms": medium_risk_algorithms,
                    "priority": "medium",
                    "estimated_effort": "low",
                    "success_criteria": "All identified vulnerabilities resolved"
                })
            
            # Generate migration recommendations
            migration_strategies = self._generate_migration_strategies(assessment)
            risk_mitigation_plan = self._create_risk_mitigation_plan(assessment)
            
            roadmap = {
                "assessment_id": assessment_id,
                "migration_phases": migration_phases,
                "total_phases": len(migration_phases),
                "estimated_duration_months": sum(phase["duration_months"] for phase in migration_phases),
                "migration_strategies": migration_strategies,
                "risk_mitigation_plan": risk_mitigation_plan,
                "success_metrics": {
                    "quantum_readiness_score": 1.0 - assessment.overall_risk_score,
                    "critical_vulnerabilities_resolved": len(critical_algorithms),
                    "overall_vulnerabilities_resolved": len(prioritized_algorithms)
                },
                "generated_at": datetime.now(UTC).isoformat()
            }
            
            return Either.success(roadmap)
            
        except Exception as e:
            logger.error(f"Migration roadmap generation failed: {e}")
            return Either.error(QuantumError(f"Roadmap generation failed: {str(e)}"))
    
    async def get_analysis_status(self) -> Either[QuantumError, Dict[str, Any]]:
        """Get algorithm analysis status and metrics."""
        try:
            status = {
                "analysis_metrics": self.analysis_metrics.copy(),
                "cached_analyses": len(self.analysis_cache),
                "vulnerability_assessments": len(self.vulnerability_assessments),
                "algorithm_patterns": len(self.algorithm_patterns),
                "threat_intelligence_sources": len(self.threat_intelligence),
                "recent_assessments": [
                    {
                        "assessment_id": assessment_id,
                        "target_system": assessment.target_system,
                        "overall_risk_score": assessment.overall_risk_score,
                        "critical_vulnerabilities": assessment.critical_vulnerabilities
                    }
                    for assessment_id, assessment in list(self.vulnerability_assessments.items())[-5:]
                ]
            }
            
            return Either.success(status)
            
        except Exception as e:
            logger.error(f"Failed to get analysis status: {e}")
            return Either.error(QuantumError(f"Status retrieval failed: {str(e)}"))
    
    # Private helper methods
    
    def _initialize_algorithm_patterns(self):
        """Initialize cryptographic algorithm detection patterns."""
        self.algorithm_patterns = {
            "rsa": {
                "type": "asymmetric",
                "patterns": [
                    r"rsa[_-]?(\d+)",
                    r"RSA[_-]?(\d+)",
                    r"generateKeyPair.*rsa.*(\d+)",
                    r"RSAKeyGenParameterSpec.*(\d+)"
                ],
                "default_key_size": 2048
            },
            "ecdsa": {
                "type": "signature",
                "patterns": [
                    r"ecdsa[_-]?p(\d+)",
                    r"ECDSA[_-]?P(\d+)",
                    r"secp(\d+)r1",
                    r"prime(\d+)v1"
                ],
                "default_key_size": 256
            },
            "aes": {
                "type": "symmetric",
                "patterns": [
                    r"aes[_-]?(\d+)",
                    r"AES[_-]?(\d+)",
                    r"Advanced.*Encryption.*(\d+)"
                ],
                "default_key_size": 256
            },
            "des": {
                "type": "symmetric",
                "patterns": [
                    r"\bdes\b",
                    r"\bDES\b",
                    r"Data.*Encryption.*Standard"
                ],
                "default_key_size": 56
            },
            "sha": {
                "type": "hash",
                "patterns": [
                    r"sha[_-]?(\d+)",
                    r"SHA[_-]?(\d+)",
                    r"Secure.*Hash.*(\d+)"
                ],
                "default_key_size": 256
            },
            "md5": {
                "type": "hash",
                "patterns": [
                    r"\bmd5\b",
                    r"\bMD5\b",
                    r"Message.*Digest.*5"
                ],
                "default_key_size": 128
            }
        }
    
    def _initialize_threat_intelligence(self):
        """Initialize quantum threat intelligence data."""
        self.threat_intelligence = {
            "quantum_computer_progress": {
                "current_qubits": 1000,  # Approximate current maximum
                "fault_tolerant_estimate": 2030,
                "cryptographically_relevant": 2035
            },
            "algorithm_vulnerabilities": {
                "rsa": "shor_algorithm",
                "ecdsa": "shor_algorithm",
                "dh": "shor_algorithm",
                "symmetric": "grover_algorithm"
            },
            "security_margins": {
                "conservative": 10,  # Years
                "moderate": 15,
                "optimistic": 20
            }
        }
    
    def _classify_algorithm_type(self, algorithm_name: str) -> str:
        """Classify algorithm type based on name."""
        algorithm_lower = algorithm_name.lower()
        
        if any(asym in algorithm_lower for asym in ["rsa", "ecc", "dh", "ecdh", "ecdsa"]):
            return "asymmetric"
        elif any(sym in algorithm_lower for sym in ["aes", "des", "chacha", "salsa"]):
            return "symmetric"
        elif any(hash_name in algorithm_lower for hash_name in ["sha", "md5", "blake", "keccak"]):
            return "hash"
        elif any(sig in algorithm_lower for sig in ["dsa", "ecdsa", "rsa-pss"]):
            return "signature"
        elif any(kem in algorithm_lower for kem in ["kyber", "ntru", "saber"]):
            return "kem"
        else:
            return "unknown"
    
    def _assess_cryptographic_strength(self, algorithm_name: str, is_vulnerable: bool) -> CryptographicStrength:
        """Assess cryptographic strength category."""
        algorithm_lower = algorithm_name.lower()
        
        # Check for post-quantum algorithms
        post_quantum_names = [alg.value for alg in PostQuantumAlgorithm]
        if any(pq_name in algorithm_lower for pq_name in post_quantum_names):
            return CryptographicStrength.QUANTUM_RESISTANT
        
        # Classical algorithms
        if is_vulnerable:
            return CryptographicStrength.CLASSICAL_ONLY
        else:
            # Non-vulnerable classical algorithms (like AES with sufficient key size)
            return CryptographicStrength.CLASSICAL_ONLY
    
    async def _analyze_vulnerability_details(self, algorithm_name: str, key_size: int,
                                           algorithm_type: str, detailed: bool) -> Dict[str, Any]:
        """Analyze detailed vulnerability information."""
        details = {
            "quantum_attack_vector": "unknown",
            "classical_security": "unknown",
            "quantum_advantage": "unknown",
            "mitigation_options": []
        }
        
        algorithm_lower = algorithm_name.lower()
        
        # Determine quantum attack vector
        if algorithm_type in ["asymmetric", "signature"] and any(alg in algorithm_lower for alg in ["rsa", "ecc", "dh"]):
            details["quantum_attack_vector"] = "shor_algorithm"
            details["quantum_advantage"] = "exponential"
        elif algorithm_type in ["symmetric", "hash"]:
            details["quantum_attack_vector"] = "grover_algorithm"
            details["quantum_advantage"] = "quadratic"
        
        # Assess classical security
        if algorithm_type == "symmetric":
            if key_size >= 256:
                details["classical_security"] = "strong"
            elif key_size >= 128:
                details["classical_security"] = "adequate"
            else:
                details["classical_security"] = "weak"
        elif algorithm_type == "asymmetric":
            if key_size >= 3072:
                details["classical_security"] = "strong"
            elif key_size >= 2048:
                details["classical_security"] = "adequate"
            else:
                details["classical_security"] = "weak"
        
        # Add mitigation options
        if details["quantum_attack_vector"] == "shor_algorithm":
            details["mitigation_options"] = ["post_quantum_kem", "post_quantum_signatures", "hybrid_approach"]
        elif details["quantum_attack_vector"] == "grover_algorithm":
            details["mitigation_options"] = ["double_key_size", "quantum_resistant_hash"]
        
        return details
    
    def _estimate_security_margin(self, algorithm_name: str, key_size: int, threat_level: QuantumThreatLevel) -> float:
        """Estimate security margin in years."""
        # Base estimates for quantum computer development
        base_timeline = {
            QuantumThreatLevel.CRITICAL: 1.0,  # 1 year or less
            QuantumThreatLevel.HIGH: 5.0,      # ~5 years
            QuantumThreatLevel.MEDIUM: 10.0,   # ~10 years
            QuantumThreatLevel.LOW: 15.0,      # ~15 years
            QuantumThreatLevel.MINIMAL: 20.0   # 20+ years
        }
        
        base_margin = base_timeline.get(threat_level, 10.0)
        
        # Adjust based on algorithm specifics
        algorithm_lower = algorithm_name.lower()
        
        if "rsa" in algorithm_lower:
            if key_size <= 1024:
                base_margin *= 0.5  # Significantly less secure
            elif key_size >= 4096:
                base_margin *= 1.2  # Slightly more secure
        
        return max(0.0, base_margin)
    
    def _get_replacement_recommendations(self, algorithm_name: str, usage_context: str,
                                       threat_level: QuantumThreatLevel) -> List[PostQuantumAlgorithm]:
        """Get post-quantum replacement recommendations."""
        recommendations = []
        
        # Get primary recommendation
        primary_rec = recommend_post_quantum_algorithm(algorithm_name, usage_context)
        if primary_rec:
            recommendations.append(primary_rec)
        
        # Add additional recommendations based on threat level
        if threat_level in [QuantumThreatLevel.CRITICAL, QuantumThreatLevel.HIGH]:
            # Add high-security alternatives
            if "encryption" in usage_context.lower() or "kem" in usage_context.lower():
                recommendations.extend([PostQuantumAlgorithm.KYBER_1024])
            elif "signature" in usage_context.lower() or "auth" in usage_context.lower():
                recommendations.extend([PostQuantumAlgorithm.DILITHIUM_5, PostQuantumAlgorithm.SPHINCS_PLUS])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_analysis_confidence(self, algorithm_name: str, algorithm_type: str) -> float:
        """Calculate confidence level for analysis."""
        confidence = 0.8  # Base confidence
        
        # Higher confidence for well-known algorithms
        well_known = ["rsa", "aes", "sha", "ecdsa", "des", "md5"]
        if any(known in algorithm_name.lower() for known in well_known):
            confidence += 0.15
        
        # Higher confidence for standard algorithm types
        if algorithm_type in ["symmetric", "asymmetric", "hash", "signature"]:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _get_threat_level_weight(self, threat_level: QuantumThreatLevel) -> float:
        """Get numerical weight for threat level."""
        weights = {
            QuantumThreatLevel.CRITICAL: 1.0,
            QuantumThreatLevel.HIGH: 0.8,
            QuantumThreatLevel.MEDIUM: 0.6,
            QuantumThreatLevel.LOW: 0.4,
            QuantumThreatLevel.MINIMAL: 0.2
        }
        return weights.get(threat_level, 0.5)
    
    def _determine_migration_urgency(self, critical_count: int, high_risk_count: int, overall_risk: float) -> str:
        """Determine migration urgency level."""
        if critical_count > 0 or overall_risk > 0.8:
            return "immediate"
        elif high_risk_count > 0 or overall_risk > 0.6:
            return "high"
        elif overall_risk > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_threat_timeline(self) -> Dict[str, datetime]:
        """Generate quantum threat timeline."""
        current_time = datetime.now(UTC)
        
        return {
            "quantum_advantage_demonstration": current_time + timedelta(days=365 * 2),
            "cryptographically_relevant_quantum_computer": current_time + timedelta(days=365 * 8),
            "large_scale_quantum_attacks": current_time + timedelta(days=365 * 12),
            "migration_deadline_recommended": current_time + timedelta(days=365 * 5)
        }
    
    async def _generate_assessment_recommendations(self, vulnerable_algs: List[AlgorithmAnalysis],
                                                 secure_algs: List[AlgorithmAnalysis],
                                                 risk_score: float) -> List[str]:
        """Generate assessment recommendations."""
        recommendations = []
        
        if not vulnerable_algs:
            recommendations.append("No quantum-vulnerable algorithms detected")
            recommendations.append("System appears quantum-ready with current cryptography")
            return recommendations
        
        # Risk-based recommendations
        if risk_score > 0.8:
            recommendations.append("URGENT: Immediate migration required for critical vulnerabilities")
            recommendations.append("Consider emergency security protocols")
        elif risk_score > 0.6:
            recommendations.append("HIGH PRIORITY: Plan migration within 6 months")
            recommendations.append("Implement hybrid classical-quantum security")
        
        # Algorithm-specific recommendations
        critical_algs = [a for a in vulnerable_algs if a.threat_level == QuantumThreatLevel.CRITICAL]
        if critical_algs:
            recommendations.append(f"Replace {len(critical_algs)} critical algorithms immediately")
        
        # General recommendations
        recommendations.append("Implement post-quantum cryptography testing environment")
        recommendations.append("Establish quantum security monitoring")
        recommendations.append("Plan for hybrid transition period")
        
        return recommendations
    
    def _generate_migration_strategies(self, assessment: VulnerabilityAssessment) -> List[str]:
        """Generate migration strategies."""
        strategies = []
        
        if assessment.critical_vulnerabilities > 0:
            strategies.append("Emergency migration for critical vulnerabilities")
        
        if len(assessment.vulnerable_algorithms) > 10:
            strategies.append("Phased migration approach for large number of algorithms")
        else:
            strategies.append("Comprehensive migration approach")
        
        strategies.extend([
            "Hybrid classical-quantum security during transition",
            "Backward compatibility maintenance",
            "Performance testing and optimization"
        ])
        
        return strategies
    
    def _create_risk_mitigation_plan(self, assessment: VulnerabilityAssessment) -> Dict[str, Any]:
        """Create risk mitigation plan."""
        return {
            "immediate_actions": [
                "Identify and isolate critical cryptographic components",
                "Implement additional monitoring for vulnerable algorithms",
                "Prepare emergency migration procedures"
            ],
            "short_term_actions": [
                "Deploy hybrid security measures",
                "Begin post-quantum algorithm testing",
                "Update security policies"
            ],
            "long_term_actions": [
                "Complete post-quantum migration",
                "Establish quantum-ready infrastructure",
                "Implement continuous quantum threat monitoring"
            ],
            "contingency_plans": [
                "Rapid algorithm replacement procedures",
                "Emergency fallback to classical algorithms",
                "Incident response for quantum attacks"
            ]
        }
    
    def _infer_usage_context(self, match_text: str, full_content: str) -> str:
        """Infer algorithm usage context from code."""
        context_indicators = {
            "authentication": ["auth", "login", "verify", "credential"],
            "encryption": ["encrypt", "decrypt", "cipher", "secure"],
            "signature": ["sign", "verify", "signature", "digital"],
            "key_exchange": ["exchange", "handshake", "negotiate"],
            "hash": ["hash", "digest", "checksum", "integrity"]
        }
        
        # Search around the match for context clues
        match_area = full_content.lower()
        
        for context, indicators in context_indicators.items():
            if any(indicator in match_area for indicator in indicators):
                return context
        
        return "general"