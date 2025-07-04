"""
Quantum Architecture - TASK_68 Phase 1 Architecture & Design

Type-safe quantum computing types with post-quantum cryptography, quantum interface preparation,
and future-proof security architecture for enterprise automation systems.

Architecture: Type Safety + Design by Contract + Post-Quantum Cryptography + Quantum Interface Design
Performance: <50ms analysis, <100ms migration planning, <200ms quantum simulation
Security: Post-quantum algorithms, quantum-safe key management, future-proof cryptographic systems
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple, Union, NewType
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, UTC
from pathlib import Path
import uuid
import hashlib

from ..core.contracts import require, ensure
from ..core.either import Either

# Branded types for quantum security
QuantumKeyId = NewType('QuantumKeyId', str)
QuantumSessionId = NewType('QuantumSessionId', str) 
PostQuantumAlgorithmId = NewType('PostQuantumAlgorithmId', str)
QuantumCircuitId = NewType('QuantumCircuitId', str)
CryptographicAssetId = NewType('CryptographicAssetId', str)

class PostQuantumAlgorithm(Enum):
    """NIST-standardized post-quantum cryptographic algorithms."""
    # Key Encapsulation Mechanisms (KEMs)
    KYBER_512 = "kyber-512"        # NIST Level 1
    KYBER_768 = "kyber-768"        # NIST Level 3
    KYBER_1024 = "kyber-1024"      # NIST Level 5
    
    # Digital Signatures
    DILITHIUM_2 = "dilithium-2"    # NIST Level 1
    DILITHIUM_3 = "dilithium-3"    # NIST Level 2
    DILITHIUM_5 = "dilithium-5"    # NIST Level 4
    
    FALCON_512 = "falcon-512"      # NIST Level 1
    FALCON_1024 = "falcon-1024"    # NIST Level 5
    
    # Hash-based signatures
    SPHINCS_PLUS = "sphincs-plus"  # Stateless hash-based

class QuantumThreatLevel(Enum):
    """Quantum computing threat assessment levels."""
    MINIMAL = "minimal"        # Current classical systems adequate
    LOW = "low"               # Quantum threat emerging (5-10 years)
    MEDIUM = "medium"         # Quantum threat imminent (2-5 years)
    HIGH = "high"             # Quantum threat critical (1-2 years)
    CRITICAL = "critical"     # Quantum computers already capable

class CryptographicStrength(Enum):
    """Cryptographic strength levels against quantum attacks."""
    CLASSICAL_ONLY = "classical_only"      # No quantum resistance
    QUANTUM_RESISTANT = "quantum_resistant" # Post-quantum algorithms
    QUANTUM_SAFE = "quantum_safe"          # Quantum + classical hybrid
    QUANTUM_NATIVE = "quantum_native"      # Native quantum cryptography

class QuantumSecurityPolicy(Enum):
    """Quantum security policy levels."""
    LEGACY = "legacy"                # Current systems only
    HYBRID = "hybrid"               # Classical + post-quantum
    POST_QUANTUM = "post_quantum"   # Post-quantum only
    QUANTUM_READY = "quantum_ready" # Full quantum preparation

class QuantumError(Exception):
    """Quantum system errors with detailed categorization."""
    
    def __init__(self, message: str, error_code: str = "QUANTUM_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now(UTC)
    
    @classmethod
    def algorithm_not_supported(cls, algorithm: str) -> 'QuantumError':
        return cls(
            f"Post-quantum algorithm not supported: {algorithm}",
            "ALGORITHM_NOT_SUPPORTED",
            {"algorithm": algorithm}
        )
    
    @classmethod
    def migration_failed(cls, asset_id: str) -> 'QuantumError':
        return cls(
            f"Cryptographic asset migration failed: {asset_id}",
            "MIGRATION_FAILED",
            {"asset_id": asset_id}
        )
    
    @classmethod
    def quantum_interface_error(cls, operation: str) -> 'QuantumError':
        return cls(
            f"Quantum interface operation failed: {operation}",
            "QUANTUM_INTERFACE_ERROR",
            {"operation": operation}
        )

@dataclass(frozen=True)
class CryptographicAsset:
    """Cryptographic asset with quantum vulnerability assessment."""
    asset_id: CryptographicAssetId
    asset_type: str  # key|certificate|signature|encryption
    algorithm: str
    key_size: int
    created_at: datetime
    usage_context: str
    quantum_vulnerable: bool
    threat_assessment: QuantumThreatLevel
    migration_priority: int  # 1-5, 5 being highest
    replacement_algorithm: Optional[PostQuantumAlgorithm] = None
    
    @require(lambda self: self.key_size > 0)
    @require(lambda self: 1 <= self.migration_priority <= 5)
    def __post_init__(self):
        pass
    
    def needs_immediate_migration(self) -> bool:
        """Check if asset needs immediate quantum migration."""
        return (
            self.quantum_vulnerable and 
            self.threat_assessment in [QuantumThreatLevel.HIGH, QuantumThreatLevel.CRITICAL] and
            self.migration_priority >= 4
        )
    
    def get_quantum_risk_score(self) -> float:
        """Calculate quantum risk score (0.0 to 1.0)."""
        if not self.quantum_vulnerable:
            return 0.0
        
        threat_weights = {
            QuantumThreatLevel.MINIMAL: 0.1,
            QuantumThreatLevel.LOW: 0.3,
            QuantumThreatLevel.MEDIUM: 0.6,
            QuantumThreatLevel.HIGH: 0.8,
            QuantumThreatLevel.CRITICAL: 1.0
        }
        
        base_risk = threat_weights.get(self.threat_assessment, 0.5)
        priority_weight = self.migration_priority / 5.0
        
        return min(1.0, base_risk * priority_weight)

@dataclass(frozen=True)
class PostQuantumMigrationPlan:
    """Migration plan for transitioning to post-quantum cryptography."""
    plan_id: str
    target_assets: List[CryptographicAssetId]
    migration_strategy: str  # hybrid|full_replacement|gradual
    target_algorithms: Dict[str, PostQuantumAlgorithm]
    estimated_duration: timedelta
    risk_assessment: Dict[str, float]
    compatibility_requirements: List[str]
    rollback_strategy: str
    validation_criteria: List[str]
    created_at: datetime
    
    @require(lambda self: len(self.target_assets) > 0)
    @require(lambda self: len(self.target_algorithms) > 0)
    def __post_init__(self):
        pass
    
    def get_migration_phases(self) -> List[Dict[str, Any]]:
        """Get ordered migration phases."""
        phases = []
        
        if self.migration_strategy == "gradual":
            # Phase 1: High-priority critical assets
            phases.append({
                "phase": 1,
                "description": "Critical asset migration",
                "asset_count": len([a for a in self.target_assets if self._is_critical_asset(a)]),
                "estimated_duration": self.estimated_duration * 0.3
            })
            
            # Phase 2: Medium-priority assets
            phases.append({
                "phase": 2,
                "description": "Medium-priority asset migration",
                "asset_count": len([a for a in self.target_assets if self._is_medium_priority_asset(a)]),
                "estimated_duration": self.estimated_duration * 0.5
            })
            
            # Phase 3: Low-priority assets
            phases.append({
                "phase": 3,
                "description": "Low-priority asset migration",
                "asset_count": len([a for a in self.target_assets if self._is_low_priority_asset(a)]),
                "estimated_duration": self.estimated_duration * 0.2
            })
        
        elif self.migration_strategy == "hybrid":
            # Single phase with hybrid implementation
            phases.append({
                "phase": 1,
                "description": "Hybrid classical-quantum deployment",
                "asset_count": len(self.target_assets),
                "estimated_duration": self.estimated_duration
            })
        
        else:  # full_replacement
            # Single phase with full replacement
            phases.append({
                "phase": 1,
                "description": "Complete post-quantum replacement",
                "asset_count": len(self.target_assets),
                "estimated_duration": self.estimated_duration
            })
        
        return phases
    
    def _is_critical_asset(self, asset_id: CryptographicAssetId) -> bool:
        """Check if asset is critical priority."""
        # Placeholder implementation
        return True  # Would check actual asset priority
    
    def _is_medium_priority_asset(self, asset_id: CryptographicAssetId) -> bool:
        """Check if asset is medium priority."""
        return True
    
    def _is_low_priority_asset(self, asset_id: CryptographicAssetId) -> bool:
        """Check if asset is low priority."""
        return True

@dataclass(frozen=True)
class QuantumReadinessAssessment:
    """Comprehensive quantum readiness assessment results."""
    assessment_id: str
    scope: str  # system|application|cryptography|protocols
    overall_readiness_score: float  # 0.0 to 1.0
    quantum_vulnerable_assets: List[CryptographicAsset]
    threat_timeline_estimate: Dict[str, datetime]
    migration_recommendations: List[str]
    compliance_status: Dict[str, bool]
    risk_factors: Dict[str, float]
    estimated_migration_cost: Optional[float] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    @require(lambda self: 0.0 <= self.overall_readiness_score <= 1.0)
    @require(lambda self: all(0.0 <= risk <= 1.0 for risk in self.risk_factors.values()))
    def __post_init__(self):
        pass
    
    def get_readiness_level(self) -> str:
        """Get categorical readiness level."""
        if self.overall_readiness_score >= 0.8:
            return "quantum_ready"
        elif self.overall_readiness_score >= 0.6:
            return "mostly_ready"
        elif self.overall_readiness_score >= 0.4:
            return "partially_ready"
        elif self.overall_readiness_score >= 0.2:
            return "minimal_readiness"
        else:
            return "not_ready"
    
    def get_critical_vulnerabilities(self) -> List[CryptographicAsset]:
        """Get assets with critical quantum vulnerabilities."""
        return [
            asset for asset in self.quantum_vulnerable_assets
            if asset.threat_assessment in [QuantumThreatLevel.HIGH, QuantumThreatLevel.CRITICAL]
        ]
    
    def estimate_quantum_threat_timeline(self) -> Dict[str, int]:
        """Estimate quantum threat timeline in years."""
        current_year = datetime.now(UTC).year
        
        return {
            "cryptographically_relevant_quantum_computer": 
                max(2030 - current_year, 1),  # Conservative estimate
            "large_scale_quantum_attacks": 
                max(2035 - current_year, 2),  # More conservative
            "quantum_supremacy_in_cryptography": 
                max(2040 - current_year, 5)   # Long-term estimate
        }

@dataclass(frozen=True)
class QuantumInterface:
    """Quantum computing interface specification."""
    interface_id: str
    interface_type: str  # computing|communication|simulation|hybrid
    quantum_platform: str  # ibm|google|amazon|microsoft|universal
    protocol_version: str
    supported_operations: List[str]
    qubit_capacity: Optional[int]
    gate_fidelity: Optional[float]
    coherence_time: Optional[float]  # microseconds
    connectivity_map: Dict[str, Any]
    error_correction_enabled: bool
    classical_integration: bool
    
    @require(lambda self: len(self.supported_operations) > 0)
    @require(lambda self: self.qubit_capacity is None or self.qubit_capacity > 0)
    @require(lambda self: self.gate_fidelity is None or 0.0 <= self.gate_fidelity <= 1.0)
    def __post_init__(self):
        pass
    
    def is_suitable_for_algorithm(self, algorithm_type: str, required_qubits: int) -> bool:
        """Check if interface is suitable for specific quantum algorithm."""
        if self.qubit_capacity and required_qubits > self.qubit_capacity:
            return False
        
        algorithm_requirements = {
            "shor": ["quantum_fourier_transform", "modular_arithmetic"],
            "grover": ["amplitude_amplification", "oracle_queries"],
            "quantum_ml": ["variational_circuits", "parameter_optimization"],
            "optimization": ["qaoa", "variational_quantum_eigensolver"]
        }
        
        required_ops = algorithm_requirements.get(algorithm_type, [])
        return all(op in self.supported_operations for op in required_ops)

@dataclass(frozen=True)
class QuantumSimulationResult:
    """Results from quantum algorithm simulation."""
    simulation_id: str
    algorithm_type: str
    qubit_count: int
    circuit_depth: int
    execution_time: float  # seconds
    measurement_results: Dict[str, int]
    fidelity_estimate: Optional[float]
    success_probability: float
    quantum_volume: Optional[int]
    noise_model_applied: Optional[str] = None
    
    @require(lambda self: self.qubit_count > 0)
    @require(lambda self: self.circuit_depth > 0)
    @require(lambda self: 0.0 <= self.success_probability <= 1.0)
    @require(lambda self: self.fidelity_estimate is None or 0.0 <= self.fidelity_estimate <= 1.0)
    def __post_init__(self):
        pass
    
    def get_measurement_distribution(self) -> Dict[str, float]:
        """Get normalized measurement probability distribution."""
        total_shots = sum(self.measurement_results.values())
        if total_shots == 0:
            return {}
        
        return {
            state: count / total_shots 
            for state, count in self.measurement_results.items()
        }
    
    def calculate_quantum_advantage(self, classical_time: float) -> Optional[float]:
        """Calculate quantum advantage over classical execution."""
        if classical_time <= 0 or self.execution_time <= 0:
            return None
        
        return classical_time / self.execution_time

@dataclass(frozen=True)
class QuantumSecurityConfiguration:
    """Quantum security system configuration."""
    config_id: str
    security_policy: QuantumSecurityPolicy
    enabled_algorithms: Set[PostQuantumAlgorithm]
    key_management_mode: str  # classical|quantum|hybrid
    distribution_protocol: str  # qkd|classical|hybrid
    monitoring_enabled: bool
    threat_detection_enabled: bool
    incident_response_enabled: bool
    compliance_frameworks: List[str]
    
    @require(lambda self: len(self.enabled_algorithms) > 0)
    def __post_init__(self):
        pass
    
    def is_quantum_safe(self) -> bool:
        """Check if configuration provides quantum safety."""
        return (
            self.security_policy in [QuantumSecurityPolicy.POST_QUANTUM, QuantumSecurityPolicy.QUANTUM_READY] and
            len(self.enabled_algorithms) > 0 and
            self.threat_detection_enabled
        )
    
    def get_security_level(self) -> CryptographicStrength:
        """Get overall cryptographic strength level."""
        if self.security_policy == QuantumSecurityPolicy.LEGACY:
            return CryptographicStrength.CLASSICAL_ONLY
        elif self.security_policy == QuantumSecurityPolicy.HYBRID:
            return CryptographicStrength.QUANTUM_SAFE
        elif self.security_policy == QuantumSecurityPolicy.POST_QUANTUM:
            return CryptographicStrength.QUANTUM_RESISTANT
        elif self.security_policy == QuantumSecurityPolicy.QUANTUM_READY:
            return CryptographicStrength.QUANTUM_NATIVE
        else:
            return CryptographicStrength.CLASSICAL_ONLY

# Utility functions for quantum operations
def generate_quantum_key_id() -> QuantumKeyId:
    """Generate unique quantum key ID."""
    return QuantumKeyId(f"qk_{uuid.uuid4().hex}")

def generate_quantum_session_id() -> QuantumSessionId:
    """Generate unique quantum session ID."""
    return QuantumSessionId(f"qs_{uuid.uuid4().hex}")

def generate_circuit_id() -> QuantumCircuitId:
    """Generate unique quantum circuit ID."""
    return QuantumCircuitId(f"qc_{uuid.uuid4().hex}")

def assess_algorithm_quantum_vulnerability(algorithm: str, key_size: int) -> Tuple[bool, QuantumThreatLevel]:
    """Assess quantum vulnerability of cryptographic algorithm."""
    vulnerable_algorithms = {
        "rsa": {
            1024: QuantumThreatLevel.CRITICAL,
            2048: QuantumThreatLevel.HIGH,
            3072: QuantumThreatLevel.MEDIUM,
            4096: QuantumThreatLevel.LOW
        },
        "ecdsa": {
            256: QuantumThreatLevel.HIGH,
            384: QuantumThreatLevel.MEDIUM,
            521: QuantumThreatLevel.LOW
        },
        "dh": {
            1024: QuantumThreatLevel.CRITICAL,
            2048: QuantumThreatLevel.HIGH,
            3072: QuantumThreatLevel.MEDIUM
        }
    }
    
    algorithm_lower = algorithm.lower()
    
    if algorithm_lower in vulnerable_algorithms:
        size_threats = vulnerable_algorithms[algorithm_lower]
        # Find the appropriate threat level for the key size
        for size, threat in sorted(size_threats.items()):
            if key_size <= size:
                return True, threat
        return True, QuantumThreatLevel.MINIMAL
    
    # Post-quantum algorithms are not vulnerable
    post_quantum_names = [alg.value for alg in PostQuantumAlgorithm]
    if any(pq_name in algorithm_lower for pq_name in post_quantum_names):
        return False, QuantumThreatLevel.MINIMAL
    
    # Unknown algorithm - assume vulnerable
    return True, QuantumThreatLevel.MEDIUM

def calculate_migration_priority(asset: CryptographicAsset) -> int:
    """Calculate migration priority for cryptographic asset."""
    base_priority = 1
    
    # High-risk algorithms get higher priority
    if asset.threat_assessment in [QuantumThreatLevel.CRITICAL, QuantumThreatLevel.HIGH]:
        base_priority += 2
    elif asset.threat_assessment == QuantumThreatLevel.MEDIUM:
        base_priority += 1
    
    # Critical usage contexts get higher priority
    critical_contexts = ["authentication", "key_exchange", "digital_signature", "encryption"]
    if any(context in asset.usage_context.lower() for context in critical_contexts):
        base_priority += 1
    
    # Recently created assets may have longer lifetime
    age_days = (datetime.now(UTC) - asset.created_at).days
    if age_days < 30:  # Recently deployed
        base_priority += 1
    
    return min(5, base_priority)

def recommend_post_quantum_algorithm(classical_algorithm: str, use_case: str) -> Optional[PostQuantumAlgorithm]:
    """Recommend appropriate post-quantum algorithm replacement."""
    
    use_case_lower = use_case.lower()
    
    # Check for encryption-related use cases
    if (use_case_lower in ["encryption", "key_exchange", "kem"] or
        "encryption" in use_case_lower or "tls" in use_case_lower):
        # Key Encapsulation Mechanisms
        if "high_security" in use_case_lower:
            return PostQuantumAlgorithm.KYBER_1024
        elif "performance" in use_case_lower:
            return PostQuantumAlgorithm.KYBER_512
        else:
            return PostQuantumAlgorithm.KYBER_768
    
    # Check for signature/authentication-related use cases
    elif (use_case_lower in ["signature", "authentication", "digital_signature"] or
          "authentication" in use_case_lower or "signing" in use_case_lower or 
          "certificate" in use_case_lower or "ssh" in use_case_lower):
        # Digital Signatures
        if "performance" in use_case_lower:
            return PostQuantumAlgorithm.FALCON_512
        elif "compatibility" in use_case_lower:
            return PostQuantumAlgorithm.DILITHIUM_2
        elif "high_security" in use_case_lower:
            return PostQuantumAlgorithm.DILITHIUM_5
        else:
            return PostQuantumAlgorithm.DILITHIUM_3
    
    # Check for hash/integrity use cases
    elif (use_case_lower in ["hash", "integrity", "long_term"] or
          "message_authentication" in use_case_lower or "hmac" in use_case_lower):
        # Hash-based signatures for long-term security
        return PostQuantumAlgorithm.SPHINCS_PLUS
    
    # Default fallback - recommend based on classical algorithm
    if classical_algorithm.lower() in ["rsa", "ecdsa", "dsa"]:
        # Signature algorithms get signature recommendations
        return PostQuantumAlgorithm.DILITHIUM_3
    elif classical_algorithm.lower() in ["aes", "des"]:
        # Symmetric algorithms get KEM recommendations
        return PostQuantumAlgorithm.KYBER_768
    
    # If we can't determine, provide a safe default
    return PostQuantumAlgorithm.KYBER_768

def create_default_quantum_config() -> QuantumSecurityConfiguration:
    """Create default quantum security configuration."""
    return QuantumSecurityConfiguration(
        config_id=f"qsc_{uuid.uuid4().hex}",
        security_policy=QuantumSecurityPolicy.HYBRID,
        enabled_algorithms={
            PostQuantumAlgorithm.KYBER_768,
            PostQuantumAlgorithm.DILITHIUM_3,
            PostQuantumAlgorithm.FALCON_512
        },
        key_management_mode="hybrid",
        distribution_protocol="hybrid",
        monitoring_enabled=True,
        threat_detection_enabled=True,
        incident_response_enabled=True,
        compliance_frameworks=["NIST", "FIPS", "Common Criteria"]
    )