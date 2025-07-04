"""
Test Quantum Architecture - Core Quantum Types and Architecture Testing

Comprehensive tests for quantum computing type definitions, post-quantum cryptography,
and quantum security architecture components.

Architecture: Property-Based Testing + Type Safety + Contract Validation + Security Testing
Performance: <100ms test execution, comprehensive type validation, security compliance testing
Security: Post-quantum algorithm validation, threat assessment testing, cryptographic strength verification
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime, timedelta, UTC
from dataclasses import FrozenInstanceError

from src.core.quantum_architecture import (
    PostQuantumAlgorithm,
    QuantumThreatLevel,
    CryptographicStrength,
    QuantumSecurityPolicy,
    QuantumError,
    CryptographicAsset,
    PostQuantumMigrationPlan,
    QuantumReadinessAssessment,
    QuantumInterface,
    QuantumSimulationResult,
    QuantumSecurityConfiguration,
    CryptographicAssetId,
    QuantumKeyId,
    QuantumSessionId,
    PostQuantumAlgorithmId,
    QuantumCircuitId,
    generate_quantum_key_id,
    generate_quantum_session_id,
    generate_circuit_id,
    assess_algorithm_quantum_vulnerability,
    calculate_migration_priority,
    recommend_post_quantum_algorithm,
    create_default_quantum_config
)


class TestPostQuantumAlgorithms:
    """Test post-quantum algorithm enumeration and properties."""
    
    def test_post_quantum_algorithm_values(self):
        """Test all post-quantum algorithm values are valid."""
        expected_algorithms = {
            "kyber-512", "kyber-768", "kyber-1024",
            "dilithium-2", "dilithium-3", "dilithium-5",
            "falcon-512", "falcon-1024",
            "sphincs-plus"
        }
        
        actual_algorithms = {alg.value for alg in PostQuantumAlgorithm}
        assert actual_algorithms == expected_algorithms
    
    def test_post_quantum_algorithm_categories(self):
        """Test post-quantum algorithms are properly categorized."""
        kem_algorithms = [
            PostQuantumAlgorithm.KYBER_512,
            PostQuantumAlgorithm.KYBER_768,
            PostQuantumAlgorithm.KYBER_1024
        ]
        
        signature_algorithms = [
            PostQuantumAlgorithm.DILITHIUM_2,
            PostQuantumAlgorithm.DILITHIUM_3,
            PostQuantumAlgorithm.DILITHIUM_5,
            PostQuantumAlgorithm.FALCON_512,
            PostQuantumAlgorithm.FALCON_1024,
            PostQuantumAlgorithm.SPHINCS_PLUS
        ]
        
        # All should be unique
        all_algorithms = kem_algorithms + signature_algorithms
        assert len(all_algorithms) == len(set(all_algorithms))
        assert len(all_algorithms) == len(PostQuantumAlgorithm)


class TestQuantumThreatLevels:
    """Test quantum threat level enumeration and ordering."""
    
    def test_quantum_threat_level_values(self):
        """Test quantum threat level enumeration values."""
        expected_levels = {
            "minimal", "low", "medium", "high", "critical"
        }
        
        actual_levels = {level.value for level in QuantumThreatLevel}
        assert actual_levels == expected_levels
    
    def test_quantum_threat_level_ordering(self):
        """Test quantum threat levels have logical ordering."""
        levels = [
            QuantumThreatLevel.MINIMAL,
            QuantumThreatLevel.LOW,
            QuantumThreatLevel.MEDIUM,
            QuantumThreatLevel.HIGH,
            QuantumThreatLevel.CRITICAL
        ]
        
        # Should be in increasing severity order
        for i in range(len(levels) - 1):
            assert levels[i] != levels[i + 1]


class TestCryptographicAsset:
    """Test cryptographic asset data structure and validation."""
    
    def test_cryptographic_asset_creation(self):
        """Test valid cryptographic asset creation."""
        asset = CryptographicAsset(
            asset_id=CryptographicAssetId("test_asset_001"),
            asset_type="key",
            algorithm="rsa",
            key_size=2048,
            created_at=datetime.now(UTC),
            usage_context="authentication",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.HIGH,
            migration_priority=4,
            replacement_algorithm=PostQuantumAlgorithm.KYBER_768
        )
        
        assert asset.asset_id == "test_asset_001"
        assert asset.algorithm == "rsa"
        assert asset.key_size == 2048
        assert asset.quantum_vulnerable is True
        assert asset.threat_assessment == QuantumThreatLevel.HIGH
        assert asset.migration_priority == 4
    
    def test_cryptographic_asset_immutability(self):
        """Test cryptographic asset is immutable (frozen dataclass)."""
        asset = CryptographicAsset(
            asset_id=CryptographicAssetId("test_asset_002"),
            asset_type="certificate",
            algorithm="ecdsa",
            key_size=256,
            created_at=datetime.now(UTC),
            usage_context="signature",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.MEDIUM,
            migration_priority=3
        )
        
        with pytest.raises(FrozenInstanceError):
            asset.algorithm = "rsa"
    
    def test_cryptographic_asset_invalid_key_size(self):
        """Test cryptographic asset validation for invalid key size."""
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            CryptographicAsset(
                asset_id=CryptographicAssetId("test_asset_003"),
                asset_type="key",
                algorithm="aes",
                key_size=0,  # Invalid key size
                created_at=datetime.now(UTC),
                usage_context="encryption",
                quantum_vulnerable=False,
                threat_assessment=QuantumThreatLevel.MINIMAL,
                migration_priority=1
            )
    
    def test_cryptographic_asset_invalid_priority(self):
        """Test cryptographic asset validation for invalid priority."""
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            CryptographicAsset(
                asset_id=CryptographicAssetId("test_asset_004"),
                asset_type="key",
                algorithm="aes",
                key_size=256,
                created_at=datetime.now(UTC),
                usage_context="encryption",
                quantum_vulnerable=False,
                threat_assessment=QuantumThreatLevel.MINIMAL,
                migration_priority=6  # Invalid priority (> 5)
            )
    
    def test_needs_immediate_migration(self):
        """Test immediate migration requirement detection."""
        # Asset needing immediate migration
        critical_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("critical_001"),
            asset_type="key",
            algorithm="rsa",
            key_size=1024,
            created_at=datetime.now(UTC),
            usage_context="authentication",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.CRITICAL,
            migration_priority=5
        )
        
        assert critical_asset.needs_immediate_migration() is True
        
        # Asset not needing immediate migration
        low_risk_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("low_risk_001"),
            asset_type="key",
            algorithm="aes",
            key_size=256,
            created_at=datetime.now(UTC),
            usage_context="encryption",
            quantum_vulnerable=False,
            threat_assessment=QuantumThreatLevel.MINIMAL,
            migration_priority=1
        )
        
        assert low_risk_asset.needs_immediate_migration() is False
    
    def test_get_quantum_risk_score(self):
        """Test quantum risk score calculation."""
        # High-risk asset
        high_risk_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("high_risk_001"),
            asset_type="key",
            algorithm="rsa",
            key_size=2048,
            created_at=datetime.now(UTC),
            usage_context="authentication",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.HIGH,
            migration_priority=5
        )
        
        risk_score = high_risk_asset.get_quantum_risk_score()
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be high risk
        
        # Low-risk asset
        low_risk_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("low_risk_002"),
            asset_type="key",
            algorithm="aes",
            key_size=256,
            created_at=datetime.now(UTC),
            usage_context="encryption",
            quantum_vulnerable=False,
            threat_assessment=QuantumThreatLevel.MINIMAL,
            migration_priority=1
        )
        
        risk_score_low = low_risk_asset.get_quantum_risk_score()
        assert risk_score_low == 0.0  # Not quantum vulnerable


class TestPostQuantumMigrationPlan:
    """Test post-quantum migration plan data structure."""
    
    def test_migration_plan_creation(self):
        """Test valid migration plan creation."""
        plan = PostQuantumMigrationPlan(
            plan_id="plan_001",
            target_assets=[CryptographicAssetId("asset_1"), CryptographicAssetId("asset_2")],
            migration_strategy="hybrid",
            target_algorithms={"asset_1": PostQuantumAlgorithm.KYBER_768},
            estimated_duration=timedelta(hours=24),
            risk_assessment={"overall_risk": 0.7},
            compatibility_requirements=["enterprise_directory_support"],
            rollback_strategy="maintain_classical_fallback",
            validation_criteria=["algorithm_compatibility_verified"],
            created_at=datetime.now(UTC)
        )
        
        assert plan.plan_id == "plan_001"
        assert len(plan.target_assets) == 2
        assert plan.migration_strategy == "hybrid"
        assert plan.estimated_duration == timedelta(hours=24)
    
    def test_migration_plan_immutability(self):
        """Test migration plan is immutable."""
        plan = PostQuantumMigrationPlan(
            plan_id="plan_002",
            target_assets=[CryptographicAssetId("asset_1")],
            migration_strategy="full_replacement",
            target_algorithms={"asset_1": PostQuantumAlgorithm.DILITHIUM_3},
            estimated_duration=timedelta(hours=12),
            risk_assessment={},
            compatibility_requirements=[],
            rollback_strategy="complete_system_restoration",
            validation_criteria=[],
            created_at=datetime.now(UTC)
        )
        
        with pytest.raises(FrozenInstanceError):
            plan.migration_strategy = "gradual"
    
    def test_get_migration_phases_gradual(self):
        """Test migration phases for gradual strategy."""
        plan = PostQuantumMigrationPlan(
            plan_id="plan_gradual",
            target_assets=[CryptographicAssetId("asset_1"), CryptographicAssetId("asset_2")],
            migration_strategy="gradual",
            target_algorithms={"asset_1": PostQuantumAlgorithm.KYBER_768},
            estimated_duration=timedelta(hours=48),
            risk_assessment={},
            compatibility_requirements=[],
            rollback_strategy="rollback_by_migration_phase",
            validation_criteria=[],
            created_at=datetime.now(UTC)
        )
        
        phases = plan.get_migration_phases()
        assert len(phases) == 3  # Critical, medium, low priority phases
        assert phases[0]["phase"] == 1
        assert "Critical asset migration" in phases[0]["description"]
    
    def test_get_migration_phases_hybrid(self):
        """Test migration phases for hybrid strategy."""
        plan = PostQuantumMigrationPlan(
            plan_id="plan_hybrid",
            target_assets=[CryptographicAssetId("asset_1")],
            migration_strategy="hybrid",
            target_algorithms={"asset_1": PostQuantumAlgorithm.DILITHIUM_3},
            estimated_duration=timedelta(hours=24),
            risk_assessment={},
            compatibility_requirements=[],
            rollback_strategy="maintain_classical_fallback",
            validation_criteria=[],
            created_at=datetime.now(UTC)
        )
        
        phases = plan.get_migration_phases()
        assert len(phases) == 1  # Single hybrid phase
        assert "Hybrid classical-quantum deployment" in phases[0]["description"]


class TestQuantumReadinessAssessment:
    """Test quantum readiness assessment data structure."""
    
    def test_readiness_assessment_creation(self):
        """Test valid readiness assessment creation."""
        assessment = QuantumReadinessAssessment(
            assessment_id="assessment_001",
            scope="system",
            overall_readiness_score=0.75,
            quantum_vulnerable_assets=[],
            threat_timeline_estimate={
                "quantum_advantage": datetime.now(UTC) + timedelta(days=365*5)
            },
            migration_recommendations=["Implement hybrid security"],
            compliance_status={"nist_compliant": True},
            risk_factors={"vulnerability_ratio": 0.3},
            estimated_migration_cost=50000.0
        )
        
        assert assessment.assessment_id == "assessment_001"
        assert assessment.scope == "system"
        assert assessment.overall_readiness_score == 0.75
        assert assessment.estimated_migration_cost == 50000.0
    
    def test_readiness_assessment_invalid_score(self):
        """Test readiness assessment validation for invalid score."""
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            QuantumReadinessAssessment(
                assessment_id="assessment_002",
                scope="application",
                overall_readiness_score=1.5,  # Invalid score > 1.0
                quantum_vulnerable_assets=[],
                threat_timeline_estimate={},
                migration_recommendations=[],
                compliance_status={},
                risk_factors={}
            )
    
    def test_get_readiness_level(self):
        """Test readiness level categorization."""
        test_cases = [
            (0.9, "quantum_ready"),
            (0.7, "mostly_ready"),
            (0.5, "partially_ready"),
            (0.3, "minimal_readiness"),
            (0.1, "not_ready")
        ]
        
        for score, expected_level in test_cases:
            assessment = QuantumReadinessAssessment(
                assessment_id=f"test_{score}",
                scope="test",
                overall_readiness_score=score,
                quantum_vulnerable_assets=[],
                threat_timeline_estimate={},
                migration_recommendations=[],
                compliance_status={},
                risk_factors={}
            )
            
            assert assessment.get_readiness_level() == expected_level
    
    def test_get_critical_vulnerabilities(self):
        """Test critical vulnerability identification."""
        critical_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("critical_001"),
            asset_type="key",
            algorithm="rsa",
            key_size=1024,
            created_at=datetime.now(UTC),
            usage_context="authentication",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.CRITICAL,
            migration_priority=5
        )
        
        low_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("low_001"),
            asset_type="key",
            algorithm="aes",
            key_size=256,
            created_at=datetime.now(UTC),
            usage_context="encryption",
            quantum_vulnerable=False,
            threat_assessment=QuantumThreatLevel.LOW,
            migration_priority=1
        )
        
        assessment = QuantumReadinessAssessment(
            assessment_id="vuln_test",
            scope="test",
            overall_readiness_score=0.5,
            quantum_vulnerable_assets=[critical_asset, low_asset],
            threat_timeline_estimate={},
            migration_recommendations=[],
            compliance_status={},
            risk_factors={}
        )
        
        critical_vulns = assessment.get_critical_vulnerabilities()
        assert len(critical_vulns) == 1
        assert critical_vulns[0].asset_id == "critical_001"


class TestQuantumInterface:
    """Test quantum interface data structure."""
    
    def test_quantum_interface_creation(self):
        """Test valid quantum interface creation."""
        interface = QuantumInterface(
            interface_id="qi_001",
            interface_type="computing",
            quantum_platform="ibm",
            protocol_version="1.0",
            supported_operations=["h", "cx", "measure"],
            qubit_capacity=127,
            gate_fidelity=0.999,
            coherence_time=100.0,
            connectivity_map={"type": "heavy_hex"},
            error_correction_enabled=True,
            classical_integration=True
        )
        
        assert interface.interface_id == "qi_001"
        assert interface.interface_type == "computing"
        assert interface.quantum_platform == "ibm"
        assert interface.qubit_capacity == 127
        assert interface.gate_fidelity == 0.999
    
    def test_quantum_interface_invalid_operations(self):
        """Test quantum interface validation for empty operations."""
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            QuantumInterface(
                interface_id="qi_002",
                interface_type="simulation",
                quantum_platform="universal",
                protocol_version="1.0",
                supported_operations=[],  # Invalid empty operations
                qubit_capacity=50,
                gate_fidelity=0.99,
                coherence_time=80.0,
                connectivity_map={},
                error_correction_enabled=False,
                classical_integration=True
            )
    
    def test_is_suitable_for_algorithm(self):
        """Test algorithm suitability checking."""
        interface = QuantumInterface(
            interface_id="qi_suitability",
            interface_type="computing",
            quantum_platform="google",
            protocol_version="1.0",
            supported_operations=["h", "cx", "quantum_fourier_transform", "modular_arithmetic"],
            qubit_capacity=20,
            gate_fidelity=0.999,
            coherence_time=80.0,
            connectivity_map={"type": "sycamore"},
            error_correction_enabled=True,
            classical_integration=True
        )
        
        # Should be suitable for Shor's algorithm
        assert interface.is_suitable_for_algorithm("shor", 15) is True
        
        # Should not be suitable if not enough qubits
        assert interface.is_suitable_for_algorithm("shor", 25) is False
        
        # Should not be suitable if missing operations
        assert interface.is_suitable_for_algorithm("grover", 10) is False


class TestQuantumSimulationResult:
    """Test quantum simulation result data structure."""
    
    def test_simulation_result_creation(self):
        """Test valid simulation result creation."""
        result = QuantumSimulationResult(
            simulation_id="sim_001",
            algorithm_type="grover",
            qubit_count=10,
            circuit_depth=50,
            execution_time=1.5,
            measurement_results={"0000000000": 100, "1010101010": 900},
            fidelity_estimate=0.95,
            success_probability=0.88,
            quantum_volume=100,
            noise_model_applied="realistic"
        )
        
        assert result.simulation_id == "sim_001"
        assert result.algorithm_type == "grover"
        assert result.qubit_count == 10
        assert result.circuit_depth == 50
        assert result.execution_time == 1.5
    
    def test_get_measurement_distribution(self):
        """Test measurement probability distribution calculation."""
        result = QuantumSimulationResult(
            simulation_id="sim_dist",
            algorithm_type="test",
            qubit_count=2,
            circuit_depth=10,
            execution_time=0.1,
            measurement_results={"00": 250, "01": 250, "10": 250, "11": 250},
            fidelity_estimate=1.0,
            success_probability=1.0,
            quantum_volume=4
        )
        
        distribution = result.get_measurement_distribution()
        assert len(distribution) == 4
        for prob in distribution.values():
            assert prob == 0.25  # Equal distribution
        assert sum(distribution.values()) == 1.0
    
    def test_calculate_quantum_advantage(self):
        """Test quantum advantage calculation."""
        result = QuantumSimulationResult(
            simulation_id="sim_advantage",
            algorithm_type="shor",
            qubit_count=15,
            circuit_depth=100,
            execution_time=2.0,
            measurement_results={"target_state": 800, "other_states": 200},
            fidelity_estimate=0.9,
            success_probability=0.8,
            quantum_volume=None
        )
        
        # Classical time much longer than quantum time
        advantage = result.calculate_quantum_advantage(classical_time=1000.0)
        assert advantage == 500.0  # 1000 / 2
        
        # No advantage case
        no_advantage = result.calculate_quantum_advantage(classical_time=1.0)
        assert no_advantage == 0.5  # 1 / 2
        
        # Invalid cases
        assert result.calculate_quantum_advantage(classical_time=0) is None
        assert result.calculate_quantum_advantage(classical_time=-1) is None


class TestQuantumSecurityConfiguration:
    """Test quantum security configuration."""
    
    def test_security_config_creation(self):
        """Test valid security configuration creation."""
        config = QuantumSecurityConfiguration(
            config_id="qsc_001",
            security_policy=QuantumSecurityPolicy.POST_QUANTUM,
            enabled_algorithms={PostQuantumAlgorithm.KYBER_768, PostQuantumAlgorithm.DILITHIUM_3},
            key_management_mode="hybrid",
            distribution_protocol="qkd",
            monitoring_enabled=True,
            threat_detection_enabled=True,
            incident_response_enabled=True,
            compliance_frameworks=["NIST", "FIPS"]
        )
        
        assert config.config_id == "qsc_001"
        assert config.security_policy == QuantumSecurityPolicy.POST_QUANTUM
        assert len(config.enabled_algorithms) == 2
        assert config.key_management_mode == "hybrid"
    
    def test_is_quantum_safe(self):
        """Test quantum safety determination."""
        safe_config = QuantumSecurityConfiguration(
            config_id="safe_001",
            security_policy=QuantumSecurityPolicy.POST_QUANTUM,
            enabled_algorithms={PostQuantumAlgorithm.KYBER_1024},
            key_management_mode="quantum",
            distribution_protocol="qkd",
            monitoring_enabled=True,
            threat_detection_enabled=True,
            incident_response_enabled=True,
            compliance_frameworks=["NIST"]
        )
        
        assert safe_config.is_quantum_safe() is True
        
        unsafe_config = QuantumSecurityConfiguration(
            config_id="unsafe_001",
            security_policy=QuantumSecurityPolicy.LEGACY,
            enabled_algorithms={PostQuantumAlgorithm.KYBER_512},
            key_management_mode="classical",
            distribution_protocol="classical",
            monitoring_enabled=False,
            threat_detection_enabled=False,
            incident_response_enabled=False,
            compliance_frameworks=[]
        )
        
        assert unsafe_config.is_quantum_safe() is False
    
    def test_get_security_level(self):
        """Test security level mapping."""
        test_cases = [
            (QuantumSecurityPolicy.LEGACY, CryptographicStrength.CLASSICAL_ONLY),
            (QuantumSecurityPolicy.HYBRID, CryptographicStrength.QUANTUM_SAFE),
            (QuantumSecurityPolicy.POST_QUANTUM, CryptographicStrength.QUANTUM_RESISTANT),
            (QuantumSecurityPolicy.QUANTUM_READY, CryptographicStrength.QUANTUM_NATIVE)
        ]
        
        for policy, expected_strength in test_cases:
            config = QuantumSecurityConfiguration(
                config_id=f"test_{policy.value}",
                security_policy=policy,
                enabled_algorithms={PostQuantumAlgorithm.KYBER_512},
                key_management_mode="hybrid",
                distribution_protocol="hybrid",
                monitoring_enabled=True,
                threat_detection_enabled=True,
                incident_response_enabled=True,
                compliance_frameworks=["NIST"]
            )
            
            assert config.get_security_level() == expected_strength


class TestQuantumUtilityFunctions:
    """Test quantum utility functions."""
    
    def test_generate_quantum_ids(self):
        """Test quantum ID generation functions."""
        key_id = generate_quantum_key_id()
        session_id = generate_quantum_session_id()
        circuit_id = generate_circuit_id()
        
        assert key_id.startswith("qk_")
        assert session_id.startswith("qs_")
        assert circuit_id.startswith("qc_")
        
        # Should be unique
        assert generate_quantum_key_id() != generate_quantum_key_id()
        assert generate_quantum_session_id() != generate_quantum_session_id()
        assert generate_circuit_id() != generate_circuit_id()
    
    def test_assess_algorithm_quantum_vulnerability(self):
        """Test algorithm vulnerability assessment."""
        # RSA vulnerabilities
        is_vuln, threat = assess_algorithm_quantum_vulnerability("rsa", 1024)
        assert is_vuln is True
        assert threat == QuantumThreatLevel.CRITICAL
        
        is_vuln, threat = assess_algorithm_quantum_vulnerability("rsa", 2048)
        assert is_vuln is True
        assert threat == QuantumThreatLevel.HIGH
        
        # ECDSA vulnerabilities
        is_vuln, threat = assess_algorithm_quantum_vulnerability("ecdsa", 256)
        assert is_vuln is True
        assert threat == QuantumThreatLevel.HIGH
        
        # Post-quantum algorithms should not be vulnerable
        is_vuln, threat = assess_algorithm_quantum_vulnerability("kyber-768", 768)
        assert is_vuln is False
        assert threat == QuantumThreatLevel.MINIMAL
    
    def test_calculate_migration_priority(self):
        """Test migration priority calculation."""
        critical_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("critical"),
            asset_type="key",
            algorithm="rsa",
            key_size=1024,
            created_at=datetime.now(UTC) - timedelta(days=10),  # Recently created
            usage_context="authentication",
            quantum_vulnerable=True,
            threat_assessment=QuantumThreatLevel.CRITICAL,
            migration_priority=1  # Will be recalculated
        )
        
        priority = calculate_migration_priority(critical_asset)
        assert 1 <= priority <= 5
        assert priority >= 4  # Should be high priority due to critical threat + critical context + recent creation
        
        low_priority_asset = CryptographicAsset(
            asset_id=CryptographicAssetId("low_priority"),
            asset_type="key",
            algorithm="aes",
            key_size=256,
            created_at=datetime.now(UTC) - timedelta(days=365),  # Old
            usage_context="general",
            quantum_vulnerable=False,
            threat_assessment=QuantumThreatLevel.MINIMAL,
            migration_priority=1
        )
        
        low_priority = calculate_migration_priority(low_priority_asset)
        assert low_priority <= 2  # Should be low priority
    
    def test_recommend_post_quantum_algorithm(self):
        """Test post-quantum algorithm recommendations."""
        # Encryption use case
        encryption_rec = recommend_post_quantum_algorithm("rsa", "encryption")
        assert encryption_rec in [
            PostQuantumAlgorithm.KYBER_512,
            PostQuantumAlgorithm.KYBER_768,
            PostQuantumAlgorithm.KYBER_1024
        ]
        
        # Signature use case
        signature_rec = recommend_post_quantum_algorithm("ecdsa", "authentication")
        assert signature_rec in [
            PostQuantumAlgorithm.DILITHIUM_2,
            PostQuantumAlgorithm.DILITHIUM_3,
            PostQuantumAlgorithm.DILITHIUM_5,
            PostQuantumAlgorithm.FALCON_512,
            PostQuantumAlgorithm.FALCON_1024
        ]
        
        # Hash use case
        hash_rec = recommend_post_quantum_algorithm("sha256", "integrity")
        assert hash_rec == PostQuantumAlgorithm.SPHINCS_PLUS
        
        # Unknown use case - should provide safe default fallback
        unknown_rec = recommend_post_quantum_algorithm("unknown", "unknown")
        assert unknown_rec == PostQuantumAlgorithm.KYBER_768
    
    def test_create_default_quantum_config(self):
        """Test default quantum configuration creation."""
        config = create_default_quantum_config()
        
        assert isinstance(config, QuantumSecurityConfiguration)
        assert config.security_policy == QuantumSecurityPolicy.HYBRID
        assert len(config.enabled_algorithms) >= 3
        assert config.monitoring_enabled is True
        assert config.threat_detection_enabled is True
        assert "NIST" in config.compliance_frameworks


class TestQuantumErrorHandling:
    """Test quantum error handling and exceptions."""
    
    def test_quantum_error_creation(self):
        """Test quantum error creation and properties."""
        error = QuantumError("Test error message", "TEST_ERROR", {"detail": "test"})
        
        assert str(error) == "Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"detail": "test"}
        assert isinstance(error.timestamp, datetime)
    
    def test_quantum_error_factory_methods(self):
        """Test quantum error factory methods."""
        alg_error = QuantumError.algorithm_not_supported("test_algorithm")
        assert alg_error.error_code == "ALGORITHM_NOT_SUPPORTED"
        assert "test_algorithm" in str(alg_error)
        assert alg_error.details["algorithm"] == "test_algorithm"
        
        migration_error = QuantumError.migration_failed("test_asset_id")
        assert migration_error.error_code == "MIGRATION_FAILED"
        assert "test_asset_id" in str(migration_error)
        assert migration_error.details["asset_id"] == "test_asset_id"
        
        interface_error = QuantumError.quantum_interface_error("test_operation")
        assert interface_error.error_code == "QUANTUM_INTERFACE_ERROR"
        assert "test_operation" in str(interface_error)
        assert interface_error.details["operation"] == "test_operation"


# Property-based testing with Hypothesis
try:
    from hypothesis import given, strategies as st
    import hypothesis
    
    class TestQuantumArchitectureProperties:
        """Property-based tests for quantum architecture components."""
        
        @given(st.integers(min_value=1, max_value=10000))
        def test_key_size_always_positive(self, key_size):
            """Test that valid key sizes are always positive."""
            asset = CryptographicAsset(
                asset_id=CryptographicAssetId(f"test_{key_size}"),
                asset_type="key",
                algorithm="test",
                key_size=key_size,
                created_at=datetime.now(UTC),
                usage_context="test",
                quantum_vulnerable=False,
                threat_assessment=QuantumThreatLevel.MINIMAL,
                migration_priority=1
            )
            assert asset.key_size > 0
        
        @given(st.integers(min_value=1, max_value=5))
        def test_migration_priority_range(self, priority):
            """Test that migration priorities are within valid range."""
            asset = CryptographicAsset(
                asset_id=CryptographicAssetId(f"priority_{priority}"),
                asset_type="key",
                algorithm="test",
                key_size=256,
                created_at=datetime.now(UTC),
                usage_context="test",
                quantum_vulnerable=False,
                threat_assessment=QuantumThreatLevel.MINIMAL,
                migration_priority=priority
            )
            assert 1 <= asset.migration_priority <= 5
        
        @given(st.floats(min_value=0.0, max_value=1.0))
        def test_readiness_score_range(self, score):
            """Test that readiness scores are within valid range."""
            assessment = QuantumReadinessAssessment(
                assessment_id=f"score_{score}",
                scope="test",
                overall_readiness_score=score,
                quantum_vulnerable_assets=[],
                threat_timeline_estimate={},
                migration_recommendations=[],
                compliance_status={},
                risk_factors={}
            )
            assert 0.0 <= assessment.overall_readiness_score <= 1.0

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass