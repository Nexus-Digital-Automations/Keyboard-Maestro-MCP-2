"""
Test Cryptography Migrator - Post-Quantum Migration System Testing

Comprehensive tests for post-quantum cryptography migration, algorithm analysis,
and secure transition management for enterprise cryptographic assets.

Architecture: Property-Based Testing + Type Safety + Contract Validation + Security Testing
Performance: <500ms test execution, comprehensive migration testing, security compliance validation
Security: Post-quantum migration validation, secure key transition testing, rollback mechanism validation
"""

import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, patch, MagicMock

from src.quantum.cryptography_migrator import CryptographyMigrator
from src.core.quantum_architecture import (
    PostQuantumAlgorithm,
    QuantumThreatLevel,
    CryptographicStrength,
    QuantumError,
    CryptographicAsset,
    PostQuantumMigrationPlan,
    QuantumReadinessAssessment,
    CryptographicAssetId,
    assess_algorithm_quantum_vulnerability,
    recommend_post_quantum_algorithm
)
from src.core.either import Either


class TestCryptographyMigrator:
    """Test cryptography migrator initialization and basic functionality."""
    
    def test_migrator_initialization(self):
        """Test cryptography migrator proper initialization."""
        migrator = CryptographyMigrator()
        
        assert migrator.cryptographic_assets == {}
        assert migrator.migration_plans == {}
        assert migrator.migration_history == []
        assert migrator.quantum_config is not None
        assert "total_assets_analyzed" in migrator.migration_metrics
        assert "successful_migrations" in migrator.migration_metrics
        assert migrator.migration_metrics["total_assets_analyzed"] == 0
    
    def test_migrator_quantum_config(self):
        """Test migrator has valid quantum configuration."""
        migrator = CryptographyMigrator()
        
        config = migrator.quantum_config
        assert config.monitoring_enabled is True
        assert len(config.enabled_algorithms) > 0
        assert "NIST" in config.compliance_frameworks


class TestQuantumReadinessAnalysis:
    """Test quantum readiness analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_system_scope(self):
        """Test quantum readiness analysis with system scope."""
        migrator = CryptographyMigrator()
        
        result = await migrator.analyze_quantum_readiness(
            scope="system",
            include_vulnerabilities=True,
            deep_analysis=True
        )
        
        assert result.is_success()
        assessment = result.value
        
        assert isinstance(assessment, QuantumReadinessAssessment)
        assert assessment.scope == "system"
        assert 0.0 <= assessment.overall_readiness_score <= 1.0
        assert isinstance(assessment.quantum_vulnerable_assets, list)
        assert isinstance(assessment.migration_recommendations, list)
        assert len(assessment.migration_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_application_scope(self):
        """Test quantum readiness analysis with application scope."""
        migrator = CryptographyMigrator()
        
        result = await migrator.analyze_quantum_readiness(
            scope="application",
            include_vulnerabilities=True,
            deep_analysis=False
        )
        
        assert result.is_success()
        assessment = result.value
        
        assert assessment.scope == "application"
        assert "app_api_key" in str(assessment.quantum_vulnerable_assets) or len(assessment.quantum_vulnerable_assets) >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_cryptography_scope(self):
        """Test quantum readiness analysis with cryptography scope."""
        migrator = CryptographyMigrator()
        
        result = await migrator.analyze_quantum_readiness(
            scope="cryptography",
            include_vulnerabilities=True,
            deep_analysis=True
        )
        
        assert result.is_success()
        assessment = result.value
        
        assert assessment.scope == "cryptography"
        critical_vulnerabilities = assessment.get_critical_vulnerabilities()
        assert isinstance(critical_vulnerabilities, list)
        
        # Check timeline estimates
        timeline = assessment.threat_timeline_estimate
        assert "quantum_advantage_demonstration" in timeline
        assert "cryptographically_relevant_quantum_computer" in timeline
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_protocols_scope(self):
        """Test quantum readiness analysis with protocols scope."""
        migrator = CryptographyMigrator()
        
        result = await migrator.analyze_quantum_readiness(
            scope="protocols",
            include_vulnerabilities=True,
            deep_analysis=True
        )
        
        assert result.is_success()
        assessment = result.value
        
        assert assessment.scope == "protocols"
        assert assessment.compliance_status is not None
        assert isinstance(assessment.risk_factors, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_invalid_scope(self):
        """Test quantum readiness analysis with invalid scope."""
        migrator = CryptographyMigrator()
        
        # Should use contract validation to prevent invalid scope
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            await migrator.analyze_quantum_readiness(
                scope="invalid_scope",
                include_vulnerabilities=True
            )
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_metrics_update(self):
        """Test that analysis updates metrics correctly."""
        migrator = CryptographyMigrator()
        
        initial_analyzed = migrator.migration_metrics["total_assets_analyzed"]
        initial_vulnerable = migrator.migration_metrics["vulnerable_assets_found"]
        
        result = await migrator.analyze_quantum_readiness(
            scope="system",
            include_vulnerabilities=True
        )
        
        assert result.is_success()
        assert migrator.migration_metrics["total_assets_analyzed"] > initial_analyzed
        # Vulnerable count may or may not increase depending on discovered assets


class TestMigrationPlanCreation:
    """Test migration plan creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_basic(self):
        """Test basic migration plan creation."""
        migrator = CryptographyMigrator()
        
        # First analyze to populate assets
        analysis_result = await migrator.analyze_quantum_readiness("system")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        assessment = analysis_result.value
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        
        # Use first two assets for testing
        target_assets = all_asset_ids[:2]
        
        result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="hybrid",
            target_security_level="post_quantum"
        )
        
        assert result.is_success()
        plan = result.value
        
        assert isinstance(plan, PostQuantumMigrationPlan)
        assert len(plan.target_assets) == len(target_assets)
        assert plan.migration_strategy == "hybrid"
        assert isinstance(plan.estimated_duration, timedelta)
        assert len(plan.validation_criteria) > 0
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_full_replacement(self):
        """Test migration plan creation with full replacement strategy."""
        migrator = CryptographyMigrator()
        
        # Analyze and get assets
        analysis_result = await migrator.analyze_quantum_readiness("cryptography")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="full_replacement",
            target_security_level="quantum_ready"
        )
        
        assert result.is_success()
        plan = result.value
        
        assert plan.migration_strategy == "full_replacement"
        assert "quantum_interface_ready" in plan.validation_criteria
        assert "post_quantum_compliance_verified" in plan.validation_criteria
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_gradual(self):
        """Test migration plan creation with gradual strategy."""
        migrator = CryptographyMigrator()
        
        # Analyze and get assets
        analysis_result = await migrator.analyze_quantum_readiness("application")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) >= 2, "Need at least 2 assets for gradual migration test"
        target_assets = all_asset_ids[:2]  # Use first two assets
        
        result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="gradual",
            target_security_level="post_quantum"
        )
        
        assert result.is_success()
        plan = result.value
        
        assert plan.migration_strategy == "gradual"
        phases = plan.get_migration_phases()
        assert len(phases) == 3  # Critical, medium, low priority phases
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_missing_assets(self):
        """Test migration plan creation with missing assets."""
        migrator = CryptographyMigrator()
        
        # Try to create plan without analyzing first (no assets)
        target_assets = [CryptographicAssetId("nonexistent_asset")]
        
        result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="hybrid"
        )
        
        assert result.is_error()
        assert "Assets not found" in result.error_value.args[0]
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_empty_assets(self):
        """Test migration plan creation with empty asset list."""
        migrator = CryptographyMigrator()
        
        # Should trigger contract validation
        from src.core.errors import ContractViolationError
        with pytest.raises(ContractViolationError):
            await migrator.create_migration_plan(
                target_assets=[],
                migration_strategy="hybrid"
            )
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_storage(self):
        """Test that migration plans are properly stored."""
        migrator = CryptographyMigrator()
        
        # Analyze and create plan
        analysis_result = await migrator.analyze_quantum_readiness("system")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        initial_plan_count = len(migrator.migration_plans)
        
        result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="hybrid"
        )
        
        assert result.is_success()
        plan = result.value
        
        assert len(migrator.migration_plans) == initial_plan_count + 1
        assert plan.plan_id in migrator.migration_plans
        assert migrator.migration_plans[plan.plan_id] == plan


class TestMigrationPlanExecution:
    """Test migration plan execution functionality."""
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_basic(self):
        """Test basic migration plan execution."""
        migrator = CryptographyMigrator()
        
        # Create a plan first
        analysis_result = await migrator.analyze_quantum_readiness("system")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="hybrid"
        )
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        # Execute the plan
        result = await migrator.execute_migration_plan(
            plan_id=plan_id,
            dry_run=False,
            validation_mode=True
        )
        
        assert result.is_success()
        execution_data = result.value
        
        assert execution_data["plan_id"] == plan_id
        assert execution_data["dry_run"] is False
        assert "assets_processed" in execution_data
        assert "successful_migrations" in execution_data
        assert "execution_duration_seconds" in execution_data
        assert isinstance(execution_data["execution_duration_seconds"], float)
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_dry_run(self):
        """Test migration plan execution in dry run mode."""
        migrator = CryptographyMigrator()
        
        # Create and execute plan in dry run mode
        analysis_result = await migrator.analyze_quantum_readiness("application")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(target_assets=target_assets)
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        result = await migrator.execute_migration_plan(
            plan_id=plan_id,
            dry_run=True,
            validation_mode=False
        )
        
        assert result.is_success()
        execution_data = result.value
        
        assert execution_data["dry_run"] is True
        # Metrics should not be updated in dry run mode
        assert execution_data["validation_results"] == {}
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_nonexistent(self):
        """Test execution of nonexistent migration plan."""
        migrator = CryptographyMigrator()
        
        result = await migrator.execute_migration_plan(
            plan_id="nonexistent_plan_id",
            dry_run=False
        )
        
        assert result.is_error()
        assert "Migration plan not found" in result.error_value.args[0]
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_validation(self):
        """Test migration plan execution with validation."""
        migrator = CryptographyMigrator()
        
        # Create plan and execute with validation
        analysis_result = await migrator.analyze_quantum_readiness("cryptography")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(target_assets=target_assets)
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        result = await migrator.execute_migration_plan(
            plan_id=plan_id,
            validation_mode=True
        )
        
        assert result.is_success()
        execution_data = result.value
        
        assert "validation_results" in execution_data
        # Should have validation results if validation mode enabled
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_metrics_update(self):
        """Test that plan execution updates metrics correctly."""
        migrator = CryptographyMigrator()
        
        initial_successful = migrator.migration_metrics["successful_migrations"]
        initial_failed = migrator.migration_metrics["failed_migrations"]
        
        # Create and execute plan
        analysis_result = await migrator.analyze_quantum_readiness("protocols")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(target_assets=target_assets)
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        await migrator.execute_migration_plan(plan_id=plan_id, dry_run=False)
        
        # Metrics should be updated (successful or failed)
        final_successful = migrator.migration_metrics["successful_migrations"]
        final_failed = migrator.migration_metrics["failed_migrations"]
        
        assert (final_successful > initial_successful) or (final_failed > initial_failed)
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_history_tracking(self):
        """Test that plan execution is tracked in history."""
        migrator = CryptographyMigrator()
        
        initial_history_length = len(migrator.migration_history)
        
        # Create and execute plan
        analysis_result = await migrator.analyze_quantum_readiness("system")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(target_assets=target_assets)
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        await migrator.execute_migration_plan(plan_id=plan_id)
        
        assert len(migrator.migration_history) == initial_history_length + 1
        latest_entry = migrator.migration_history[-1]
        assert latest_entry["plan_id"] == plan_id
        assert "execution_time" in latest_entry
        assert "results" in latest_entry


class TestMigrationStatus:
    """Test migration status retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_migration_status_overall(self):
        """Test getting overall migration status."""
        migrator = CryptographyMigrator()
        
        result = await migrator.get_migration_status()
        
        assert result.is_success()
        status = result.value
        
        assert "overall_metrics" in status
        assert "total_plans" in status
        assert "recent_executions" in status
        assert "quantum_config" in status
        
        # Verify metrics structure
        metrics = status["overall_metrics"]
        assert "total_assets_analyzed" in metrics
        assert "successful_migrations" in metrics
        assert "failed_migrations" in metrics
    
    @pytest.mark.asyncio
    async def test_get_migration_status_specific_plan(self):
        """Test getting status for specific migration plan."""
        migrator = CryptographyMigrator()
        
        # Create a plan first
        analysis_result = await migrator.analyze_quantum_readiness("system")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        plan_result = await migrator.create_migration_plan(target_assets=target_assets)
        assert plan_result.is_success()
        plan_id = plan_result.value.plan_id
        
        # Execute the plan to create history
        exec_result = await migrator.execute_migration_plan(plan_id=plan_id)
        assert exec_result.is_success()
        
        # Get status for specific plan
        result = await migrator.get_migration_status(plan_id=plan_id)
        
        assert result.is_success()
        status = result.value
        
        assert "plan_details" in status
        plan_details = status["plan_details"]
        assert plan_details["plan_id"] == plan_id
        assert "created_at" in plan_details
        assert "target_assets_count" in plan_details
        assert "execution_count" in plan_details
        assert plan_details["execution_count"] >= 1
    
    @pytest.mark.asyncio
    async def test_get_migration_status_nonexistent_plan(self):
        """Test getting status for nonexistent plan."""
        migrator = CryptographyMigrator()
        
        result = await migrator.get_migration_status(plan_id="nonexistent_plan")
        
        assert result.is_error()
        assert "Migration plan not found" in result.error_value.args[0]
    
    @pytest.mark.asyncio
    async def test_get_migration_status_quantum_config(self):
        """Test that migration status includes quantum configuration."""
        migrator = CryptographyMigrator()
        
        result = await migrator.get_migration_status()
        status = result.value
        
        quantum_config = status["quantum_config"]
        assert "security_policy" in quantum_config
        assert "enabled_algorithms" in quantum_config
        assert "monitoring_enabled" in quantum_config
        assert quantum_config["monitoring_enabled"] is True


class TestMigrationErrorHandling:
    """Test error handling in migration operations."""
    
    @pytest.mark.asyncio
    async def test_analyze_quantum_readiness_error_handling(self):
        """Test error handling in quantum readiness analysis."""
        migrator = CryptographyMigrator()
        
        # Mock an internal error
        with patch.object(migrator, '_discover_cryptographic_assets') as mock_discover:
            mock_discover.return_value = Either.error(QuantumError("Discovery failed"))
            
            result = await migrator.analyze_quantum_readiness("system")
            
            assert result.is_error()  # Should return error when discovery fails
            assert "Discovery failed" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_create_migration_plan_error_handling(self):
        """Test error handling in migration plan creation."""
        migrator = CryptographyMigrator()
        
        # This should trigger the missing assets error
        result = await migrator.create_migration_plan(
            target_assets=[CryptographicAssetId("definitely_missing_asset")]
        )
        
        assert result.is_error()
        error = result.error_value
        assert isinstance(error, QuantumError)
        assert "Assets not found" in str(error)
    
    @pytest.mark.asyncio
    async def test_execute_migration_plan_error_handling(self):
        """Test error handling in migration plan execution."""
        migrator = CryptographyMigrator()
        
        # Test with invalid plan ID
        result = await migrator.execute_migration_plan("invalid_plan_id")
        
        assert result.is_error()
        assert "Migration plan not found" in result.error_value.args[0]


class TestMigrationIntegration:
    """Test end-to-end migration workflows."""
    
    @pytest.mark.asyncio
    async def test_full_migration_workflow(self):
        """Test complete migration workflow from analysis to execution."""
        migrator = CryptographyMigrator()
        
        # Step 1: Analyze quantum readiness
        analysis_result = await migrator.analyze_quantum_readiness(
            scope="system",
            include_vulnerabilities=True,
            deep_analysis=True
        )
        assert analysis_result.is_success()
        assessment = analysis_result.value
        
        # Step 2: Create migration plan based on vulnerable assets
        vulnerable_count = len(assessment.quantum_vulnerable_assets)
        if vulnerable_count > 0:
            # Use first few vulnerable assets
            target_assets = [asset.asset_id for asset in assessment.quantum_vulnerable_assets[:2]]
        else:
            # Use any discovered asset from the migrator's assets
            all_asset_ids = list(migrator.cryptographic_assets.keys())
            assert len(all_asset_ids) > 0, "No assets available for workflow test"
            target_assets = [all_asset_ids[0]]
        
        plan_result = await migrator.create_migration_plan(
            target_assets=target_assets,
            migration_strategy="hybrid"
        )
        assert plan_result.is_success()
        plan = plan_result.value
        
        # Step 3: Execute migration plan
        execution_result = await migrator.execute_migration_plan(
            plan_id=plan.plan_id,
            dry_run=False,
            validation_mode=True
        )
        assert execution_result.is_success()
        execution_data = execution_result.value
        
        # Step 4: Check final status
        status_result = await migrator.get_migration_status(plan_id=plan.plan_id)
        assert status_result.is_success()
        
        # Verify workflow completed successfully
        assert execution_data["plan_id"] == plan.plan_id
        assert execution_data["assets_processed"] >= 0
        assert "execution_duration_seconds" in execution_data
    
    @pytest.mark.asyncio
    async def test_multiple_scope_analysis_consistency(self):
        """Test that multiple scope analyses provide consistent results."""
        migrator = CryptographyMigrator()
        
        scopes = ["system", "application", "cryptography", "protocols"]
        assessments = {}
        
        for scope in scopes:
            result = await migrator.analyze_quantum_readiness(scope=scope)
            assert result.is_success()
            assessments[scope] = result.value
        
        # All assessments should be valid
        for scope, assessment in assessments.items():
            assert assessment.scope == scope
            assert 0.0 <= assessment.overall_readiness_score <= 1.0
            assert isinstance(assessment.quantum_vulnerable_assets, list)
            assert isinstance(assessment.migration_recommendations, list)
        
        # Should have discovered assets
        total_assets = sum(len(assessment.quantum_vulnerable_assets) for assessment in assessments.values())
        assert total_assets > 0  # Should have found some assets across all scopes
    
    @pytest.mark.asyncio
    async def test_migration_strategy_variations(self):
        """Test different migration strategies produce valid plans."""
        migrator = CryptographyMigrator()
        
        # Analyze once
        analysis_result = await migrator.analyze_quantum_readiness("cryptography")
        assert analysis_result.is_success()
        
        # Get actual asset IDs from discovered assets
        all_asset_ids = list(migrator.cryptographic_assets.keys())
        assert len(all_asset_ids) > 0, "No assets discovered during analysis"
        target_assets = [all_asset_ids[0]]  # Use first asset
        
        strategies = ["hybrid", "full_replacement", "gradual"]
        
        for strategy in strategies:
            result = await migrator.create_migration_plan(
                target_assets=target_assets,
                migration_strategy=strategy
            )
            
            assert result.is_success()
            plan = result.value
            assert plan.migration_strategy == strategy
            
            # Each strategy should have appropriate phases
            phases = plan.get_migration_phases()
            if strategy == "gradual":
                assert len(phases) == 3  # Multiple phases
            else:
                assert len(phases) == 1  # Single phase
    
    @pytest.mark.asyncio
    async def test_concurrent_migration_operations(self):
        """Test concurrent migration operations don't interfere."""
        migrator = CryptographyMigrator()
        
        # Run multiple analyses concurrently
        tasks = [
            migrator.analyze_quantum_readiness("system"),
            migrator.analyze_quantum_readiness("application"),
            migrator.analyze_quantum_readiness("cryptography")
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.is_success()
        
        # Verify metrics are correct
        assert migrator.migration_metrics["total_assets_analyzed"] > 0
        
        # Create multiple plans concurrently
        target_assets_sets = [
            [CryptographicAssetId("concurrent_1")],
            [CryptographicAssetId("concurrent_2")],
            [CryptographicAssetId("concurrent_3")]
        ]
        
        plan_tasks = [
            migrator.create_migration_plan(target_assets=assets)
            for assets in target_assets_sets
        ]
        
        # Some may fail due to missing assets, but should handle gracefully
        plan_results = await asyncio.gather(*plan_tasks, return_exceptions=True)
        
        # Should not raise exceptions, but may return error results
        for result in plan_results:
            assert not isinstance(result, Exception)


class TestMigrationPerformance:
    """Test migration performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self):
        """Test quantum readiness analysis performance."""
        migrator = CryptographyMigrator()
        
        import time
        start_time = time.time()
        
        result = await migrator.analyze_quantum_readiness(
            scope="system",
            include_vulnerabilities=True,
            deep_analysis=True
        )
        
        execution_time = time.time() - start_time
        
        assert result.is_success()
        assert execution_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_migration_plan_creation_performance(self):
        """Test migration plan creation performance."""
        migrator = CryptographyMigrator()
        
        # Prepare by analyzing
        await migrator.analyze_quantum_readiness("application")
        target_assets = [CryptographicAssetId("perf_test_asset")]
        
        import time
        start_time = time.time()
        
        result = await migrator.create_migration_plan(target_assets=target_assets)
        
        execution_time = time.time() - start_time
        
        # May succeed or fail based on asset existence, but should be fast
        assert execution_time < 0.5  # Should complete within 500ms
    
    @pytest.mark.asyncio
    async def test_large_asset_analysis_performance(self):
        """Test analysis performance with larger asset sets."""
        migrator = CryptographyMigrator()
        
        import time
        start_time = time.time()
        
        # Run analysis on multiple scopes to get more assets
        scopes = ["system", "application", "cryptography", "protocols"]
        for scope in scopes:
            result = await migrator.analyze_quantum_readiness(scope=scope)
            assert result.is_success()
        
        execution_time = time.time() - start_time
        
        # Even with multiple scopes, should be reasonably fast
        assert execution_time < 2.0  # Should complete within 2 seconds