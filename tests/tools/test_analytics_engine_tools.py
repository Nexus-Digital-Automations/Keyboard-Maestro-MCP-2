"""
Comprehensive test suite for analytics engine tools.

Tests the complete analytics engine functionality including metrics collection,
ML insights, ROI analysis, dashboard generation, and enterprise reporting.

Security: Enterprise-grade test validation with comprehensive security coverage.
Performance: Test execution optimized for comprehensive coverage.
Type Safety: Complete integration with analytics architecture testing.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings

from src.server.tools.analytics_engine_tools import (
    km_analytics_engine, analytics_engine, AnalyticsEngine
)
from src.core.analytics_architecture import (
    MetricType, AnalyticsScope, AnalysisDepth, VisualizationFormat, 
    PrivacyMode, AnalyticsConfiguration
)
from src.core.either import Either
from src.core.errors import ValidationError, AnalyticsError
from fastmcp import Context


class TestAnalyticsEngine:
    """Test analytics engine core functionality."""
    
    @pytest.fixture
    def analytics_engine_instance(self):
        """Create analytics engine instance for testing."""
        return AnalyticsEngine()
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        return {
            'performance': {
                'km_clipboard_manager': {
                    'execution_time_ms': 45.2,
                    'memory_usage_mb': 12.5,
                    'cpu_utilization': 0.15,
                    'success_rate': 0.98,
                    'error_count': 1,
                    'throughput': 25.0
                },
                'km_app_control': {
                    'execution_time_ms': 78.6,
                    'memory_usage_mb': 18.3,
                    'cpu_utilization': 0.22,
                    'success_rate': 0.96,
                    'error_count': 2,
                    'throughput': 18.5
                }
            },
            'roi': {
                'km_clipboard_manager': {
                    'time_saved_hours': 2.5,
                    'cost_saved_dollars': 125.0,
                    'efficiency_gain_percent': 35.0,
                    'calculated_roi': 1.25
                },
                'km_app_control': {
                    'time_saved_hours': 3.2,
                    'cost_saved_dollars': 160.0,
                    'efficiency_gain_percent': 42.0,
                    'calculated_roi': 1.60
                }
            },
            'timestamp': datetime.now(UTC)
        }
    
    def test_analytics_engine_initialization(self, analytics_engine_instance):
        """Test analytics engine initialization."""
        assert analytics_engine_instance.config is not None
        assert isinstance(analytics_engine_instance.config, AnalyticsConfiguration)
        assert analytics_engine_instance.metrics_collector is not None
        assert analytics_engine_instance.ml_engine is not None
        assert len(analytics_engine_instance.ecosystem_tools) > 0
        assert 'km_clipboard_manager' in analytics_engine_instance.ecosystem_tools
    
    @pytest.mark.asyncio
    async def test_collect_ecosystem_metrics(self, analytics_engine_instance):
        """Test ecosystem metrics collection."""
        # Mock the metrics collector
        with patch.object(analytics_engine_instance.metrics_collector, 'collect_performance_metrics') as mock_perf, \
             patch.object(analytics_engine_instance.metrics_collector, 'collect_roi_metrics') as mock_roi:
            
            # Setup mocks
            from src.core.analytics_architecture import PerformanceMetrics, ROIMetrics
            
            mock_perf_data = PerformanceMetrics(
                tool_name="km_test_tool",
                operation="test_op",
                execution_time_ms=50.0,
                memory_usage_mb=15.0,
                cpu_utilization=0.20,
                success_rate=0.95,
                error_count=1,
                throughput=20.0
            )
            
            mock_roi_data = ROIMetrics(
                tool_name="km_test_tool",
                time_saved_hours=2.0,
                cost_saved_dollars=100.0,
                efficiency_gain_percent=30.0,
                automation_accuracy=0.95,
                user_satisfaction=4.2,
                implementation_cost=1000.0,
                maintenance_cost=100.0,
                calculated_roi=1.0
            )
            
            mock_perf.return_value = Either.right(mock_perf_data)
            mock_roi.return_value = Either.right(mock_roi_data)
            
            # Test with subset of tools
            result = await analytics_engine_instance.collect_ecosystem_metrics(['km_test_tool'])
            
            assert 'performance' in result
            assert 'roi' in result
            assert 'timestamp' in result
            assert 'km_test_tool' in result['performance']
            assert 'km_test_tool' in result['roi']
            assert result['performance']['km_test_tool']['execution_time_ms'] == 50.0
            assert result['roi']['km_test_tool']['calculated_roi'] == 1.0
    
    @pytest.mark.asyncio
    async def test_generate_ml_insights(self, analytics_engine_instance, sample_metrics_data):
        """Test ML insights generation."""
        with patch.object(analytics_engine_instance.ml_engine, 'generate_comprehensive_insights') as mock_insights:
            from src.core.analytics_architecture import MLInsight, MLModelType, create_insight_id
            
            mock_insight = MLInsight(
                insight_id=create_insight_id(MLModelType.PATTERN_RECOGNITION),
                model_type=MLModelType.PATTERN_RECOGNITION,
                confidence=0.85,
                description="High performance consistency detected",
                recommendation="Continue current optimization strategies",
                supporting_data={"pattern": "stable_performance"},
                impact_score=0.75
            )
            
            mock_insights.return_value = [mock_insight]
            
            insights = await analytics_engine_instance.generate_ml_insights(sample_metrics_data)
            
            assert len(insights) == 1
            assert insights[0]['model_type'] == 'pattern_recognition'
            assert insights[0]['confidence'] == 0.85
            assert 'description' in insights[0]
            assert 'recommendation' in insights[0]
    
    @pytest.mark.asyncio
    async def test_calculate_ecosystem_roi(self, analytics_engine_instance, sample_metrics_data):
        """Test ecosystem ROI calculation."""
        roi_analysis = await analytics_engine_instance.calculate_ecosystem_roi(sample_metrics_data)
        
        assert 'total_time_saved_hours' in roi_analysis
        assert 'total_cost_saved_dollars' in roi_analysis
        assert 'average_roi' in roi_analysis
        assert 'top_performing_tools' in roi_analysis
        assert 'improvement_opportunities' in roi_analysis
        
        # Check calculations
        assert roi_analysis['total_time_saved_hours'] == 5.7  # 2.5 + 3.2
        assert roi_analysis['total_cost_saved_dollars'] == 285.0  # 125.0 + 160.0
        assert roi_analysis['average_roi'] == 1.425  # (1.25 + 1.60) / 2
        assert len(roi_analysis['top_performing_tools']) <= 5
    
    @pytest.mark.asyncio
    async def test_generate_dashboard_data(self, analytics_engine_instance):
        """Test dashboard data generation."""
        with patch.object(analytics_engine_instance, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine_instance, 'generate_ml_insights') as mock_insights, \
             patch.object(analytics_engine_instance, 'calculate_ecosystem_roi') as mock_roi:
            
            mock_collect.return_value = {'performance': {}, 'roi': {}}
            mock_insights.return_value = []
            mock_roi.return_value = {'average_roi': 1.2}
            
            dashboard_data = await analytics_engine_instance.generate_dashboard_data(
                AnalyticsScope.ECOSYSTEM,
                "24h",
                VisualizationFormat.DASHBOARD
            )
            
            assert 'scope' in dashboard_data
            assert 'time_range' in dashboard_data
            assert 'format' in dashboard_data
            assert 'generated_at' in dashboard_data
            assert 'data' in dashboard_data
            assert dashboard_data['scope'] == 'ecosystem'
            assert dashboard_data['time_range'] == '24h'
            assert dashboard_data['format'] == 'dashboard'
    
    def test_calculate_average_metric(self, analytics_engine_instance, sample_metrics_data):
        """Test average metric calculation."""
        avg_response_time = analytics_engine_instance._calculate_average_metric(
            sample_metrics_data, 'execution_time_ms'
        )
        
        expected_avg = (45.2 + 78.6) / 2
        assert abs(avg_response_time - expected_avg) < 0.01
    
    def test_calculate_total_metric(self, analytics_engine_instance, sample_metrics_data):
        """Test total metric calculation."""
        total_memory = analytics_engine_instance._calculate_total_metric(
            sample_metrics_data, 'memory_usage_mb'
        )
        
        expected_total = 12.5 + 18.3
        assert abs(total_memory - expected_total) < 0.01
    
    @pytest.mark.asyncio
    async def test_get_system_health_indicators(self, analytics_engine_instance, sample_metrics_data):
        """Test system health indicators calculation."""
        health = await analytics_engine_instance._get_system_health_indicators(sample_metrics_data)
        
        assert 'status' in health
        assert 'health_score' in health
        assert 'indicators' in health
        assert health['status'] in ['excellent', 'good', 'fair', 'needs_attention']
        assert 0 <= health['health_score'] <= 100
        assert 'average_response_time_ms' in health['indicators']
        assert 'average_success_rate' in health['indicators']
    
    @pytest.mark.asyncio
    async def test_format_executive_summary(self, analytics_engine_instance):
        """Test executive summary formatting."""
        test_data = {
            'metrics_summary': {
                'total_tools_analyzed': 48,
                'average_response_time': 62.9,
                'ecosystem_success_rate': 0.97
            },
            'roi_analysis': {
                'total_cost_saved_dollars': 15000.0,
                'average_roi': 1.25,
                'total_time_saved_hours': 120.0,
                'top_performing_tools': [
                    {'tool': 'km_clipboard_manager', 'roi': 1.8},
                    {'tool': 'km_app_control', 'roi': 1.6}
                ]
            },
            'ml_insights': [
                {'insight_id': 'test1', 'title': 'Test Insight 1'},
                {'insight_id': 'test2', 'title': 'Test Insight 2'},
                {'insight_id': 'test3', 'title': 'Test Insight 3'}
            ],
            'system_health': {'status': 'excellent', 'health_score': 85}
        }
        
        summary = await analytics_engine_instance._format_executive_summary(test_data)
        
        assert 'key_metrics' in summary
        assert 'system_health' in summary
        assert 'top_insights' in summary
        assert 'roi_highlights' in summary
        assert summary['key_metrics']['tools_monitored'] == 48
        assert len(summary['top_insights']) == 3
        assert '$15,000.00' in summary['key_metrics']['total_cost_savings']


class TestAnalyticsEngineTool:
    """Test analytics engine MCP tool."""
    
    @pytest.mark.asyncio
    async def test_collect_operation(self):
        """Test collect operation."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine.metrics_collector, 'get_collection_statistics') as mock_stats:
            
            mock_collect.return_value = {'performance': {'test_tool': {}}, 'roi': {}}
            mock_stats.return_value = {'total_metrics_collected': 100, 'success_rate': 0.95}
            
            result = await km_analytics_engine(
                operation="collect",
                analytics_scope="ecosystem"
            )
            
            assert result['operation'] == 'collect'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'metrics' in result['data']
            assert 'collection_statistics' in result['data']
            assert 'metadata' in result
    
    @pytest.mark.asyncio
    async def test_analyze_operation(self):
        """Test analyze operation."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine, 'calculate_ecosystem_roi') as mock_roi, \
             patch.object(analytics_engine, 'generate_ml_insights') as mock_insights:
            
            mock_collect.return_value = {'performance': {'test_tool': {}}, 'roi': {}}
            mock_roi.return_value = {'average_roi': 1.2}
            mock_insights.return_value = [{'insight_id': 'test', 'model_type': 'pattern_recognition'}]
            
            result = await km_analytics_engine(
                operation="analyze",
                analytics_scope="ecosystem",
                ml_insights=True,
                anomaly_detection=True
            )
            
            assert result['operation'] == 'analyze'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'performance_analysis' in result['data']
            assert 'roi_analysis' in result['data']
            assert 'ml_insights' in result['data']
            assert 'anomaly_detection' in result['data']
    
    @pytest.mark.asyncio
    async def test_dashboard_operation(self):
        """Test dashboard operation."""
        with patch.object(analytics_engine, 'generate_dashboard_data') as mock_dashboard:
            
            mock_dashboard.return_value = {
                'scope': 'ecosystem',
                'data': {'metrics_summary': {}, 'performance_overview': {}}
            }
            
            result = await km_analytics_engine(
                operation="dashboard",
                analytics_scope="ecosystem",
                visualization_format="dashboard"
            )
            
            assert result['operation'] == 'dashboard'
            assert result['status'] == 'success'
            assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_report_operation(self):
        """Test report operation."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine, 'generate_ml_insights') as mock_insights, \
             patch.object(analytics_engine, 'calculate_ecosystem_roi') as mock_roi, \
             patch.object(analytics_engine, '_format_executive_summary') as mock_format, \
             patch.object(analytics_engine, '_get_system_health_indicators') as mock_health:
            
            mock_collect.return_value = {'performance': {'test_tool': {}}, 'roi': {}}
            mock_insights.return_value = []
            mock_roi.return_value = {'average_roi': 1.2}
            mock_format.return_value = {'key_metrics': {}}
            mock_health.return_value = {'status': 'excellent'}
            
            result = await km_analytics_engine(
                operation="report",
                analytics_scope="ecosystem"
            )
            
            assert result['operation'] == 'report'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'executive_summary' in result['data']
            assert 'detailed_metrics' in result['data']
            assert 'recommendations' in result['data']
    
    @pytest.mark.asyncio
    async def test_predict_operation(self):
        """Test predict operation."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect:
            
            mock_collect.return_value = {'performance': {'test_tool': {'execution_time_ms': 50}}, 'roi': {}}
            
            result = await km_analytics_engine(
                operation="predict",
                analytics_scope="ecosystem"
            )
            
            assert result['operation'] == 'predict'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'predictions' in result['data']
            assert 'forecast_horizon' in result['data']
            assert 'confidence_level' in result['data']
    
    @pytest.mark.asyncio
    async def test_optimize_operation(self):
        """Test optimize operation."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine, 'generate_ml_insights') as mock_insights, \
             patch.object(analytics_engine, 'calculate_ecosystem_roi') as mock_roi, \
             patch.object(analytics_engine, '_get_system_health_indicators') as mock_health:
            
            mock_collect.return_value = {
                'performance': {
                    'slow_tool': {'execution_time_ms': 500},
                    'fast_tool': {'execution_time_ms': 50}
                },
                'roi': {}
            }
            mock_insights.return_value = [
                {'impact_score': 0.9, 'recommendation': 'Optimize database queries'}
            ]
            mock_roi.return_value = {
                'improvement_opportunities': [{'tool': 'slow_tool', 'roi': 0.3}]
            }
            mock_health.return_value = {'health_score': 75}
            
            result = await km_analytics_engine(
                operation="optimize",
                analytics_scope="ecosystem"
            )
            
            assert result['operation'] == 'optimize'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'optimization_recommendations' in result['data']
            assert 'current_performance_baseline' in result['data']
            assert 'potential_improvements' in result['data']
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation handling."""
        with pytest.raises(Exception):  # Should raise ToolError
            await km_analytics_engine(operation="invalid_operation")
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid analytics scope
        with pytest.raises(Exception):
            await km_analytics_engine(
                operation="collect",
                analytics_scope="invalid_scope"
            )
        
        # Test invalid visualization format
        with pytest.raises(Exception):
            await km_analytics_engine(
                operation="dashboard",
                visualization_format="invalid_format"
            )
        
        # Test invalid privacy mode
        with pytest.raises(Exception):
            await km_analytics_engine(
                operation="collect",
                privacy_mode="invalid_mode"
            )
    
    @pytest.mark.asyncio
    async def test_metrics_types_validation(self):
        """Test metrics types validation."""
        # Valid metrics types
        result = await km_analytics_engine(
            operation="collect",
            metrics_types=["performance", "roi", "usage"]
        )
        assert result['status'] == 'success'
        
        # Invalid metrics type
        with pytest.raises(Exception):
            await km_analytics_engine(
                operation="collect",
                metrics_types=["invalid_metric_type"]
            )
    
    @pytest.mark.asyncio
    async def test_execution_metadata(self):
        """Test execution metadata is included."""
        result = await km_analytics_engine(
            operation="collect",
            analytics_scope="ecosystem"
        )
        
        assert 'metadata' in result
        assert 'execution_time_seconds' in result['metadata']
        assert 'timestamp' in result['metadata']
        assert 'analytics_scope' in result['metadata']
        assert 'analysis_depth' in result['metadata']
        assert 'privacy_mode' in result['metadata']
        assert isinstance(result['metadata']['execution_time_seconds'], float)


class TestAnalyticsEngineProperties:
    """Property-based tests for analytics engine."""
    
    @given(st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=1, max_size=20))
    def test_average_calculation_properties(self, values):
        """Test average calculation properties."""
        # Create mock metrics data
        metrics_data = {
            'performance': {
                f'tool_{i}': {'execution_time_ms': val} 
                for i, val in enumerate(values)
            }
        }
        
        engine = AnalyticsEngine()
        avg = engine._calculate_average_metric(metrics_data, 'execution_time_ms')
        
        # Properties that should always hold
        assert avg >= min(values)
        assert avg <= max(values)
        assert abs(avg - sum(values) / len(values)) < 0.001
    
    @given(st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=20))
    def test_total_calculation_properties(self, values):
        """Test total calculation properties."""
        metrics_data = {
            'performance': {
                f'tool_{i}': {'memory_usage_mb': val} 
                for i, val in enumerate(values)
            }
        }
        
        engine = AnalyticsEngine()
        total = engine._calculate_total_metric(metrics_data, 'memory_usage_mb')
        
        # Properties that should always hold
        assert total >= 0
        assert abs(total - sum(values)) < 0.001
        if values:
            assert total >= max(values)
    
    @given(
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=10.0, max_value=1000.0)
    )
    @pytest.mark.asyncio
    async def test_roi_calculation_properties(self, time_saved, cost_saved):
        """Test ROI calculation properties."""
        roi_data = {
            'tool_1': {
                'time_saved_hours': time_saved,
                'cost_saved_dollars': cost_saved,
                'calculated_roi': cost_saved / max(cost_saved * 0.1, 1.0)  # Simplified ROI
            }
        }
        
        metrics_data = {'roi': roi_data}
        
        engine = AnalyticsEngine()
        roi_analysis = await engine.calculate_ecosystem_roi(metrics_data)
        
        # Properties that should always hold
        assert roi_analysis['total_time_saved_hours'] >= 0
        assert roi_analysis['total_cost_saved_dollars'] >= 0
        assert roi_analysis['total_time_saved_hours'] == time_saved
        assert roi_analysis['total_cost_saved_dollars'] == cost_saved


class TestAnalyticsEngineIntegration:
    """Integration tests for analytics engine."""
    
    @pytest.mark.asyncio
    async def test_full_analytics_workflow(self):
        """Test complete analytics workflow from collection to optimization."""
        # Mock all external dependencies
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect, \
             patch.object(analytics_engine, 'generate_ml_insights') as mock_insights, \
             patch.object(analytics_engine, 'calculate_ecosystem_roi') as mock_roi, \
             patch.object(analytics_engine, '_get_system_health_indicators') as mock_health:
            
            # Setup comprehensive mock data
            mock_metrics = {
                'performance': {
                    'km_clipboard_manager': {
                        'execution_time_ms': 45.0,
                        'success_rate': 0.98,
                        'memory_usage_mb': 12.0,
                        'cpu_utilization': 0.15
                    },
                    'km_app_control': {
                        'execution_time_ms': 85.0,
                        'success_rate': 0.95,
                        'memory_usage_mb': 18.0,
                        'cpu_utilization': 0.25
                    }
                },
                'roi': {}
            }
            
            mock_roi_analysis = {
                'average_roi': 1.35,
                'total_cost_saved_dollars': 5000.0,
                'improvement_opportunities': []
            }
            
            mock_insights_data = [
                {
                    'insight_id': 'insight_1',
                    'model_type': 'pattern_recognition',
                    'confidence': 0.85,
                    'description': 'Consistent performance pattern detected',
                    'recommendation': 'Maintain current optimization strategies',
                    'impact_score': 0.75
                }
            ]
            
            mock_health_data = {
                'status': 'excellent',
                'health_score': 88,
                'indicators': {
                    'average_response_time_ms': 65.0,
                    'average_success_rate': 0.965
                }
            }
            
            mock_collect.return_value = mock_metrics
            mock_insights.return_value = mock_insights_data
            mock_roi.return_value = mock_roi_analysis
            mock_health.return_value = mock_health_data
            
            # Test each operation in sequence
            operations = ["collect", "analyze", "dashboard", "report", "optimize"]
            
            for operation in operations:
                result = await km_analytics_engine(
                    operation=operation,
                    analytics_scope="ecosystem",
                    ml_insights=True,
                    roi_calculation=True
                )
                
                assert result['status'] == 'success'
                assert result['operation'] == operation
                assert 'metadata' in result
                assert 'data' in result
                
                # Verify execution time is reasonable (<5 seconds)
                assert result['metadata']['execution_time_seconds'] < 5.0
    
    @pytest.mark.asyncio
    async def test_analytics_consistency_across_operations(self):
        """Test that analytics results are consistent across different operations."""
        with patch.object(analytics_engine, 'collect_ecosystem_metrics') as mock_collect:
            
            # Use same mock data for all operations
            consistent_metrics = {
                'performance': {
                    'test_tool': {
                        'execution_time_ms': 100.0,
                        'success_rate': 0.95,
                        'memory_usage_mb': 20.0
                    }
                },
                'roi': {}
            }
            
            mock_collect.return_value = consistent_metrics
            
            # Run collect and analyze operations
            collect_result = await km_analytics_engine(operation="collect")
            analyze_result = await km_analytics_engine(operation="analyze")
            
            # Verify both operations see the same underlying data
            assert collect_result['status'] == 'success'
            assert analyze_result['status'] == 'success'
            
            # Both should have processed the same tool
            collected_tools = collect_result['data']['metrics']['performance'].keys()
            analyzed_tools = analyze_result['data']['performance_analysis'].keys()
            assert 'test_tool' in collected_tools
            assert 'test_tool' in analyzed_tools