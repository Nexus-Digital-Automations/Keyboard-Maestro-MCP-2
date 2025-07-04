"""
Automated compliance report generation with comprehensive analytics.

This module provides sophisticated report generation capabilities for compliance
reporting, risk analysis, trend analytics, and regulatory documentation with
support for multiple output formats and automated insights.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, UTC
import uuid
import json
import logging
from pathlib import Path

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.audit_framework import (
    AuditEvent, ComplianceReport, ComplianceStandard, RiskLevel, 
    AuditError, AuditEventType
)
from .event_logger import EventLogger
from .compliance_monitor import ComplianceMonitor


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Automated compliance report generation with analytics and insights."""
    
    def __init__(self, event_logger: EventLogger, compliance_monitor: ComplianceMonitor):
        self.event_logger = event_logger
        self.compliance_monitor = compliance_monitor
        self.report_cache: Dict[str, ComplianceReport] = {}
        self.cache_expiry = timedelta(hours=1)
    
    @require(lambda self, standard: isinstance(standard, ComplianceStandard))
    @require(lambda self, period_start: isinstance(period_start, datetime))
    @require(lambda self, period_end: isinstance(period_end, datetime))
    async def generate_compliance_report(self, 
                                       standard: ComplianceStandard,
                                       period_start: datetime, 
                                       period_end: datetime,
                                       include_recommendations: bool = True) -> Either[AuditError, ComplianceReport]:
        """Generate comprehensive compliance report with analytics."""
        try:
            # Check cache first
            cache_key = f"{standard.value}_{period_start.isoformat()}_{period_end.isoformat()}"
            cached_report = self._get_cached_report(cache_key)
            if cached_report:
                return Either.right(cached_report)
            
            # Query relevant events for the period
            events = await self.event_logger.query_events(
                filters={'compliance_standard': standard.value},
                time_range=(period_start, period_end)
            )
            
            # Filter events by compliance standard
            relevant_events = [e for e in events if e.matches_compliance_standard(standard)]
            
            # Analyze compliance violations
            violations = []
            risk_scores = []
            violation_details = []
            
            for event in relevant_events:
                # Check event against compliance rules
                event_violations = await self.compliance_monitor.monitor_event(event)
                
                if event_violations:
                    violations.extend(event_violations)
                    violation_details.append({
                        'event': event,
                        'rules_violated': event_violations
                    })
                
                # Calculate risk score for event
                event_risk = self._calculate_event_risk_score(event, standard)
                risk_scores.append(event_risk)
            
            # Calculate metrics
            total_events = len(relevant_events)
            violations_found = len(set(v.rule_id for v in violations))  # Unique violations
            average_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
            
            # Calculate compliance percentage
            compliance_percentage = self._calculate_compliance_percentage(
                total_events, violations_found, standard
            )
            
            # Generate findings
            findings = self._generate_detailed_findings(violation_details, standard)
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_comprehensive_recommendations(
                    violations, findings, standard, relevant_events
                )
            
            # Create comprehensive report
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                standard=standard,
                period_start=period_start,
                period_end=period_end,
                total_events=total_events,
                violations_found=violations_found,
                risk_score=average_risk_score,
                compliance_percentage=compliance_percentage,
                findings=findings,
                recommendations=recommendations,
                generated_at=datetime.now(UTC)
            )
            
            # Cache report
            self.report_cache[cache_key] = report
            
            logger.info(f"Generated {standard.value} compliance report: "
                       f"{compliance_percentage:.1f}% compliant, {violations_found} violations")
            
            return Either.right(report)
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return Either.left(AuditError.report_generation_failed(str(e)))
    
    async def generate_executive_summary(self, 
                                       standards: List[ComplianceStandard],
                                       period_start: datetime,
                                       period_end: datetime) -> Dict[str, Any]:
        """Generate executive summary across multiple compliance standards."""
        try:
            summary = {
                'period': {
                    'start': period_start.isoformat(),
                    'end': period_end.isoformat(),
                    'days': (period_end - period_start).days
                },
                'overall_compliance': {
                    'average_score': 0.0,
                    'total_violations': 0,
                    'critical_issues': 0,
                    'trends': {}
                },
                'standards': {},
                'key_metrics': {},
                'action_items': [],
                'generated_at': datetime.now(UTC).isoformat()
            }
            
            all_scores = []
            total_violations = 0
            critical_issues = 0
            
            # Generate reports for each standard
            for standard in standards:
                report_result = await self.generate_compliance_report(
                    standard, period_start, period_end
                )
                
                if report_result.is_right():
                    report = report_result.get_right()
                    
                    summary['standards'][standard.value] = {
                        'compliance_percentage': report.compliance_percentage,
                        'violations': report.violations_found,
                        'risk_score': report.risk_score,
                        'grade': report.get_compliance_grade(),
                        'status': 'compliant' if report.is_compliant() else 'non_compliant'
                    }
                    
                    all_scores.append(report.compliance_percentage)
                    total_violations += report.violations_found
                    
                    # Count critical issues
                    critical_findings = [f for f in report.findings 
                                       if f.get('severity') == 'critical']
                    critical_issues += len(critical_findings)
                    
                    # Add top recommendations as action items
                    if report.recommendations:
                        summary['action_items'].extend(
                            report.recommendations[:2]  # Top 2 recommendations per standard
                        )
            
            # Calculate overall metrics
            if all_scores:
                summary['overall_compliance']['average_score'] = sum(all_scores) / len(all_scores)
            
            summary['overall_compliance']['total_violations'] = total_violations
            summary['overall_compliance']['critical_issues'] = critical_issues
            
            # Generate key metrics
            summary['key_metrics'] = await self._generate_key_metrics(
                period_start, period_end, standards
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {}
    
    async def generate_trend_analysis(self, 
                                    standard: ComplianceStandard,
                                    periods: List[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        """Generate compliance trend analysis across multiple periods."""
        try:
            trends = {
                'standard': standard.value,
                'periods': [],
                'trends': {
                    'compliance_score': [],
                    'violation_count': [],
                    'risk_score': []
                },
                'analysis': {
                    'direction': 'stable',
                    'trend_strength': 0.0,
                    'prediction': {}
                },
                'insights': []
            }
            
            # Generate reports for each period
            for period_start, period_end in periods:
                report_result = await self.generate_compliance_report(
                    standard, period_start, period_end, include_recommendations=False
                )
                
                if report_result.is_right():
                    report = report_result.get_right()
                    
                    period_data = {
                        'start': period_start.isoformat(),
                        'end': period_end.isoformat(),
                        'compliance_percentage': report.compliance_percentage,
                        'violations': report.violations_found,
                        'risk_score': report.risk_score
                    }
                    
                    trends['periods'].append(period_data)
                    trends['trends']['compliance_score'].append(report.compliance_percentage)
                    trends['trends']['violation_count'].append(report.violations_found)
                    trends['trends']['risk_score'].append(report.risk_score)
            
            # Analyze trends
            if len(trends['trends']['compliance_score']) >= 2:
                trends['analysis'] = self._analyze_compliance_trends(trends['trends'])
                trends['insights'] = self._generate_trend_insights(trends['analysis'], standard)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return {}
    
    async def export_report(self, 
                          report: ComplianceReport, 
                          format: str = 'json',
                          file_path: Optional[str] = None) -> Either[AuditError, str]:
        """Export compliance report to specified format."""
        try:
            if format.lower() == 'json':
                return await self._export_json_report(report, file_path)
            elif format.lower() == 'html':
                return await self._export_html_report(report, file_path)
            elif format.lower() == 'csv':
                return await self._export_csv_report(report, file_path)
            elif format.lower() == 'pdf':
                return await self._export_pdf_report(report, file_path)
            else:
                return Either.left(AuditError.report_generation_failed(
                    f"Unsupported export format: {format}"
                ))
                
        except Exception as e:
            return Either.left(AuditError.report_generation_failed(f"Export failed: {e}"))
    
    def _calculate_event_risk_score(self, event: AuditEvent, standard: ComplianceStandard) -> float:
        """Calculate risk score for individual event in compliance context."""
        base_scores = {
            RiskLevel.LOW: 10.0,
            RiskLevel.MEDIUM: 30.0,
            RiskLevel.HIGH: 60.0,
            RiskLevel.CRITICAL: 90.0
        }
        
        base_score = base_scores.get(event.risk_level, 10.0)
        
        # Adjust based on event type
        high_risk_events = {
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.COMPLIANCE_VIOLATION,
            AuditEventType.DATA_DELETED,
            AuditEventType.SENSITIVE_DATA_ACCESS,
            AuditEventType.PRIVILEGE_ESCALATION
        }
        
        if event.event_type in high_risk_events:
            base_score *= 1.5
        
        # Adjust based on compliance standard
        if standard == ComplianceStandard.HIPAA and 'phi_access' in event.compliance_tags:
            base_score *= 1.3
        elif standard == ComplianceStandard.PCI_DSS and 'payment_data' in event.compliance_tags:
            base_score *= 1.4
        elif standard == ComplianceStandard.GDPR and 'personal_data' in event.compliance_tags:
            base_score *= 1.2
        
        return min(100.0, base_score)
    
    def _calculate_compliance_percentage(self, 
                                       total_events: int, 
                                       violations_found: int,
                                       standard: ComplianceStandard) -> float:
        """Calculate compliance percentage with standard-specific weighting."""
        if total_events == 0:
            return 100.0
        
        # Basic compliance calculation
        basic_compliance = max(0.0, (total_events - violations_found) / total_events * 100.0)
        
        # Apply standard-specific adjustments
        standard_weights = {
            ComplianceStandard.HIPAA: 0.9,      # More stringent
            ComplianceStandard.PCI_DSS: 0.85,   # Very stringent
            ComplianceStandard.GDPR: 0.9,       # More stringent
            ComplianceStandard.SOC2: 0.95,      # Standard
            ComplianceStandard.GENERAL: 1.0     # No adjustment
        }
        
        weight = standard_weights.get(standard, 1.0)
        adjusted_compliance = basic_compliance * weight
        
        return min(100.0, max(0.0, adjusted_compliance))
    
    def _generate_detailed_findings(self, 
                                  violation_details: List[Dict],
                                  standard: ComplianceStandard) -> List[Dict[str, Any]]:
        """Generate detailed findings from violation analysis."""
        findings = []
        
        # Group violations by rule
        rule_violations = {}
        for detail in violation_details:
            for rule in detail['rules_violated']:
                if rule.rule_id not in rule_violations:
                    rule_violations[rule.rule_id] = {
                        'rule': rule,
                        'events': [],
                        'count': 0
                    }
                rule_violations[rule.rule_id]['events'].append(detail['event'])
                rule_violations[rule.rule_id]['count'] += 1
        
        # Create findings for each violated rule
        for rule_id, violation_data in rule_violations.items():
            rule = violation_data['rule']
            events = violation_data['events']
            
            # Analyze violation pattern
            user_ids = set(e.user_id for e in events)
            time_span = max(e.timestamp for e in events) - min(e.timestamp for e in events)
            
            finding = {
                'rule_id': rule_id,
                'rule_name': rule.name,
                'standard': rule.standard.value,
                'severity': rule.severity.value,
                'violation_count': violation_data['count'],
                'affected_users': len(user_ids),
                'time_span_hours': time_span.total_seconds() / 3600,
                'description': rule.description,
                'first_occurrence': min(e.timestamp for e in events).isoformat(),
                'last_occurrence': max(e.timestamp for e in events).isoformat(),
                'pattern_analysis': self._analyze_violation_pattern(events),
                'risk_assessment': self._assess_finding_risk(rule, events),
                'remediation_priority': self._calculate_remediation_priority(rule, violation_data['count'])
            }
            
            findings.append(finding)
        
        # Sort by remediation priority
        findings.sort(key=lambda x: x['remediation_priority'], reverse=True)
        
        return findings
    
    def _generate_comprehensive_recommendations(self, 
                                              violations: List,
                                              findings: List[Dict],
                                              standard: ComplianceStandard,
                                              events: List[AuditEvent]) -> List[str]:
        """Generate comprehensive, actionable recommendations."""
        recommendations = []
        
        # General recommendations based on violations
        if violations:
            recommendations.append(
                "Conduct immediate review of all identified compliance violations "
                "and implement corrective actions within 48 hours"
            )
            
            high_severity_violations = [v for v in violations 
                                     if v.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if high_severity_violations:
                recommendations.append(
                    f"Prioritize {len(high_severity_violations)} high/critical severity violations "
                    "for immediate remediation and root cause analysis"
                )
        
        # Standard-specific recommendations
        if standard == ComplianceStandard.HIPAA:
            if any('phi_access' in e.compliance_tags for e in events):
                recommendations.extend([
                    "Review and strengthen PHI access controls and monitoring procedures",
                    "Implement mandatory PHI handling training for all staff with access",
                    "Consider implementing additional encryption for PHI at rest and in transit"
                ])
        
        elif standard == ComplianceStandard.GDPR:
            if any('personal_data' in e.compliance_tags for e in events):
                recommendations.extend([
                    "Review data processing activities and ensure proper consent documentation",
                    "Implement automated data retention policies to prevent violations",
                    "Conduct privacy impact assessments for all personal data processing"
                ])
        
        elif standard == ComplianceStandard.PCI_DSS:
            if any('payment_data' in e.compliance_tags for e in events):
                recommendations.extend([
                    "Implement additional payment data encryption and tokenization",
                    "Review and restrict access to payment processing systems",
                    "Conduct quarterly PCI DSS compliance assessments"
                ])
        
        # Pattern-based recommendations
        failed_auth_events = [e for e in events 
                            if e.event_type == AuditEventType.AUTHENTICATION_FAILURE]
        if len(failed_auth_events) > 10:
            recommendations.append(
                "Implement multi-factor authentication and account lockout policies "
                "to address high volume of authentication failures"
            )
        
        # Training and process recommendations
        if len(findings) > 5:
            recommendations.extend([
                "Provide comprehensive compliance training to all relevant personnel",
                "Implement regular compliance monitoring and reporting procedures",
                "Establish a compliance committee to oversee ongoing compliance efforts"
            ])
        
        # Technical recommendations
        if any(f.get('severity') == 'critical' for f in findings):
            recommendations.extend([
                "Implement real-time compliance monitoring with automated alerting",
                "Consider deploying data loss prevention (DLP) solutions",
                "Enhance logging and monitoring capabilities for better visibility"
            ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _analyze_violation_pattern(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze pattern in violation events."""
        if not events:
            return {}
        
        # Time pattern analysis
        timestamps = [e.timestamp for e in events]
        time_deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_deltas.append(delta)
        
        # User pattern analysis
        user_counts = {}
        for event in events:
            user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1
        
        # Action pattern analysis
        action_counts = {}
        for event in events:
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
        
        return {
            'frequency': 'high' if len(events) > 10 else 'medium' if len(events) > 3 else 'low',
            'time_clustering': 'clustered' if time_deltas and min(time_deltas) < 300 else 'distributed',
            'user_pattern': 'single_user' if len(user_counts) == 1 else 'multiple_users',
            'top_users': sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_actions': sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'span_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
        }
    
    def _assess_finding_risk(self, rule, events: List[AuditEvent]) -> str:
        """Assess risk level of a finding."""
        if rule.severity == RiskLevel.CRITICAL:
            return 'critical'
        elif rule.severity == RiskLevel.HIGH or len(events) > 10:
            return 'high'
        elif rule.severity == RiskLevel.MEDIUM or len(events) > 3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_remediation_priority(self, rule, violation_count: int) -> int:
        """Calculate remediation priority score."""
        severity_scores = {
            RiskLevel.CRITICAL: 100,
            RiskLevel.HIGH: 75,
            RiskLevel.MEDIUM: 50,
            RiskLevel.LOW: 25
        }
        
        base_score = severity_scores.get(rule.severity, 25)
        frequency_multiplier = min(2.0, 1.0 + (violation_count / 10))
        
        return int(base_score * frequency_multiplier)
    
    def _analyze_compliance_trends(self, trends: Dict[str, List]) -> Dict[str, Any]:
        """Analyze compliance trends over time."""
        compliance_scores = trends['compliance_score']
        
        if len(compliance_scores) < 2:
            return {'direction': 'insufficient_data'}
        
        # Calculate trend direction
        first_half = compliance_scores[:len(compliance_scores)//2]
        second_half = compliance_scores[len(compliance_scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 5:
            direction = 'improving'
        elif second_avg < first_avg - 5:
            direction = 'declining'
        else:
            direction = 'stable'
        
        # Calculate trend strength
        trend_strength = abs(second_avg - first_avg) / 100.0
        
        return {
            'direction': direction,
            'trend_strength': trend_strength,
            'first_period_avg': first_avg,
            'recent_period_avg': second_avg,
            'change_percentage': ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
        }
    
    def _generate_trend_insights(self, analysis: Dict[str, Any], standard: ComplianceStandard) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        direction = analysis.get('direction')
        if direction == 'improving':
            insights.append(f"Compliance for {standard.value} is showing positive improvement trends")
        elif direction == 'declining':
            insights.append(f"Compliance for {standard.value} is declining and requires immediate attention")
        elif direction == 'stable':
            insights.append(f"Compliance for {standard.value} remains stable")
        
        trend_strength = analysis.get('trend_strength', 0)
        if trend_strength > 0.2:
            insights.append("Strong trend detected - significant changes in compliance patterns")
        elif trend_strength > 0.1:
            insights.append("Moderate trend detected - monitoring recommended")
        
        return insights
    
    async def _generate_key_metrics(self, 
                                   period_start: datetime,
                                   period_end: datetime,
                                   standards: List[ComplianceStandard]) -> Dict[str, Any]:
        """Generate key compliance metrics."""
        try:
            # Get all events for the period
            all_events = await self.event_logger.query_events(
                filters={},
                time_range=(period_start, period_end)
            )
            
            metrics = {
                'total_events': len(all_events),
                'high_risk_events': len([e for e in all_events if e.is_high_risk()]),
                'unique_users': len(set(e.user_id for e in all_events)),
                'authentication_failures': len([e for e in all_events 
                                              if e.event_type == AuditEventType.AUTHENTICATION_FAILURE]),
                'data_access_events': len([e for e in all_events 
                                         if e.event_type in [AuditEventType.DATA_ACCESSED,
                                                           AuditEventType.SENSITIVE_DATA_ACCESS]]),
                'security_violations': len([e for e in all_events 
                                          if e.event_type == AuditEventType.SECURITY_VIOLATION])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating key metrics: {e}")
            return {}
    
    def _get_cached_report(self, cache_key: str) -> Optional[ComplianceReport]:
        """Get cached report if valid."""
        if cache_key in self.report_cache:
            report = self.report_cache[cache_key]
            if datetime.now(UTC) - report.generated_at < self.cache_expiry:
                return report
            else:
                del self.report_cache[cache_key]
        return None
    
    async def _export_json_report(self, 
                                report: ComplianceReport, 
                                file_path: Optional[str]) -> Either[AuditError, str]:
        """Export report to JSON format."""
        try:
            report_data = {
                'report_id': report.report_id,
                'standard': report.standard.value,
                'period_start': report.period_start.isoformat(),
                'period_end': report.period_end.isoformat(),
                'total_events': report.total_events,
                'violations_found': report.violations_found,
                'risk_score': report.risk_score,
                'compliance_percentage': report.compliance_percentage,
                'compliance_grade': report.get_compliance_grade(),
                'risk_category': report.get_risk_category(),
                'findings': report.findings,
                'recommendations': report.recommendations,
                'generated_at': report.generated_at.isoformat()
            }
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                return Either.right(file_path)
            else:
                return Either.right(json.dumps(report_data, indent=2, default=str))
                
        except Exception as e:
            return Either.left(AuditError.report_generation_failed(f"JSON export failed: {e}"))
    
    async def _export_html_report(self, 
                                report: ComplianceReport, 
                                file_path: Optional[str]) -> Either[AuditError, str]:
        """Export report to HTML format."""
        # Implementation for HTML export would go here
        # For now, return a simple HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Report - {report.standard.value}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .metric {{ margin: 10px 0; }}
                .finding {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; }}
                .critical {{ border-color: #ff0000; }}
                .high {{ border-color: #ff6600; }}
                .medium {{ border-color: #ffcc00; }}
                .low {{ border-color: #00cc00; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Report: {report.standard.value}</h1>
                <p>Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}</p>
                <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="metric">Compliance Score: {report.compliance_percentage:.1f}%</div>
            <div class="metric">Risk Score: {report.risk_score:.1f}/100</div>
            <div class="metric">Total Events: {report.total_events}</div>
            <div class="metric">Violations Found: {report.violations_found}</div>
            
            <h2>Findings</h2>
            {''.join([f'<div class="finding {finding.get("severity", "low")}">'
                     f'<h3>{finding.get("rule_name", "Unknown")}</h3>'
                     f'<p>{finding.get("description", "")}</p>'
                     f'<p>Violations: {finding.get("violation_count", 0)}</p>'
                     f'</div>' for finding in report.findings])}
            
            <h2>Recommendations</h2>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in report.recommendations])}
            </ul>
        </body>
        </html>
        """
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(html_content)
            return Either.right(file_path)
        else:
            return Either.right(html_content)
    
    async def _export_csv_report(self, 
                               report: ComplianceReport, 
                               file_path: Optional[str]) -> Either[AuditError, str]:
        """Export report findings to CSV format."""
        try:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Rule ID', 'Rule Name', 'Standard', 'Severity', 
                'Violation Count', 'Description', 'First Occurrence', 'Last Occurrence'
            ])
            
            # Write findings
            for finding in report.findings:
                writer.writerow([
                    finding.get('rule_id', ''),
                    finding.get('rule_name', ''),
                    finding.get('standard', ''),
                    finding.get('severity', ''),
                    finding.get('violation_count', 0),
                    finding.get('description', ''),
                    finding.get('first_occurrence', ''),
                    finding.get('last_occurrence', '')
                ])
            
            csv_content = output.getvalue()
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(csv_content)
                return Either.right(file_path)
            else:
                return Either.right(csv_content)
                
        except Exception as e:
            return Either.left(AuditError.report_generation_failed(f"CSV export failed: {e}"))
    
    async def _export_pdf_report(self, 
                               report: ComplianceReport, 
                               file_path: Optional[str]) -> Either[AuditError, str]:
        """Export report to PDF format (placeholder)."""
        # PDF generation would require additional dependencies
        # For now, return error indicating it's not implemented
        return Either.left(AuditError.report_generation_failed(
            "PDF export not implemented - requires additional dependencies"
        ))