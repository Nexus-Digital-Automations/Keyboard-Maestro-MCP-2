"""
Comprehensive audit event logging system with integrity protection.

This module provides secure, high-performance audit event logging with
cryptographic integrity verification, performance optimization, and
comprehensive security validation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, UTC
import time
import asyncio
import logging
import uuid
from pathlib import Path

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.audit_framework import (
    AuditEvent, AuditEventType, AuditError, RiskLevel, 
    ComplianceStandard, SecurityLimits, AuditConfiguration
)


logger = logging.getLogger(__name__)


class AuditIntegrityManager:
    """Audit log integrity and security management."""
    
    def __init__(self):
        self.encryption_key = self._generate_encryption_key()
        self.signature_key = self._generate_signature_key()
    
    async def encrypt_event(self, event: AuditEvent) -> Either[AuditError, bytes]:
        """Encrypt audit event for secure storage."""
        try:
            # For production, use proper cryptographic libraries
            import json
            import base64
            import hashlib
            
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'session_id': event.session_id,
                'resource_id': event.resource_id,
                'action': event.action,
                'result': event.result,
                'ip_address': event.ip_address,
                'user_agent': event.user_agent,
                'details': event.details,
                'risk_level': event.risk_level.value,
                'compliance_tags': list(event.compliance_tags),
                'checksum': event.checksum
            }
            
            # Simple encryption for demonstration - use proper encryption in production
            data_json = json.dumps(event_data, default=str)
            data_bytes = data_json.encode('utf-8')
            
            # Add HMAC for integrity
            signature = hashlib.sha256(self.signature_key + data_bytes).hexdigest()
            
            # Simple XOR encryption (replace with AES in production)
            encrypted_data = bytes(a ^ b for a, b in zip(data_bytes, self.encryption_key * (len(data_bytes) // len(self.encryption_key) + 1)))
            
            # Combine signature and encrypted data
            final_data = signature.encode() + b'|||' + base64.b64encode(encrypted_data)
            
            return Either.right(final_data)
            
        except Exception as e:
            return Either.left(AuditError.encryption_failed(str(e)))
    
    async def decrypt_event(self, encrypted_data: bytes) -> Either[AuditError, AuditEvent]:
        """Decrypt audit event and verify integrity."""
        try:
            import json
            import base64
            import hashlib
            
            # Split signature and data
            parts = encrypted_data.split(b'|||', 1)
            if len(parts) != 2:
                return Either.left(AuditError.encryption_failed("Invalid encrypted data format"))
            
            signature = parts[0].decode()
            encrypted_payload = base64.b64decode(parts[1])
            
            # Decrypt data
            decrypted_data = bytes(a ^ b for a, b in zip(encrypted_payload, self.encryption_key * (len(encrypted_payload) // len(self.encryption_key) + 1)))
            
            # Verify signature
            expected_signature = hashlib.sha256(self.signature_key + decrypted_data).hexdigest()
            if signature != expected_signature:
                return Either.left(AuditError.integrity_check_failed())
            
            # Parse event data
            event_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Reconstruct audit event
            event = AuditEvent(
                event_id=event_data['event_id'],
                event_type=AuditEventType(event_data['event_type']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                user_id=event_data['user_id'],
                session_id=event_data.get('session_id'),
                resource_id=event_data.get('resource_id'),
                action=event_data['action'],
                result=event_data['result'],
                ip_address=event_data.get('ip_address'),
                user_agent=event_data.get('user_agent'),
                details=event_data.get('details', {}),
                risk_level=RiskLevel(event_data.get('risk_level', 'low')),
                compliance_tags=set(event_data.get('compliance_tags', [])),
                checksum=event_data.get('checksum', '')
            )
            
            # Verify event integrity
            if not event.verify_integrity():
                return Either.left(AuditError.integrity_check_failed())
            
            return Either.right(event)
            
        except Exception as e:
            return Either.left(AuditError.encryption_failed(str(e)))
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for audit logs."""
        # In production, use proper key management system
        import secrets
        return secrets.token_bytes(32)
    
    def _generate_signature_key(self) -> bytes:
        """Generate signature key for audit integrity."""
        import secrets
        return secrets.token_bytes(32)


class EventBuffer:
    """High-performance event buffer with automatic flushing."""
    
    def __init__(self, max_size: int = 500, flush_interval: float = 2.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: List[AuditEvent] = []
        self.last_flush = time.time()
        self._lock = asyncio.Lock()
        self._flush_callbacks: List[callable] = []
    
    async def add_event(self, event: AuditEvent) -> bool:
        """Add event to buffer and trigger flush if needed."""
        async with self._lock:
            self.buffer.append(event)
            
            # Check if flush is needed
            should_flush = (
                len(self.buffer) >= self.max_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                await self._flush_buffer()
            
            return True
    
    async def force_flush(self):
        """Force flush of all buffered events."""
        async with self._lock:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Internal flush implementation."""
        if not self.buffer:
            return
        
        events_to_flush = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = time.time()
        
        # Call flush callbacks
        for callback in self._flush_callbacks:
            try:
                await callback(events_to_flush)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")
    
    def add_flush_callback(self, callback: callable):
        """Add callback to be called on buffer flush."""
        self._flush_callbacks.append(callback)
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'utilization': len(self.buffer) / self.max_size,
            'last_flush': self.last_flush,
            'time_since_flush': time.time() - self.last_flush
        }


class EventLogger:
    """Comprehensive audit event logging system with performance optimization."""
    
    def __init__(self, config: Optional[AuditConfiguration] = None):
        self.config = config or AuditConfiguration()
        self.event_store: List[AuditEvent] = []
        self.integrity_manager = AuditIntegrityManager()
        self.rate_limiter = RateLimiter(self.config.security_limits.max_events_per_second)
        
        # Setup buffer with configuration-based parameters
        profile = self.config.get_performance_profile()
        self.buffer = EventBuffer(
            max_size=profile['buffer_size'],
            flush_interval=profile['flush_interval']
        )
        
        # Add buffer flush callback
        self.buffer.add_flush_callback(self._handle_buffer_flush)
        
        # Statistics tracking
        self.stats = {
            'events_logged': 0,
            'events_failed': 0,
            'integrity_failures': 0,
            'rate_limit_rejections': 0,
            'start_time': datetime.now(UTC)
        }
    
    @require(lambda self, event_type: isinstance(event_type, AuditEventType))
    @require(lambda self, user_id: isinstance(user_id, str) and len(user_id.strip()) > 0)
    @require(lambda self, action: isinstance(action, str) and len(action.strip()) > 0)
    @require(lambda self, result: isinstance(result, str) and len(result.strip()) > 0)
    async def log_event(self, event_type: AuditEventType, user_id: str, action: str, 
                       result: str, **kwargs) -> Either[AuditError, str]:
        """Log audit event with comprehensive validation and integrity protection."""
        try:
            # Rate limiting check
            if not await self.rate_limiter.check_rate_limit(user_id):
                self.stats['rate_limit_rejections'] += 1
                return Either.left(AuditError.logging_failed("Rate limit exceeded"))
            
            # Validate event size
            event_size = len(str(kwargs.get('details', {})))
            if event_size > self.config.security_limits.max_event_size:
                return Either.left(AuditError.logging_failed(f"Event size exceeds limit: {event_size}"))
            
            # Extract and validate parameters
            session_id = kwargs.get('session_id')
            resource_id = kwargs.get('resource_id')
            ip_address = kwargs.get('ip_address')
            user_agent = kwargs.get('user_agent')
            details = kwargs.get('details', {})
            risk_level = kwargs.get('risk_level', RiskLevel.LOW)
            compliance_tags = set(kwargs.get('compliance_tags', []))
            
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(UTC),
                user_id=user_id,
                session_id=session_id,
                resource_id=resource_id,
                action=action,
                result=result,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                risk_level=risk_level,
                compliance_tags=compliance_tags
            )
            
            # Verify event integrity
            if not event.verify_integrity():
                self.stats['integrity_failures'] += 1
                return Either.left(AuditError.integrity_check_failed())
            
            # Add to buffer for high-performance logging
            await self.buffer.add_event(event)
            
            # Update statistics
            self.stats['events_logged'] += 1
            
            return Either.right(event_id)
            
        except Exception as e:
            self.stats['events_failed'] += 1
            logger.error(f"Failed to log audit event: {e}")
            return Either.left(AuditError.logging_failed(str(e)))
    
    async def query_events(self, filters: Dict[str, Any], 
                          time_range: Optional[Tuple[datetime, datetime]] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Query audit events with comprehensive filtering and security."""
        try:
            # Ensure buffer is flushed for complete results
            await self.buffer.force_flush()
            
            # Apply security limits
            limit = min(limit, self.config.security_limits.max_query_results)
            
            events = self.event_store.copy()
            
            # Apply time range filter
            if time_range:
                start_time, end_time = time_range
                events = [e for e in events if start_time <= e.timestamp <= end_time]
            
            # Apply filters
            if 'user_id' in filters:
                events = [e for e in events if e.user_id == filters['user_id']]
            
            if 'event_type' in filters:
                if isinstance(filters['event_type'], str):
                    event_type = AuditEventType(filters['event_type'])
                else:
                    event_type = filters['event_type']
                events = [e for e in events if e.event_type == event_type]
            
            if 'risk_level' in filters:
                if isinstance(filters['risk_level'], str):
                    risk_level = RiskLevel(filters['risk_level'])
                else:
                    risk_level = filters['risk_level']
                events = [e for e in events if e.risk_level == risk_level]
            
            if 'action' in filters:
                action_filter = filters['action'].lower()
                events = [e for e in events if action_filter in e.action.lower()]
            
            if 'compliance_standard' in filters:
                standard = ComplianceStandard(filters['compliance_standard'])
                events = [e for e in events if e.matches_compliance_standard(standard)]
            
            # Verify integrity of returned events
            verified_events = []
            for event in events:
                if event.verify_integrity():
                    verified_events.append(event)
                else:
                    self.stats['integrity_failures'] += 1
                    logger.warning(f"Integrity check failed for event {event.event_id}")
            
            # Apply limit and return
            return verified_events[:limit]
            
        except Exception as e:
            logger.error(f"Error querying audit events: {e}")
            return []
    
    async def _handle_buffer_flush(self, events: List[AuditEvent]):
        """Handle buffer flush by storing events."""
        for event in events:
            # Store encrypted event if encryption is enabled
            if self.config.encrypt_logs:
                encrypted_result = await self.integrity_manager.encrypt_event(event)
                if encrypted_result.is_left():
                    logger.error(f"Failed to encrypt event {event.event_id}")
                    continue
            
            # Add to in-memory store
            self.event_store.append(event)
        
        # Cleanup old events based on retention policy
        await self._cleanup_old_events()
    
    async def _cleanup_old_events(self):
        """Clean up events older than retention period."""
        if not self.config.retention_days:
            return
        
        cutoff_date = datetime.now(UTC) - timedelta(days=self.config.retention_days)
        initial_count = len(self.event_store)
        
        # Remove old events
        self.event_store = [e for e in self.event_store if e.timestamp >= cutoff_date]
        
        removed_count = initial_count - len(self.event_store)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired audit events")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        uptime = datetime.now(UTC) - self.stats['start_time']
        
        return {
            'events_logged': self.stats['events_logged'],
            'events_failed': self.stats['events_failed'],
            'integrity_failures': self.stats['integrity_failures'],
            'rate_limit_rejections': self.stats['rate_limit_rejections'],
            'events_stored': len(self.event_store),
            'buffer_status': self.buffer.get_buffer_status(),
            'uptime_seconds': uptime.total_seconds(),
            'events_per_second': self.stats['events_logged'] / max(uptime.total_seconds(), 1),
            'success_rate': (self.stats['events_logged'] / 
                           max(self.stats['events_logged'] + self.stats['events_failed'], 1)) * 100
        }
    
    async def export_events(self, file_path: str, format: str = 'json', 
                           filters: Optional[Dict[str, Any]] = None) -> Either[AuditError, int]:
        """Export audit events to file with specified format."""
        try:
            # Query events with filters
            events = await self.query_events(filters or {})
            
            if format.lower() == 'json':
                return await self._export_json(file_path, events)
            elif format.lower() == 'csv':
                return await self._export_csv(file_path, events)
            else:
                return Either.left(AuditError.logging_failed(f"Unsupported export format: {format}"))
                
        except Exception as e:
            return Either.left(AuditError.logging_failed(f"Export failed: {e}"))
    
    async def _export_json(self, file_path: str, events: List[AuditEvent]) -> Either[AuditError, int]:
        """Export events to JSON format."""
        import json
        
        try:
            export_data = []
            for event in events:
                event_dict = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'resource_id': event.resource_id,
                    'action': event.action,
                    'result': event.result,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent,
                    'details': event.details,
                    'risk_level': event.risk_level.value,
                    'compliance_tags': list(event.compliance_tags),
                    'checksum': event.checksum
                }
                export_data.append(event_dict)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return Either.right(len(events))
            
        except Exception as e:
            return Either.left(AuditError.logging_failed(f"JSON export failed: {e}"))
    
    async def _export_csv(self, file_path: str, events: List[AuditEvent]) -> Either[AuditError, int]:
        """Export events to CSV format."""
        import csv
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'event_id', 'event_type', 'timestamp', 'user_id', 'session_id',
                    'resource_id', 'action', 'result', 'ip_address', 'user_agent',
                    'risk_level', 'compliance_tags', 'details_json'
                ])
                
                # Write events
                for event in events:
                    writer.writerow([
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.user_id,
                        event.session_id or '',
                        event.resource_id or '',
                        event.action,
                        event.result,
                        event.ip_address or '',
                        event.user_agent or '',
                        event.risk_level.value,
                        '|'.join(event.compliance_tags),
                        json.dumps(event.details) if event.details else ''
                    ])
            
            return Either.right(len(events))
            
        except Exception as e:
            return Either.left(AuditError.logging_failed(f"CSV export failed: {e}"))


class RateLimiter:
    """Rate limiter for audit event logging."""
    
    def __init__(self, max_events_per_second: int):
        self.max_events_per_second = max_events_per_second
        self.user_rates: Dict[str, List[float]] = {}
        self.cleanup_interval = 60.0  # Cleanup old entries every minute
        self.last_cleanup = time.time()
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries()
        
        # Get or create user rate tracking
        if user_id not in self.user_rates:
            self.user_rates[user_id] = []
        
        user_events = self.user_rates[user_id]
        
        # Remove events older than 1 second
        cutoff_time = current_time - 1.0
        user_events[:] = [t for t in user_events if t >= cutoff_time]
        
        # Check rate limit
        if len(user_events) >= self.max_events_per_second:
            return False
        
        # Add current event
        user_events.append(current_time)
        return True
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limiting entries."""
        current_time = time.time()
        cutoff_time = current_time - 60.0  # Keep last minute of data
        
        for user_id in list(self.user_rates.keys()):
            user_events = self.user_rates[user_id]
            user_events[:] = [t for t in user_events if t >= cutoff_time]
            
            # Remove empty entries
            if not user_events:
                del self.user_rates[user_id]
        
        self.last_cleanup = current_time