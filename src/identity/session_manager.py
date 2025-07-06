"""
Session Manager - TASK_67 Phase 2 Core Identity Engine

Multi-user session management with context switching, session persistence, and security monitoring.
Handles user session lifecycle, context isolation, and cross-session data management.

Architecture: Session Management + Design by Contract + Type Safety + Security Monitoring
Performance: <50ms session operations, <100ms context switching, <200ms session validation
Security: Session isolation, secure tokens, activity monitoring, concurrent session management
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..core.contracts import require
from ..core.either import Either
from ..core.user_identity_architecture import (
    IdentityError,
    UserProfileId,
    UserSession,
    UserSessionId,
)

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session state enumeration."""

    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class SessionContext:
    """Session context information."""

    session_id: UserSessionId
    user_profile_id: UserProfileId
    environment_data: dict[str, Any]
    preferences: dict[str, Any]
    temporary_data: dict[str, Any]
    isolation_level: str  # strict|moderate|relaxed


@dataclass(frozen=True)
class SessionTransition:
    """Session state transition record."""

    transition_id: str
    session_id: UserSessionId
    from_state: SessionState
    to_state: SessionState
    timestamp: datetime
    reason: str
    automated: bool


@dataclass(frozen=True)
class SessionSwitchRequest:
    """Request to switch user context."""

    current_session_id: UserSessionId | None
    target_user_profile_id: UserProfileId
    preserve_context: bool
    security_validation: bool
    switch_reason: str


class SessionManager:
    """Multi-user session management and context switching system."""

    def __init__(self):
        self.active_sessions: dict[UserSessionId, UserSession] = {}
        self.session_contexts: dict[UserSessionId, SessionContext] = {}
        self.session_transitions: list[SessionTransition] = []
        self.user_sessions: dict[UserProfileId, set[UserSessionId]] = {}
        self.session_locks: dict[UserSessionId, asyncio.Lock] = {}
        self.context_isolation_policies: dict[str, dict[str, Any]] = {}
        self.concurrent_session_limits: dict[UserProfileId, int] = {}

        # Initialize default policies
        self._initialize_session_policies()

    def _initialize_session_policies(self) -> None:
        """Initialize default session policies."""
        self.context_isolation_policies = {
            "strict": {
                "cross_session_data_sharing": False,
                "environment_isolation": True,
                "preference_isolation": True,
                "memory_isolation": True,
            },
            "moderate": {
                "cross_session_data_sharing": True,
                "environment_isolation": True,
                "preference_isolation": False,
                "memory_isolation": False,
            },
            "relaxed": {
                "cross_session_data_sharing": True,
                "environment_isolation": False,
                "preference_isolation": False,
                "memory_isolation": False,
            },
        }

    @require(lambda user_session: user_session.session_id is not None)
    async def create_session(
        self,
        user_session: UserSession,
        context_data: dict[str, Any] | None = None,
        isolation_level: str = "moderate",
    ) -> Either[IdentityError, SessionContext]:
        """Create a new user session with context."""
        try:
            session_id = user_session.session_id
            user_profile_id = user_session.user_profile_id

            # Check concurrent session limits
            user_session_count = len(self.user_sessions.get(user_profile_id, set()))
            session_limit = self.concurrent_session_limits.get(
                user_profile_id, 5
            )  # Default limit

            if user_session_count >= session_limit:
                return Either.error(
                    IdentityError(
                        f"Maximum concurrent sessions exceeded: {session_limit}",
                        "SESSION_LIMIT_EXCEEDED",
                    )
                )

            # Store session
            self.active_sessions[session_id] = user_session

            # Create session lock
            self.session_locks[session_id] = asyncio.Lock()

            # Track user sessions
            if user_profile_id not in self.user_sessions:
                self.user_sessions[user_profile_id] = set()
            self.user_sessions[user_profile_id].add(session_id)

            # Create session context
            session_context = SessionContext(
                session_id=session_id,
                user_profile_id=user_profile_id,
                environment_data=context_data or {},
                preferences={},
                temporary_data={},
                isolation_level=isolation_level,
            )

            self.session_contexts[session_id] = session_context

            # Log session creation
            await self._log_session_transition(
                session_id,
                SessionState.TERMINATED,  # From no session
                SessionState.ACTIVE,
                "session_created",
                True,
            )

            logger.info(f"Created session {session_id} for user {user_profile_id}")
            return Either.success(session_context)

        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return Either.error(
                IdentityError(
                    f"Session creation error: {str(e)}", "SESSION_CREATION_ERROR"
                )
            )

    @require(lambda session_id: session_id is not None)
    async def get_session(
        self, session_id: UserSessionId
    ) -> Either[IdentityError, UserSession]:
        """Get session by ID with validation."""
        try:
            if session_id not in self.active_sessions:
                return Either.error(
                    IdentityError(
                        f"Session not found: {session_id}", "SESSION_NOT_FOUND"
                    )
                )

            session = self.active_sessions[session_id]

            # Validate session is still active
            if session.is_expired():
                await self._expire_session(session_id, "session_expired")
                return Either.error(IdentityError.session_expired(session_id))

            # Update last activity
            updated_session = session.update_activity()
            self.active_sessions[session_id] = updated_session

            return Either.success(updated_session)

        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return Either.error(
                IdentityError(
                    f"Session retrieval error: {str(e)}", "SESSION_RETRIEVAL_ERROR"
                )
            )

    @require(lambda session_id: session_id is not None)
    async def get_session_context(
        self, session_id: UserSessionId
    ) -> Either[IdentityError, SessionContext]:
        """Get session context information."""
        try:
            if session_id not in self.session_contexts:
                return Either.error(
                    IdentityError(
                        f"Session context not found: {session_id}",
                        "SESSION_CONTEXT_NOT_FOUND",
                    )
                )

            # Verify session is still active
            session_result = await self.get_session(session_id)
            if session_result.is_error():
                return Either.error(session_result.error)

            context = self.session_contexts[session_id]
            return Either.success(context)

        except Exception as e:
            logger.error(f"Session context retrieval failed: {e}")
            return Either.error(
                IdentityError(
                    f"Context retrieval error: {str(e)}", "CONTEXT_RETRIEVAL_ERROR"
                )
            )

    @require(lambda switch_request: switch_request.target_user_profile_id is not None)
    async def switch_user_context(
        self, switch_request: SessionSwitchRequest
    ) -> Either[IdentityError, SessionContext]:
        """Switch user context with security validation."""
        try:
            current_session_id = switch_request.current_session_id
            target_user_id = switch_request.target_user_profile_id

            # Security validation if required
            if switch_request.security_validation:
                security_result = await self._validate_context_switch_security(
                    switch_request
                )
                if security_result.is_error():
                    return Either.error(security_result.error)

            # Preserve current context if requested
            preserved_context = {}
            if switch_request.preserve_context and current_session_id:
                current_context_result = await self.get_session_context(
                    current_session_id
                )
                if current_context_result.is_success():
                    preserved_context = (
                        current_context_result.value.temporary_data.copy()
                    )

            # Get target user sessions
            target_sessions = self.user_sessions.get(target_user_id, set())

            if target_sessions:
                # Switch to existing session for target user
                target_session_id = next(iter(target_sessions))  # Get first session

                # Update context with preserved data
                if preserved_context:
                    await self._merge_session_context(
                        target_session_id, preserved_context
                    )

                context_result = await self.get_session_context(target_session_id)
                if context_result.is_success():
                    # Log context switch
                    await self._log_session_transition(
                        target_session_id,
                        SessionState.IDLE,
                        SessionState.ACTIVE,
                        f"context_switch_{switch_request.switch_reason}",
                        True,
                    )

                    logger.info(
                        f"Switched to existing session {target_session_id} for user {target_user_id}"
                    )
                    return context_result

            # No existing session, need to create one
            return Either.error(
                IdentityError(
                    f"No active session found for user {target_user_id}",
                    "NO_TARGET_SESSION",
                )
            )

        except Exception as e:
            logger.error(f"Context switch failed: {e}")
            return Either.error(
                IdentityError(f"Context switch error: {str(e)}", "CONTEXT_SWITCH_ERROR")
            )

    async def _validate_context_switch_security(
        self, switch_request: SessionSwitchRequest
    ) -> Either[IdentityError, bool]:
        """Validate security requirements for context switching."""
        try:
            # Check if current user has permission to switch to target user
            current_session_id = switch_request.current_session_id

            if current_session_id:
                current_session_result = await self.get_session(current_session_id)
                if current_session_result.is_error():
                    return Either.error(current_session_result.error)

                current_session = current_session_result.value
                current_permissions = current_session.authentication_result.permissions

                # Check for admin permissions or user management rights
                if (
                    "admin" not in current_permissions
                    and "user_management" not in current_permissions
                ):
                    return Either.error(
                        IdentityError(
                            "Insufficient permissions for context switching",
                            "INSUFFICIENT_PERMISSIONS",
                        )
                    )

            return Either.success(True)

        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return Either.error(
                IdentityError(
                    f"Security validation error: {str(e)}", "SECURITY_VALIDATION_ERROR"
                )
            )

    async def _merge_session_context(
        self, session_id: UserSessionId, preserved_data: dict[str, Any]
    ) -> None:
        """Merge preserved context data into target session."""
        try:
            if session_id in self.session_contexts:
                current_context = self.session_contexts[session_id]

                # Create updated context with merged data
                updated_context = SessionContext(
                    session_id=current_context.session_id,
                    user_profile_id=current_context.user_profile_id,
                    environment_data=current_context.environment_data,
                    preferences=current_context.preferences,
                    temporary_data={**current_context.temporary_data, **preserved_data},
                    isolation_level=current_context.isolation_level,
                )

                self.session_contexts[session_id] = updated_context

        except Exception as e:
            logger.error(f"Context merge failed: {e}")

    @require(lambda session_id: session_id is not None)
    async def terminate_session(
        self, session_id: UserSessionId, reason: str = "user_logout"
    ) -> Either[IdentityError, bool]:
        """Terminate a user session."""
        try:
            if session_id not in self.active_sessions:
                return Either.error(
                    IdentityError(
                        f"Session not found: {session_id}", "SESSION_NOT_FOUND"
                    )
                )

            # Use session lock to prevent race conditions
            async with self.session_locks[session_id]:
                session = self.active_sessions[session_id]
                user_profile_id = session.user_profile_id

                # Remove from active sessions
                del self.active_sessions[session_id]

                # Remove session context
                if session_id in self.session_contexts:
                    del self.session_contexts[session_id]

                # Remove from user sessions tracking
                if user_profile_id in self.user_sessions:
                    self.user_sessions[user_profile_id].discard(session_id)
                    if not self.user_sessions[user_profile_id]:
                        del self.user_sessions[user_profile_id]

                # Remove session lock
                del self.session_locks[session_id]

                # Log session termination
                await self._log_session_transition(
                    session_id,
                    SessionState.ACTIVE,
                    SessionState.TERMINATED,
                    reason,
                    True,
                )

                logger.info(
                    f"Terminated session {session_id} for user {user_profile_id}, reason: {reason}"
                )
                return Either.success(True)

        except Exception as e:
            logger.error(f"Session termination failed: {e}")
            return Either.error(
                IdentityError(
                    f"Session termination error: {str(e)}", "SESSION_TERMINATION_ERROR"
                )
            )

    async def _expire_session(self, session_id: UserSessionId, reason: str) -> None:
        """Mark session as expired and clean up."""
        try:
            await self.terminate_session(session_id, f"expired_{reason}")
        except Exception as e:
            logger.error(f"Session expiration failed: {e}")

    async def _log_session_transition(
        self,
        session_id: UserSessionId,
        from_state: SessionState,
        to_state: SessionState,
        reason: str,
        automated: bool,
    ) -> None:
        """Log session state transition."""
        try:
            transition = SessionTransition(
                transition_id=secrets.token_hex(16),
                session_id=session_id,
                from_state=from_state,
                to_state=to_state,
                timestamp=datetime.now(UTC),
                reason=reason,
                automated=automated,
            )

            self.session_transitions.append(transition)

            # Keep only recent transitions (last 1000)
            if len(self.session_transitions) > 1000:
                self.session_transitions = self.session_transitions[-1000:]

        except Exception as e:
            logger.error(f"Session transition logging failed: {e}")

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count."""
        try:
            expired_sessions = []
            datetime.now(UTC)

            for session_id, session in self.active_sessions.items():
                if session.is_expired():
                    expired_sessions.append(session_id)

            # Clean up expired sessions
            cleanup_count = 0
            for session_id in expired_sessions:
                result = await self.terminate_session(session_id, "cleanup_expired")
                if result.is_success():
                    cleanup_count += 1

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} expired sessions")

            return cleanup_count

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return 0

    async def get_user_sessions(
        self, user_profile_id: UserProfileId
    ) -> Either[IdentityError, list[dict[str, Any]]]:
        """Get all active sessions for a user."""
        try:
            user_session_ids = self.user_sessions.get(user_profile_id, set())
            sessions_info = []

            for session_id in user_session_ids:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    context = self.session_contexts.get(session_id)

                    session_info = {
                        "session_id": session_id,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        "security_level": session.security_level.value,
                        "activity_count": session.activity_count,
                        "isolation_level": context.isolation_level
                        if context
                        else "unknown",
                    }
                    sessions_info.append(session_info)

            return Either.success(sessions_info)

        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return Either.error(
                IdentityError(
                    f"User sessions retrieval error: {str(e)}", "USER_SESSIONS_ERROR"
                )
            )

    async def get_session_analytics(self) -> dict[str, Any]:
        """Get session management analytics."""
        try:
            current_time = datetime.now(UTC)

            # Calculate session statistics
            total_sessions = len(self.active_sessions)
            unique_users = len(self.user_sessions)

            # Calculate session durations
            session_durations = []
            for session in self.active_sessions.values():
                duration = (
                    current_time - session.created_at
                ).total_seconds() / 60  # minutes
                session_durations.append(duration)

            avg_session_duration = (
                sum(session_durations) / len(session_durations)
                if session_durations
                else 0
            )

            # Count recent transitions
            recent_transitions = [
                t
                for t in self.session_transitions
                if (current_time - t.timestamp).total_seconds() < 3600  # Last hour
            ]

            analytics = {
                "total_active_sessions": total_sessions,
                "unique_active_users": unique_users,
                "average_session_duration_minutes": round(avg_session_duration, 2),
                "recent_transitions_count": len(recent_transitions),
                "concurrent_session_utilization": {
                    user_id: len(sessions)
                    for user_id, sessions in self.user_sessions.items()
                },
                "isolation_level_distribution": {},
                "analytics_timestamp": current_time.isoformat(),
            }

            # Calculate isolation level distribution
            isolation_counts = {}
            for context in self.session_contexts.values():
                level = context.isolation_level
                isolation_counts[level] = isolation_counts.get(level, 0) + 1

            analytics["isolation_level_distribution"] = isolation_counts

            return analytics

        except Exception as e:
            logger.error(f"Session analytics failed: {e}")
            return {}

    async def set_concurrent_session_limit(
        self, user_profile_id: UserProfileId, limit: int
    ) -> Either[IdentityError, bool]:
        """Set concurrent session limit for a user."""
        try:
            if limit < 1:
                return Either.error(
                    IdentityError("Session limit must be at least 1", "INVALID_LIMIT")
                )

            self.concurrent_session_limits[user_profile_id] = limit

            # Check if user currently exceeds new limit
            current_sessions = len(self.user_sessions.get(user_profile_id, set()))
            if current_sessions > limit:
                # Terminate oldest sessions to meet limit
                user_session_ids = list(self.user_sessions.get(user_profile_id, set()))
                sessions_to_terminate = user_session_ids[: current_sessions - limit]

                for session_id in sessions_to_terminate:
                    await self.terminate_session(session_id, "limit_enforcement")

            logger.info(f"Set session limit for user {user_profile_id}: {limit}")
            return Either.success(True)

        except Exception as e:
            logger.error(f"Failed to set session limit: {e}")
            return Either.error(
                IdentityError(
                    f"Session limit setting error: {str(e)}", "LIMIT_SETTING_ERROR"
                )
            )
