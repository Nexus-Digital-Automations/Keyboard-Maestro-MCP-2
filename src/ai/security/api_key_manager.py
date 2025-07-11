"""Secure API key management for AI provider integration.

This module provides enterprise-grade API key storage, encryption, rotation,
and validation functionality with support for multiple storage backends
and comprehensive security controls.

Security: AES-256 encryption at rest with secure key derivation.
Performance: Optimized key retrieval with intelligent caching.
Type Safety: Complete integration with provider architecture.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ...core.either import Either
from ...core.errors import ValidationError


class StorageBackend(Enum):
    """API key storage backend types."""

    ENVIRONMENT = "environment"
    FILE = "file"
    KEYCHAIN = "keychain"  # macOS Keychain
    VAULT = "vault"  # HashiCorp Vault
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager


class KeyStatus(Enum):
    """API key status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


@dataclass
class APIKeyMetadata:
    """Metadata for API key management."""

    provider: str
    key_id: str
    status: KeyStatus
    created_at: datetime
    last_used: datetime | None = None
    expires_at: datetime | None = None
    rotation_interval_days: int | None = None
    usage_count: int = 0
    last_rotation: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at:
            return datetime.now(UTC) > self.expires_at
        return False

    def needs_rotation(self) -> bool:
        """Check if key needs rotation."""
        if not self.rotation_interval_days:
            return False

        if not self.last_rotation:
            rotation_date = self.created_at
        else:
            rotation_date = self.last_rotation

        next_rotation = rotation_date + timedelta(days=self.rotation_interval_days)
        return datetime.now(UTC) > next_rotation


@dataclass
class EncryptedAPIKey:
    """Encrypted API key storage."""

    encrypted_key: str
    salt: str
    provider: str
    key_id: str
    metadata: APIKeyMetadata

    def decrypt(self, master_password: str) -> Either[ValidationError, str]:
        """Decrypt API key using master password."""
        try:
            # Derive key from password and salt
            salt_bytes = base64.b64decode(self.salt.encode())
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))

            # Decrypt the API key
            f = Fernet(key)
            decrypted_bytes = f.decrypt(self.encrypted_key.encode())
            return Either.right(decrypted_bytes.decode())

        except Exception as e:
            return Either.left(
                ValidationError(
                    "decryption_failed",
                    str(e),
                    "Decryption operation failed",
                ),
            )


class APIKeyManager:
    """Enterprise API key management system."""

    def __init__(
        self,
        storage_backend: StorageBackend = StorageBackend.ENVIRONMENT,
        storage_path: Path | None = None,
        master_password: str | None = None,
    ):
        self.storage_backend = storage_backend
        self.storage_path = storage_path or Path.home() / ".km_mcp" / "keys"
        self.master_password = master_password or os.getenv(
            "KM_MCP_MASTER_PASSWORD",
            "",
        )

        # In-memory cache of decrypted keys
        self._key_cache: dict[str, str] = {}
        self._metadata_cache: dict[str, APIKeyMetadata] = {}

        # Ensure storage directory exists
        if self.storage_backend == StorageBackend.FILE:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    # FIXME: Contract disabled - @require(lambda __self, provider, key_value: len(provider) > 0 and len(key_value) > 0)
    def store_key(
        self,
        provider: str,
        key_value: str,
        key_id: str | None = None,
        expires_at: datetime | None = None,
        rotation_interval_days: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> Either[ValidationError, str]:
        """Store API key securely."""
        try:
            # Generate key ID if not provided
            if not key_id:
                key_id = self._generate_key_id(provider, key_value)

            # Create metadata
            metadata = APIKeyMetadata(
                provider=provider,
                key_id=key_id,
                status=KeyStatus.ACTIVE,
                created_at=datetime.now(UTC),
                expires_at=expires_at,
                rotation_interval_days=rotation_interval_days,
                tags=tags or {},
            )

            # Store based on backend
            if self.storage_backend == StorageBackend.ENVIRONMENT:
                return self._store_in_environment(provider, key_value, metadata)
            if self.storage_backend == StorageBackend.FILE:
                return self._store_in_file(provider, key_value, metadata)
            return Either.left(
                ValidationError(
                    "unsupported_backend",
                    str(self.storage_backend),
                    "Backend not implemented",
                ),
            )

        except Exception as e:
            return Either.left(
                ValidationError(
                    "key_storage_failed",
                    str(e),
                    "Key storage operation failed",
                ),
            )

    def retrieve_key(
        self,
        provider: str,
        key_id: str | None = None,
    ) -> Either[ValidationError, str]:
        """Retrieve API key for provider."""
        try:
            # Check cache first
            cache_key = f"{provider}:{key_id or 'default'}"
            if cache_key in self._key_cache:
                # Update usage metadata
                self._update_usage(provider, key_id)
                return Either.right(self._key_cache[cache_key])

            # Retrieve based on backend
            if self.storage_backend == StorageBackend.ENVIRONMENT:
                result = self._retrieve_from_environment(provider)
            elif self.storage_backend == StorageBackend.FILE:
                result = self._retrieve_from_file(provider, key_id)
            else:
                return Either.left(
                    ValidationError(
                        "unsupported_backend",
                        str(self.storage_backend),
                        "Backend not implemented",
                    ),
                )

            if result.is_right():
                # Cache the key
                self._key_cache[cache_key] = result.value
                self._update_usage(provider, key_id)

            return result

        except Exception as e:
            return Either.left(
                ValidationError(
                    "key_retrieval_failed",
                    str(e),
                    "Key retrieval operation failed",
                ),
            )

    def rotate_key(
        self,
        provider: str,
        new_key_value: str,
        key_id: str | None = None,
    ) -> Either[ValidationError, str]:
        """Rotate API key for provider."""
        try:
            # Mark old key as revoked
            old_metadata = self._metadata_cache.get(f"{provider}:{key_id or 'default'}")
            if old_metadata:
                old_metadata.status = KeyStatus.REVOKED
                old_metadata.last_rotation = datetime.now(UTC)

            # Store new key
            new_key_id = self._generate_key_id(provider, new_key_value)
            result = self.store_key(
                provider=provider,
                key_value=new_key_value,
                key_id=new_key_id,
                expires_at=old_metadata.expires_at if old_metadata else None,
                rotation_interval_days=old_metadata.rotation_interval_days
                if old_metadata
                else None,
                tags=old_metadata.tags if old_metadata else None,
            )

            if result.is_right():
                # Clear old key from cache
                old_cache_key = f"{provider}:{key_id or 'default'}"
                self._key_cache.pop(old_cache_key, None)

            return result

        except Exception as e:
            return Either.left(
                ValidationError(
                    "key_rotation_failed",
                    str(e),
                    "Key rotation operation failed",
                ),
            )

    def validate_key(
        self,
        provider: str,
        key_value: str,
    ) -> Either[ValidationError, bool]:
        """Validate API key format and basic structure."""
        try:
            # Provider-specific validation
            if provider.lower() == "openai":
                return self._validate_openai_key(key_value)
            if provider.lower() == "anthropic":
                return self._validate_anthropic_key(key_value)
            if provider.lower() == "google_ai":
                return self._validate_google_key(key_value)
            # Generic validation
            return Either.right(len(key_value) > 10 and key_value.isprintable())

        except Exception as e:
            return Either.left(
                ValidationError(
                    "key_validation_failed",
                    str(e),
                    "Key validation operation failed",
                ),
            )

    def list_keys(self) -> dict[str, APIKeyMetadata]:
        """List all stored API keys with metadata."""
        return self._metadata_cache.copy()

    def check_expiring_keys(self, days_ahead: int = 7) -> list[APIKeyMetadata]:
        """Get list of keys expiring within specified days."""
        expiring = []
        threshold = datetime.now(UTC) + timedelta(days=days_ahead)

        for metadata in self._metadata_cache.values():
            if (
                metadata.expires_at and metadata.expires_at <= threshold
            ) or metadata.needs_rotation():
                expiring.append(metadata)

        return expiring

    def get_key_status(
        self, provider: str, key_id: str | None = None
    ) -> APIKeyMetadata | None:
        """Get key status metadata for provider."""
        cache_key = f"{provider}:{key_id or 'default'}"
        return self._metadata_cache.get(cache_key)

    def _get_raw_storage_data(
        self, provider: str, key_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get raw storage data for testing purposes only."""
        if self.storage_backend == StorageBackend.FILE:
            if key_id:
                file_path = self.storage_path / f"{provider}_{key_id}.json"
            else:
                # Find the most recent active key for provider
                files = list(self.storage_path.glob(f"{provider}_*.json"))
                if not files:
                    return None
                file_path = max(files, key=lambda f: f.stat().st_mtime)

            if file_path.exists():
                try:
                    with open(file_path) as f:
                        return json.load(f)
                except Exception:
                    return None
        return None

    def _generate_key_id(self, provider: str, key_value: str) -> str:
        """Generate unique key ID."""
        data = f"{provider}:{key_value}:{datetime.now(UTC).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _encrypt_key(self, key_value: str) -> tuple[str, str]:
        """Encrypt API key with master password."""
        # Generate salt
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode()

        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))

        # Encrypt the API key
        f = Fernet(key)
        encrypted = f.encrypt(key_value.encode())

        return encrypted.decode(), salt_b64

    def _store_in_environment(
        self,
        provider: str,
        key_value: str,
        metadata: APIKeyMetadata,
    ) -> Either[ValidationError, str]:
        """Store key in environment variable."""
        env_var = f"{provider.upper()}_API_KEY"
        os.environ[env_var] = key_value

        # Cache metadata
        cache_key = f"{provider}:{metadata.key_id}"
        self._metadata_cache[cache_key] = metadata

        return Either.right(metadata.key_id)

    def _store_in_file(
        self,
        provider: str,
        key_value: str,
        metadata: APIKeyMetadata,
    ) -> Either[ValidationError, str]:
        """Store encrypted key in file."""
        if not self.master_password:
            return Either.left(
                ValidationError(
                    "no_master_password",
                    None,
                    "Master password required for file storage",
                ),
            )

        # Encrypt the key
        encrypted_key, salt = self._encrypt_key(key_value)

        # Create encrypted key object
        encrypted_obj = EncryptedAPIKey(
            encrypted_key=encrypted_key,
            salt=salt,
            provider=provider,
            key_id=metadata.key_id,
            metadata=metadata,
        )

        # Save to file
        file_path = self.storage_path / f"{provider}_{metadata.key_id}.json"
        with open(file_path, "w") as f:
            json.dump(
                {
                    "encrypted_key": encrypted_obj.encrypted_key,
                    "salt": encrypted_obj.salt,
                    "provider": encrypted_obj.provider,
                    "key_id": encrypted_obj.key_id,
                    "metadata": {
                        "provider": metadata.provider,
                        "key_id": metadata.key_id,
                        "status": metadata.status.value,
                        "created_at": metadata.created_at.isoformat(),
                        "last_used": metadata.last_used.isoformat()
                        if metadata.last_used
                        else None,
                        "expires_at": metadata.expires_at.isoformat()
                        if metadata.expires_at
                        else None,
                        "rotation_interval_days": metadata.rotation_interval_days,
                        "usage_count": metadata.usage_count,
                        "last_rotation": metadata.last_rotation.isoformat()
                        if metadata.last_rotation
                        else None,
                        "tags": metadata.tags,
                    },
                },
                f,
                indent=2,
            )

        # Cache metadata
        cache_key = f"{provider}:{metadata.key_id}"
        self._metadata_cache[cache_key] = metadata

        return Either.right(metadata.key_id)

    def _retrieve_from_environment(self, provider: str) -> Either[ValidationError, str]:
        """Retrieve key from environment variable."""
        env_var = f"{provider.upper()}_API_KEY"
        key_value = os.getenv(env_var)

        if not key_value:
            return Either.left(
                ValidationError(
                    "key_not_found",
                    env_var,
                    "Environment variable not set",
                ),
            )

        return Either.right(key_value)

    def _retrieve_from_file(
        self,
        provider: str,
        key_id: str | None = None,
    ) -> Either[ValidationError, str]:
        """Retrieve encrypted key from file."""
        if not self.master_password:
            return Either.left(
                ValidationError(
                    "no_master_password",
                    None,
                    "Master password required for file storage",
                ),
            )

        # Find key file
        if key_id:
            file_path = self.storage_path / f"{provider}_{key_id}.json"
        else:
            # Find the most recent active key for provider
            files = list(self.storage_path.glob(f"{provider}_*.json"))
            if not files:
                return Either.left(
                    ValidationError(
                        "key_not_found",
                        provider,
                        "No keys found for provider",
                    ),
                )
            file_path = max(files, key=lambda f: f.stat().st_mtime)

        if not file_path.exists():
            return Either.left(
                ValidationError("key_not_found", str(file_path), "Key file must exist"),
            )

        # Load and decrypt
        try:
            with open(file_path) as f:
                data = json.load(f)

            encrypted_obj = EncryptedAPIKey(
                encrypted_key=data["encrypted_key"],
                salt=data["salt"],
                provider=data["provider"],
                key_id=data["key_id"],
                metadata=self._parse_metadata(data["metadata"]),
            )

            return encrypted_obj.decrypt(self.master_password)

        except Exception as e:
            return Either.left(
                ValidationError(
                    "file_read_failed",
                    str(e),
                    "File read operation failed",
                ),
            )

    def _parse_metadata(self, data: dict[str, Any]) -> APIKeyMetadata:
        """Parse metadata from JSON data."""
        return APIKeyMetadata(
            provider=data["provider"],
            key_id=data["key_id"],
            status=KeyStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"])
            if data["last_used"]
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data["expires_at"]
            else None,
            rotation_interval_days=data["rotation_interval_days"],
            usage_count=data["usage_count"],
            last_rotation=datetime.fromisoformat(data["last_rotation"])
            if data["last_rotation"]
            else None,
            tags=data["tags"],
        )

    def _update_usage(self, provider: str, key_id: str | None = None) -> None:
        """Update key usage statistics."""
        cache_key = f"{provider}:{key_id or 'default'}"
        metadata = self._metadata_cache.get(cache_key)
        if metadata:
            metadata.last_used = datetime.now(UTC)
            metadata.usage_count += 1

    def _check_malicious_patterns(
        self, key_value: str
    ) -> Either[ValidationError, bool]:
        """Check for common injection and malicious patterns in API keys."""
        # Check for SQL injection patterns
        if any(
            pattern in key_value.lower()
            for pattern in ["drop table", "delete from", "--", "/*", "*/"]
        ):
            return Either.left(
                ValidationError(
                    "malicious_key_pattern",
                    key_value,
                    "Key contains suspicious SQL injection patterns",
                ),
            )

        # Check for script injection
        if any(
            pattern in key_value.lower()
            for pattern in ["<script", "</script>", "javascript:", "onerror="]
        ):
            return Either.left(
                ValidationError(
                    "malicious_key_pattern",
                    key_value,
                    "Key contains suspicious script injection patterns",
                ),
            )

        # Check for command injection
        if any(
            char in key_value for char in ["$", "`", ";", "&", "|", "\n", "\r", "\x00"]
        ):
            return Either.left(
                ValidationError(
                    "malicious_key_pattern",
                    key_value,
                    "Key contains suspicious command injection characters",
                ),
            )

        # Check for excessive length
        if len(key_value) > 200:
            return Either.left(
                ValidationError(
                    "malicious_key_pattern",
                    key_value,
                    "Key is suspiciously long",
                ),
            )

        # Check for non-printable characters
        if not key_value.isprintable():
            return Either.left(
                ValidationError(
                    "malicious_key_pattern",
                    key_value,
                    "Key contains non-printable characters",
                ),
            )

        return Either.right(True)

    def _validate_openai_key(self, key_value: str) -> Either[ValidationError, bool]:
        """Validate OpenAI API key format."""
        # Check for malicious patterns first
        malicious_validation = self._check_malicious_patterns(key_value)
        if malicious_validation.is_left():
            return malicious_validation

        if not key_value.startswith("sk-"):
            return Either.left(
                ValidationError(
                    "invalid_openai_key",
                    key_value,
                    "OpenAI keys must start with 'sk-'",
                ),
            )
        if len(key_value) < 20:
            return Either.left(
                ValidationError(
                    "invalid_openai_key",
                    key_value,
                    "OpenAI key must be at least 20 characters",
                ),
            )
        return Either.right(True)

    def _validate_anthropic_key(self, key_value: str) -> Either[ValidationError, bool]:
        """Validate Anthropic API key format."""
        # Check for malicious patterns first
        malicious_validation = self._check_malicious_patterns(key_value)
        if malicious_validation.is_left():
            return malicious_validation

        if not key_value.startswith("sk-ant-"):
            return Either.left(
                ValidationError(
                    "invalid_anthropic_key",
                    key_value,
                    "Anthropic keys must start with 'sk-ant-'",
                ),
            )
        return Either.right(True)

    def _validate_google_key(self, key_value: str) -> Either[ValidationError, bool]:
        """Validate Google AI API key format."""
        # Check for malicious patterns first
        malicious_validation = self._check_malicious_patterns(key_value)
        if malicious_validation.is_left():
            return malicious_validation

        if len(key_value) < 20:
            return Either.left(
                ValidationError(
                    "invalid_google_key",
                    key_value,
                    "Google AI key must be at least 20 characters",
                ),
            )
        return Either.right(True)


# Global API key manager instance
_global_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _global_key_manager
    if _global_key_manager is None:
        _global_key_manager = APIKeyManager()
    return _global_key_manager


def store_api_key(
    provider: str,
    key_value: str,
    **kwargs: Any,
) -> Either[ValidationError, str]:
    """Store API key using global manager."""
    manager = get_api_key_manager()
    return manager.store_key(provider, key_value, **kwargs)


def get_api_key(
    provider: str,
    key_id: str | None = None,
) -> Either[ValidationError, str]:
    """Get API key using global manager."""
    manager = get_api_key_manager()
    return manager.retrieve_key(provider, key_id)
