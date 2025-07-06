"""
AI Provider Clients Package.

This package provides comprehensive AI provider integration with support for
multiple AI services, automatic fallback, health monitoring, and enterprise-grade
configuration management.

Available Providers:
- OpenAI (GPT-3.5, GPT-4, Embeddings)
- Google AI (Placeholder - Future Implementation)
- Anthropic (Placeholder - Future Implementation)
- Azure OpenAI (Placeholder - Future Implementation)

Usage:
    from src.ai.providers import get_provider_factory, initialize_providers_from_env

    # Initialize from environment
    providers = initialize_providers_from_env()

    # Get provider registry for request processing
    registry = get_provider_registry()
    result = await registry.process_with_fallback(request)
"""

from .base_client import (
    AuthenticationType,
    BaseProviderClient,
    ProviderCapabilities,
    ProviderHealth,
    ProviderRegistry,
    ProviderStatus,
    RateLimitInfo,
)
from .openai_client import OpenAIClient, create_openai_client
from .provider_factory import (
    SAMPLE_CONFIG,
    ProviderConfig,
    ProviderFactory,
    ProviderType,
    get_provider_client,
    get_provider_factory,
    get_provider_registry,
    initialize_providers_from_env,
)

__all__ = [
    # Base classes
    "BaseProviderClient",
    "ProviderStatus",
    "AuthenticationType",
    "ProviderCapabilities",
    "ProviderHealth",
    "RateLimitInfo",
    "ProviderRegistry",
    # OpenAI implementation
    "OpenAIClient",
    "create_openai_client",
    # Factory and configuration
    "ProviderFactory",
    "ProviderType",
    "ProviderConfig",
    "get_provider_factory",
    "initialize_providers_from_env",
    "get_provider_client",
    "get_provider_registry",
    "SAMPLE_CONFIG",
]
