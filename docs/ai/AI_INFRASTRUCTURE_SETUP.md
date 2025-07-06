# AI Infrastructure Setup & Configuration Guide

## Overview

This guide provides comprehensive setup and configuration instructions for the Keyboard Maestro MCP AI infrastructure, including provider configuration, cache optimization, cost management, and security setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Provider Configuration](#provider-configuration)
3. [Cache System Configuration](#cache-system-configuration)
4. [Cost Optimization Setup](#cost-optimization-setup)
5. [Security Configuration](#security-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Migration from Mock Implementation](#migration-from-mock-implementation)

## Prerequisites

### System Requirements

- Python 3.9 or higher
- Virtual environment (recommended: uv)
- Minimum 4GB RAM for optimal cache performance
- SSD storage recommended for L3 cache persistence

### Required Dependencies

```bash
# Install core dependencies
uv add openai anthropic google-generativeai
uv add cryptography pyyaml
uv add tiktoken  # For OpenAI token counting
uv add hypothesis pytest  # For testing
```

### Environment Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies
uv sync
```

## Provider Configuration

### OpenAI Configuration

#### 1. API Key Setup

```bash
# Set environment variable (recommended)
export OPENAI_API_KEY="sk-your-api-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=sk-your-api-key-here" >> .env
```

#### 2. Configuration File

Create `config/ai_config.yaml`:

```yaml
config_version: "1.0"
environment: "production"
default_provider: "openai"
default_model: "gpt-3.5-turbo"

providers:
  openai:
    provider_name: "openai"
    enabled: true
    api_key_env_var: "OPENAI_API_KEY"
    base_url: "https://api.openai.com/v1"
    timeout_seconds: 30.0
    max_retries: 3
    rate_limit_rpm: 3500  # Requests per minute
    rate_limit_tpm: 90000  # Tokens per minute
    priority: 1
    models:
      gpt-3.5-turbo:
        model_name: "gpt-3.5-turbo"
        provider: "openai"
        enabled: true
        max_tokens: 4096
        context_window: 16384
        temperature: 0.7
        cost_per_input_token: 0.001
        cost_per_output_token: 0.002
      gpt-4:
        model_name: "gpt-4"
        provider: "openai"
        enabled: true
        max_tokens: 8192
        context_window: 8192
        temperature: 0.7
        cost_per_input_token: 0.03
        cost_per_output_token: 0.06
```

### Anthropic Configuration (Future Implementation)

```yaml
providers:
  anthropic:
    provider_name: "anthropic"
    enabled: true
    api_key_env_var: "ANTHROPIC_API_KEY"
    base_url: "https://api.anthropic.com"
    timeout_seconds: 30.0
    max_retries: 3
    rate_limit_rpm: 1000
    priority: 2
    models:
      claude-3-haiku:
        model_name: "claude-3-haiku-20240307"
        provider: "anthropic"
        enabled: true
        max_tokens: 4096
        context_window: 200000
        cost_per_input_token: 0.00025
        cost_per_output_token: 0.00125
```

### Google AI Configuration (Future Implementation)

```yaml
providers:
  google_ai:
    provider_name: "google_ai"
    enabled: true
    api_key_env_var: "GOOGLE_AI_API_KEY"
    base_url: "https://generativelanguage.googleapis.com"
    timeout_seconds: 30.0
    max_retries: 3
    rate_limit_rpm: 1500
    priority: 3
    models:
      gemini-pro:
        model_name: "gemini-pro"
        provider: "google_ai"
        enabled: true
        max_tokens: 8192
        context_window: 32768
        cost_per_input_token: 0.0005
        cost_per_output_token: 0.0015
```

## Cache System Configuration

### Multi-Level Cache Setup

Add to `config/ai_config.yaml`:

```yaml
cache:
  enabled: true
  default_ttl_hours: 6
  max_cache_size_mb: 100
  compression_enabled: true
  
  # L1 Cache (Memory - Fastest)
  l1_max_entries: 500
  l1_eviction_policy: "intelligent"
  
  # L2 Cache (Compressed Memory)
  l2_max_entries: 2000
  l2_compression_level: 6  # zlib compression level
  
  # L3 Cache (Persistent Disk)
  l3_enabled: true
  l3_directory: "./cache/l3"
  l3_max_size_gb: 1.0
  
  # Predictive Features
  prefetch_enabled: true
  prefetch_threshold: 3  # Access count to trigger prefetch
  
  # Namespace Configuration
  namespace_isolation: true
  default_namespace: "ai_operations"
```

### Cache Directory Setup

```bash
# Create cache directories
mkdir -p cache/l3
chmod 755 cache/l3

# For production, consider dedicated cache volume
# mkdir -p /opt/km-mcp/cache/l3
```

### Cache Performance Tuning

```yaml
cache:
  # High-performance configuration
  l1_max_entries: 1000      # More memory cache
  l2_max_entries: 5000      # Larger compressed cache
  compression_enabled: true  # Enable compression for L2/L3
  prefetch_enabled: true    # Enable predictive prefetching
  
  # Memory-constrained configuration
  l1_max_entries: 200       # Smaller memory footprint
  l2_max_entries: 800       # Reduced L2 cache
  l3_enabled: false         # Disable disk cache
  compression_enabled: false # Disable compression for speed
```

## Cost Optimization Setup

### Budget Configuration

Add to `config/ai_config.yaml`:

```yaml
cost:
  enabled: true
  track_usage: true
  budget_enforcement: true
  cost_reporting_enabled: true
  
  # Budget Settings
  default_budget_monthly: 1000.00  # USD
  alert_thresholds: [0.5, 0.8, 0.95]  # 50%, 80%, 95%
  
  # Optimization Strategy
  cost_optimization_strategy: "balanced"  # aggressive|balanced|conservative
  auto_optimization: false  # Manual approval required
  
  # Model Selection
  efficiency_tracking: true
  model_recommendation_enabled: true
```

### Budget Management

```python
# Example: Create monthly budget
from src.ai.cost_optimization import CostOptimizer, CostBudget, BudgetPeriod
from decimal import Decimal
from datetime import datetime, UTC

cost_optimizer = CostOptimizer()

# Create production budget
production_budget = CostBudget(
    budget_id="prod_monthly_2024",
    name="Production Monthly Budget",
    amount=Decimal("2500.00"),
    period=BudgetPeriod.MONTHLY,
    start_date=datetime.now(UTC),
    alert_thresholds=[0.5, 0.75, 0.9, 0.95]
)

result = cost_optimizer.add_budget(production_budget)
if result.is_right():
    print(f"Budget created: {result.right_value}")
```

### Cost Monitoring Setup

```bash
# Environment variables for cost tracking
export AI_COST_TRACKING=true
export AI_BUDGET_ALERTS=true
export AI_COST_REPORT_EMAIL="admin@yourcompany.com"
```

## Security Configuration

### API Key Security

#### Encryption at Rest

```yaml
security:
  api_key_encryption: true
  encryption_algorithm: "AES-256"
  key_derivation: "PBKDF2"
  key_derivation_iterations: 100000
  
  # Master password for key encryption (set via environment)
  master_password_env_var: "AI_MASTER_PASSWORD"
```

#### Environment Variables

```bash
# Master password for API key encryption
export AI_MASTER_PASSWORD="your-secure-master-password"

# Enable security features
export AI_AUDIT_LOGGING=true
export AI_REQUEST_LOGGING=true
export AI_RESPONSE_LOGGING=false  # May contain sensitive data
```

### Audit Configuration

```yaml
security:
  audit_enabled: true
  audit_log_path: "./logs/ai_audit.log"
  audit_rotation_size: "100MB"
  audit_retention_days: 90
  
  request_logging: true
  response_logging: false  # Disable for sensitive data
  data_anonymization: true
  
  # Access Control
  max_request_size_mb: 10
  rate_limiting_enabled: true
  allowed_domains: []  # Empty = allow all
  blocked_domains: []  # Blocked domains list
```

### SSL/TLS Configuration

```yaml
security:
  # HTTPS Configuration
  ssl_verify: true
  ssl_cert_path: ""  # Leave empty for default CA bundle
  ssl_key_path: ""
  
  # Custom CA certificates
  ca_bundle_path: ""  # Path to custom CA bundle if needed
```

## Performance Tuning

### Cache Performance

```python
# Example: Performance monitoring
from src.ai.caching_system import IntelligentCacheManager

cache_manager = IntelligentCacheManager()

# Get performance report
report = cache_manager.get_cache_efficiency_report()
print(f"Cache hit ratio: {report['cache_hit_ratio']}")
print(f"Average response time: {report['avg_response_time']}ms")

# Optimize cache settings based on usage patterns
optimization_report = cache_manager.optimize_cache()
print(f"Optimization recommendations: {optimization_report}")
```

### Provider Performance

```yaml
providers:
  openai:
    # Performance tuning
    timeout_seconds: 15.0     # Faster timeout for responsiveness
    max_retries: 2           # Fewer retries for speed
    rate_limit_rpm: 3500     # Match your API tier
    
    # Connection pooling
    connection_pool_size: 10
    keep_alive: true
    
    # Request optimization
    request_compression: true
    response_compression: true
```

### System Performance

```bash
# Environment variables for performance tuning
export AI_CACHE_MAX_MEMORY=128  # MB
export AI_WORKER_THREADS=4      # Concurrent processing
export AI_BATCH_SIZE=10         # Batch processing size
export AI_PREFETCH_ENABLED=true # Enable cache prefetching
```

## Monitoring & Troubleshooting

### Health Monitoring

```python
# Example: Provider health monitoring
from src.ai.providers.provider_factory import ProviderFactory

factory = ProviderFactory()
status = factory.get_provider_status()

for provider_name, provider_status in status.items():
    print(f"{provider_name}: {provider_status['status']}")
    if not provider_status['enabled']:
        print(f"  Issue: {provider_status.get('error', 'Unknown')}")
```

### Performance Monitoring

```python
# Example: Performance metrics
from src.ai.cost_optimization import CostOptimizer

cost_optimizer = CostOptimizer()
breakdown = cost_optimizer.get_cost_breakdown(period_days=7)

print(f"Weekly usage: {breakdown['total_requests']} requests")
print(f"Total cost: ${breakdown['total_cost']}")
print(f"Average cost per request: ${breakdown['avg_cost_per_request']}")
```

### Log Configuration

```bash
# Create log directories
mkdir -p logs/ai
chmod 755 logs/ai

# Log rotation setup (using logrotate)
cat > /etc/logrotate.d/km-mcp-ai << EOF
/path/to/km-mcp/logs/ai/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 app app
}
EOF
```

### Troubleshooting Common Issues

#### API Key Issues

```bash
# Test API key validity
python -c "
from src.ai.security.api_key_manager import APIKeyManager
manager = APIKeyManager()
result = manager.validate_key('openai', 'your-api-key')
print('Valid' if result.is_right() else f'Invalid: {result.left_value}')
"
```

#### Cache Issues

```bash
# Clear cache if corrupted
rm -rf cache/l3/*

# Test cache functionality
python -c "
from src.ai.caching_system import CacheManager
cache = CacheManager(max_size=100)
cache.put('test', 'value')
result = cache.get('test')
print('Cache working' if result == 'value' else 'Cache issue')
"
```

#### Performance Issues

```bash
# Check system resources
free -h  # Memory usage
df -h    # Disk usage
top -p $(pgrep -f "python")  # Process monitoring

# Monitor cache performance
tail -f logs/ai/cache_performance.log
```

## Migration from Mock Implementation

### Step 1: Backup Current Configuration

```bash
# Backup existing configuration
cp -r config config.backup.$(date +%Y%m%d)
```

### Step 2: Update Configuration

```bash
# Replace mock configuration with real implementation config
cp config/ai_config.yaml.example config/ai_config.yaml
# Edit config/ai_config.yaml with your settings
```

### Step 3: Environment Setup

```bash
# Set required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export AI_MASTER_PASSWORD="your-secure-master-password"
export AI_COST_TRACKING=true
```

### Step 4: Initialize Real Implementation

```python
# Migration script example
from src.ai.model_manager import AIModelManager
from src.ai.config.ai_config import load_ai_config

# Load configuration
config_result = load_ai_config()
if config_result.is_left():
    print(f"Configuration error: {config_result.left_value}")
    exit(1)

# Initialize AI model manager with real implementations
model_manager = AIModelManager()
initialization_result = model_manager.initialize()

if initialization_result.is_right():
    print("AI infrastructure successfully initialized!")
    
    # Test functionality
    test_result = model_manager.process_ai_request({
        "operation": "analyze",
        "input": "Test input",
        "parameters": {"temperature": 0.7}
    })
    
    if test_result.is_right():
        print("Test successful - real implementation active")
    else:
        print(f"Test failed: {test_result.left_value}")
else:
    print(f"Initialization failed: {initialization_result.left_value}")
```

### Step 5: Verification

```bash
# Run test suite to verify migration
uv run pytest tests/test_ai/ -v

# Check system health
python -c "
from src.ai.providers.provider_factory import ProviderFactory
factory = ProviderFactory()
status = factory.get_provider_status()
print('All providers healthy:' if all(s['enabled'] for s in status.values()) else 'Issues detected')
"
```

### Step 6: Monitor Initial Operation

```bash
# Monitor logs for first hour
tail -f logs/ai/*.log

# Check performance metrics
python scripts/check_ai_performance.py

# Verify cost tracking
python scripts/check_cost_tracking.py
```

## Production Deployment Checklist

### Security Checklist

- [ ] API keys stored securely (encrypted at rest)
- [ ] Master password set and secured
- [ ] Audit logging enabled and configured
- [ ] Rate limiting configured appropriately
- [ ] SSL/TLS verification enabled
- [ ] Log rotation configured
- [ ] Access controls implemented

### Performance Checklist

- [ ] Cache directories created with appropriate permissions
- [ ] Cache size limits configured for available memory
- [ ] Provider timeouts and retries optimized
- [ ] Performance monitoring enabled
- [ ] Resource limits configured

### Monitoring Checklist

- [ ] Health checks implemented
- [ ] Cost monitoring and alerts configured
- [ ] Log aggregation setup
- [ ] Performance metrics collection enabled
- [ ] Backup procedures for cache and logs

### Documentation Checklist

- [ ] Configuration documented for team
- [ ] Runbook created for operations team
- [ ] Troubleshooting guide available
- [ ] Contact information for escalation
- [ ] Change management procedures documented

## Support and Resources

### Documentation

- [AI Model Manager API Reference](./AI_MODEL_MANAGER_API.md)
- [Cache System Documentation](./CACHE_SYSTEM.md)
- [Cost Optimization Guide](./COST_OPTIMIZATION.md)
- [Security Implementation Details](./SECURITY.md)

### Monitoring Scripts

- `scripts/health_check.py` - Provider health monitoring
- `scripts/performance_report.py` - Performance metrics
- `scripts/cost_analysis.py` - Cost analysis and reporting
- `scripts/cache_analysis.py` - Cache performance analysis

### Contact Information

- **Technical Support**: tech-support@yourcompany.com
- **Security Issues**: security@yourcompany.com
- **Performance Issues**: performance@yourcompany.com

---

*Last Updated: 2025-07-06*
*Version: 1.0*