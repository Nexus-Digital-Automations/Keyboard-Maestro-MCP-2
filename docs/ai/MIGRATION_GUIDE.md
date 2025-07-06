# Migration Guide: Mock to Real AI Implementation

## Overview

This guide provides step-by-step instructions for migrating from the mock AI implementation to the real AI infrastructure with live provider integrations, intelligent caching, and cost optimization.

## Migration Timeline

**Estimated Duration**: 2-4 hours
**Downtime Required**: 15-30 minutes
**Complexity**: Medium

## Pre-Migration Checklist

### Prerequisites

- [ ] Python 3.9+ environment ready
- [ ] Valid API keys for AI providers
- [ ] Backup of current configuration
- [ ] Test environment for validation
- [ ] Team notification of maintenance window

### Required Access

- [ ] OpenAI API key with appropriate billing setup
- [ ] System administrator access
- [ ] Database/cache directory write permissions
- [ ] Log directory access

### Dependencies

```bash
# Verify required packages are installed
uv list | grep -E "(openai|cryptography|pyyaml|tiktoken)"

# If missing, install:
uv add openai cryptography pyyaml tiktoken
```

## Step-by-Step Migration

### Phase 1: Environment Preparation

#### 1.1 Create Backup

```bash
# Create migration backup
mkdir -p migration_backup/$(date +%Y%m%d_%H%M)
BACKUP_DIR="migration_backup/$(date +%Y%m%d_%H%M)"

# Backup configuration
cp -r config/ $BACKUP_DIR/config_backup/
cp -r src/server/tools/ai_model_management.py $BACKUP_DIR/
cp -r src/ai/ $BACKUP_DIR/ai_backup/

echo "Backup created in $BACKUP_DIR"
```

#### 1.2 Prepare Environment Variables

```bash
# Create .env file for API keys
cat > .env << EOF
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# AI Infrastructure Configuration
AI_MASTER_PASSWORD=your-secure-master-password
AI_COST_TRACKING=true
AI_AUDIT_LOGGING=true
AI_DEBUG_MODE=false

# Cache Configuration
AI_CACHE_ENABLED=true
AI_CACHE_MAX_MEMORY=128

# Security Configuration
AI_REQUEST_LOGGING=true
AI_RESPONSE_LOGGING=false
EOF

# Load environment variables
source .env
```

#### 1.3 Create Directory Structure

```bash
# Create required directories
mkdir -p config/ai
mkdir -p cache/l3
mkdir -p logs/ai
mkdir -p docs/ai

# Set appropriate permissions
chmod 755 cache/l3
chmod 755 logs/ai
chmod 644 .env
```

### Phase 2: Configuration Setup

#### 2.1 Create AI Configuration

```bash
# Create main AI configuration file
cat > config/ai_config.yaml << 'EOF'
config_version: "1.0"
environment: "production"
default_provider: "openai"
default_model: "gpt-3.5-turbo"
debug_mode: false

providers:
  openai:
    provider_name: "openai"
    enabled: true
    api_key_env_var: "OPENAI_API_KEY"
    base_url: "https://api.openai.com/v1"
    timeout_seconds: 30.0
    max_retries: 3
    rate_limit_rpm: 3500
    rate_limit_tpm: 90000
    priority: 1
    health_check_interval: 300
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

cache:
  enabled: true
  default_ttl_hours: 6
  max_cache_size_mb: 100
  compression_enabled: true
  l1_max_entries: 500
  l2_max_entries: 2000
  l3_enabled: true
  l3_directory: "./cache/l3"
  prefetch_enabled: true
  eviction_policy: "intelligent"
  namespace_isolation: true

cost:
  enabled: true
  default_budget_monthly: 1000.00
  alert_thresholds: [0.5, 0.8, 0.95]
  auto_optimization: false
  track_usage: true
  cost_optimization_strategy: "balanced"
  budget_enforcement: true
  cost_reporting_enabled: true

security:
  api_key_encryption: true
  request_logging: true
  response_logging: false
  audit_enabled: true
  data_anonymization: true
  max_request_size_mb: 10
  allowed_domains: []
  blocked_domains: []
EOF
```

#### 2.2 Validate Configuration

```bash
# Test configuration loading
python -c "
from src.ai.config.ai_config import load_ai_config
result = load_ai_config()
if result.is_right():
    print('✅ Configuration valid')
    config = result.right_value
    print(f'Default provider: {config.default_provider}')
    print(f'Providers configured: {list(config.providers.keys())}')
else:
    print(f'❌ Configuration error: {result.left_value}')
    exit(1)
"
```

### Phase 3: Code Migration

#### 3.1 Update AI Model Management Tools

```bash
# Create patch file for ai_model_management.py
cat > migration_patch.py << 'EOF'
#!/usr/bin/env python3
"""
Migration script to replace mock implementations with real implementations.
"""

import re
import shutil
from pathlib import Path

def migrate_ai_model_management():
    """Migrate ai_model_management.py to use real implementations."""
    
    file_path = Path("src/server/tools/ai_model_management.py")
    
    # Read current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace mock imports with real imports
    content = re.sub(
        r'from src\.ai\.model_manager import MockIntelligentCacheManager, MockCostOptimizer',
        'from src.ai.caching_system import IntelligentCacheManager\nfrom src.ai.cost_optimization import CostOptimizer',
        content
    )
    
    # Replace mock instantiations with real implementations
    content = re.sub(
        r'self\.cache_manager = MockIntelligentCacheManager\(\)',
        'self.cache_manager = IntelligentCacheManager()',
        content
    )
    
    content = re.sub(
        r'self\.cost_optimizer = MockCostOptimizer\(\)',
        'self.cost_optimizer = CostOptimizer()',
        content
    )
    
    # Update imports to include real implementations
    if 'from src.ai.caching_system import IntelligentCacheManager' not in content:
        import_section = content.find('from src.core.ai_integration import')
        if import_section != -1:
            insert_point = content.find('\n', import_section) + 1
            content = (content[:insert_point] + 
                      'from src.ai.caching_system import IntelligentCacheManager\n' +
                      'from src.ai.cost_optimization import CostOptimizer\n' +
                      content[insert_point:])
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")

if __name__ == "__main__":
    migrate_ai_model_management()
    print("Migration patch applied successfully!")
EOF

# Run migration patch
python migration_patch.py
```

#### 3.2 Update AI Model Manager

```bash
# Update model manager initialization
python -c "
import sys
sys.path.append('src')

from ai.model_manager import AIModelManager
from ai.providers.provider_factory import ProviderFactory

# Test initialization
try:
    model_manager = AIModelManager()
    print('✅ AIModelManager created successfully')
    
    # Test provider factory
    factory = ProviderFactory()
    print('✅ ProviderFactory created successfully')
    
    # Test provider initialization
    providers = factory.initialize_from_environment()
    print(f'✅ Providers initialized: {list(providers.keys())}')
    
except Exception as e:
    print(f'❌ Initialization error: {e}')
    sys.exit(1)
"
```

### Phase 4: Testing and Validation

#### 4.1 Run Integration Tests

```bash
# Run AI infrastructure tests
echo "Running integration tests..."
uv run pytest tests/test_ai/test_real_integrations.py -v

# Run property-based tests
echo "Running property-based tests..."
uv run pytest tests/test_ai/test_property_based.py -v

# Run performance tests
echo "Running performance benchmarks..."
uv run pytest tests/test_ai/test_performance_benchmarks.py -v

# Run security validation
echo "Running security tests..."
uv run pytest tests/test_ai/test_security_validation.py -v
```

#### 4.2 Functional Validation

```bash
# Test end-to-end functionality
python -c "
import asyncio
from src.ai.caching_system import IntelligentCacheManager
from src.ai.cost_optimization import CostOptimizer
from src.ai.providers.openai_client import OpenAIClient
from src.core.ai_integration import AIRequest, AIOperation

async def test_migration():
    print('Testing real implementations...')
    
    # Test cache manager
    cache_manager = IntelligentCacheManager()
    print('✅ Cache manager initialized')
    
    # Test cost optimizer
    cost_optimizer = CostOptimizer()
    cost_optimizer.record_usage(
        operation=AIOperation.ANALYZE,
        model_used='gpt-3.5-turbo',
        input_tokens=10,
        output_tokens=5,
        cost=0.0001,
        processing_time=1.0
    )
    breakdown = cost_optimizer.get_cost_breakdown()
    print(f'✅ Cost tracking active: {breakdown[\"total_requests\"]} requests logged')
    
    # Test OpenAI client (without actual API call)
    client = OpenAIClient(
        api_key='test-key',
        model='gpt-3.5-turbo'
    )
    print('✅ OpenAI client initialized')
    
    print('All real implementations working correctly!')

asyncio.run(test_migration())
"
```

#### 4.3 Cache System Validation

```bash
# Test cache functionality
python -c "
import asyncio
from src.ai.caching_system import IntelligentCacheManager, CacheKey
from src.core.ai_integration import AIOperation

async def test_cache():
    cache_manager = IntelligentCacheManager()
    
    # Test cache operations
    test_key = 'migration_test'
    test_value = {'result': 'test successful', 'timestamp': '2025-07-06'}
    
    # Put operation
    await cache_manager.put_ai_result(
        AIOperation.ANALYZE,
        test_key,
        test_value,
        {'temperature': 0.7}
    )
    print('✅ Cache PUT operation successful')
    
    # Get operation
    result = await cache_manager.get_ai_result(
        AIOperation.ANALYZE,
        test_key,
        {'temperature': 0.7}
    )
    
    if result == test_value:
        print('✅ Cache GET operation successful')
    else:
        print('❌ Cache GET operation failed')
        exit(1)
    
    # Get cache statistics
    report = cache_manager.get_cache_efficiency_report()
    print(f'✅ Cache efficiency report: {report[\"cache_efficiency_score\"]}')

asyncio.run(test_cache())
"
```

### Phase 5: Production Deployment

#### 5.1 Service Restart (Downtime Window)

```bash
# Stop existing services
echo "Stopping services for migration..."
# systemctl stop km-mcp-server  # If running as service

# Clear any existing mock cache
rm -rf cache/mock_*

# Restart services with new implementation
echo "Starting services with real AI implementation..."
# systemctl start km-mcp-server  # If running as service

echo "Migration deployment complete!"
```

#### 5.2 Health Check

```bash
# Comprehensive health check
python -c "
from src.ai.providers.provider_factory import ProviderFactory
from src.ai.security.api_key_manager import APIKeyManager
from src.ai.config.ai_config import get_ai_config_manager

print('=== AI Infrastructure Health Check ===')

# Check configuration
config_manager = get_ai_config_manager()
config_result = config_manager.load_config()
if config_result.is_right():
    print('✅ Configuration loaded successfully')
else:
    print(f'❌ Configuration error: {config_result.left_value}')

# Check providers
factory = ProviderFactory()
status = factory.get_provider_status()
for provider, provider_status in status.items():
    if provider_status['enabled']:
        print(f'✅ Provider {provider}: {provider_status[\"status\"]}')
    else:
        print(f'❌ Provider {provider}: disabled')

# Check API key manager
api_manager = APIKeyManager()
print('✅ API key manager initialized')

print('=== Health Check Complete ===')
"
```

#### 5.3 Performance Monitoring

```bash
# Monitor initial performance
python -c "
import time
import asyncio
from src.ai.caching_system import IntelligentCacheManager
from src.ai.cost_optimization import CostOptimizer

async def monitor_performance():
    print('=== Performance Monitoring ===')
    
    cache_manager = IntelligentCacheManager()
    cost_optimizer = CostOptimizer()
    
    # Monitor cache performance
    start_time = time.time()
    
    # Simulate some operations
    for i in range(10):
        await cache_manager.put_ai_result(
            'analyze',
            f'test_{i}',
            {'result': f'test_result_{i}'},
            {'temperature': 0.7}
        )
    
    cache_time = time.time() - start_time
    print(f'✅ Cache operations: {cache_time:.3f}s for 10 operations')
    
    # Monitor cost tracking
    start_time = time.time()
    
    for i in range(10):
        cost_optimizer.record_usage(
            operation='analyze',
            model_used='gpt-3.5-turbo',
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            processing_time=1.0
        )
    
    cost_time = time.time() - start_time
    print(f'✅ Cost tracking: {cost_time:.3f}s for 10 records')
    
    # Get efficiency report
    report = cache_manager.get_cache_efficiency_report()
    print(f'✅ Cache efficiency: {report.get(\"cache_efficiency_score\", \"N/A\")}')
    
    print('=== Performance Monitoring Complete ===')

asyncio.run(monitor_performance())
"
```

### Phase 6: Post-Migration Validation

#### 6.1 Integration Validation

```bash
# Test actual AI model management tools
python -c "
from src.server.tools.ai_model_management import (
    km_ai_analyze_content,
    km_ai_generate_text,
    km_ai_classify_content
)

# Test analyze function
try:
    result = km_ai_analyze_content({
        'content': 'This is a test message for migration validation',
        'analysis_type': 'sentiment',
        'model': 'gpt-3.5-turbo'
    })
    print('✅ AI analyze function working')
except Exception as e:
    print(f'❌ AI analyze error: {e}')

# Test generate function
try:
    result = km_ai_generate_text({
        'prompt': 'Generate a brief test message',
        'max_tokens': 50,
        'model': 'gpt-3.5-turbo'
    })
    print('✅ AI generate function working')
except Exception as e:
    print(f'❌ AI generate error: {e}')

print('Integration validation complete!')
"
```

#### 6.2 Monitoring Setup

```bash
# Setup ongoing monitoring
cat > scripts/migration_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Post-migration monitoring script.
"""

import asyncio
import time
from datetime import datetime, UTC
from src.ai.providers.provider_factory import ProviderFactory
from src.ai.cost_optimization import CostOptimizer
from src.ai.caching_system import IntelligentCacheManager

async def monitor_migration():
    """Monitor migration success metrics."""
    
    print(f"=== Migration Monitor - {datetime.now(UTC)} ===")
    
    # Provider health
    factory = ProviderFactory()
    status = factory.get_provider_status()
    
    healthy_providers = sum(1 for s in status.values() if s['enabled'])
    total_providers = len(status)
    print(f"Providers: {healthy_providers}/{total_providers} healthy")
    
    # Cost tracking
    cost_optimizer = CostOptimizer()
    breakdown = cost_optimizer.get_cost_breakdown(period_days=1)
    print(f"Today's usage: {breakdown['total_requests']} requests, ${breakdown['total_cost']:.4f}")
    
    # Cache performance
    cache_manager = IntelligentCacheManager()
    report = cache_manager.get_cache_efficiency_report()
    print(f"Cache efficiency: {report.get('cache_efficiency_score', 'N/A')}")
    
    print("=== Monitor Complete ===\n")

if __name__ == "__main__":
    asyncio.run(monitor_migration())
EOF

chmod +x scripts/migration_monitor.py

# Run initial monitoring
python scripts/migration_monitor.py
```

## Migration Verification Checklist

### Functional Verification

- [ ] AI model management tools using real implementations
- [ ] Cache system operational (L1, L2, L3 if enabled)
- [ ] Cost tracking recording usage accurately
- [ ] API key management working securely
- [ ] Provider health monitoring active
- [ ] Configuration loading correctly

### Performance Verification

- [ ] Cache hit rates >50% for repeated operations
- [ ] Response times <500ms for cached requests
- [ ] Cost calculations completing <5ms
- [ ] Memory usage within expected limits
- [ ] No performance degradation from mock implementation

### Security Verification

- [ ] API keys encrypted at rest
- [ ] Audit logging active
- [ ] Request validation working
- [ ] Rate limiting functional
- [ ] No sensitive data in logs

### Monitoring Verification

- [ ] Provider health checks running
- [ ] Cost alerts configured and tested
- [ ] Performance metrics collecting
- [ ] Log rotation working
- [ ] Backup procedures tested

## Rollback Procedure

If issues are encountered, use this rollback procedure:

### Quick Rollback

```bash
# Stop services
# systemctl stop km-mcp-server

# Restore from backup
BACKUP_DIR=$(ls -t migration_backup/ | head -1)
cp migration_backup/$BACKUP_DIR/ai_model_management.py src/server/tools/
cp -r migration_backup/$BACKUP_DIR/config_backup/* config/

# Restart services
# systemctl start km-mcp-server

echo "Rollback complete - using mock implementation"
```

### Validate Rollback

```bash
# Test mock functionality
python -c "
from src.server.tools.ai_model_management import km_ai_analyze_content

result = km_ai_analyze_content({
    'content': 'test',
    'analysis_type': 'sentiment'
})

if 'mock' in str(result).lower():
    print('✅ Rollback successful - mock implementation active')
else:
    print('❌ Rollback issue - check configuration')
"
```

## Troubleshooting

### Common Issues

#### API Key Issues

```bash
# Test API key
python -c "
import os
from src.ai.security.api_key_manager import APIKeyManager

key = os.getenv('OPENAI_API_KEY')
if not key:
    print('❌ OPENAI_API_KEY not set')
    exit(1)

manager = APIKeyManager()
result = manager.validate_key('openai', key)
if result.is_right():
    print('✅ API key valid')
else:
    print(f'❌ API key invalid: {result.left_value}')
"
```

#### Cache Issues

```bash
# Reset cache
rm -rf cache/l3/*
mkdir -p cache/l3
chmod 755 cache/l3

# Test cache
python -c "
from src.ai.caching_system import CacheManager
cache = CacheManager(max_size=10)
cache.put('test', 'value')
result = cache.get('test')
print('✅ Cache working' if result == 'value' else '❌ Cache issue')
"
```

#### Configuration Issues

```bash
# Validate configuration
python -c "
from src.ai.config.ai_config import load_ai_config
result = load_ai_config()
if result.is_left():
    print(f'❌ Config error: {result.left_value}')
    exit(1)
print('✅ Configuration valid')
"
```

### Support Contacts

- **Technical Issues**: Create issue in repository
- **API Key Problems**: Check with provider (OpenAI, etc.)
- **Performance Issues**: Review resource allocation
- **Security Concerns**: Contact security team

## Post-Migration Optimization

### Week 1: Monitor and Tune

- Monitor cache hit rates and adjust cache sizes
- Review cost patterns and adjust budgets
- Optimize provider timeouts based on actual performance
- Fine-tune rate limits based on usage patterns

### Week 2: Advanced Configuration

- Implement additional providers (Anthropic, Google AI)
- Configure advanced caching strategies
- Setup automated cost reporting
- Implement custom monitoring dashboards

### Month 1: Scale and Optimize

- Analyze usage patterns for optimization
- Implement auto-scaling if needed
- Review security audit logs
- Plan for additional AI capabilities

---

*Migration Guide Version: 1.0*
*Last Updated: 2025-07-06*
*Estimated Success Rate: 95%+ with proper preparation*