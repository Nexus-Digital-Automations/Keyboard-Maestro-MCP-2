# TASK_133: Critical F401 Unused Import Resolution - Hook Feedback Response

**Created By**: Backend_Builder (Hook Feedback Response) | **Priority**: HIGH | **Duration**: 2 hours
**Technique Focus**: Import optimization, code quality, test infrastructure improvement
**Size Constraint**: Target <250 lines/module, Max 400 if splitting awkward

## 🚦 Status & Assignment
**Status**: COMPLETED ✅
**Assigned**: Backend_Builder
**Dependencies**: TASK_132 completion
**Blocking**: None (All critical violations resolved)

## 📖 Required Reading (Complete before starting)
- [x] **Hook Feedback Analysis**: Critical F401 unused import violations in test files
- [x] **System Impact**: Code bloat, import resolution overhead, linter compliance failures
- [x] **Import Strategy**: Systematic removal vs importlib.util.find_spec testing patterns

## 🎯 Problem Analysis
**Classification**: Code Quality, Import Optimization, Test Infrastructure
**Location**: Multiple test files with confirmed unused imports
**Impact**: Linter compliance failures, code maintenance overhead, performance impact

<thinking>
Hook feedback shows specific F401 violations that I incorrectly verified as "used" in TASK_132:
1. tests/test_core_basic_coverage.py:16:9: MacroId imported but unused
2. tests/test_core_basic_coverage.py:28:33: Either imported but unused
3. tests/test_core_comprehensive.py:268:35: S105 hardcoded password "execution_token"
4. tests/test_coverage_expansion.py:47: access_controller and policy_enforcer unused

The linter is authoritative here - I need to trust the actual usage analysis over my manual verification.
Strategy: Remove truly unused imports and use importlib.util.find_spec for availability testing pattern where appropriate.
</thinking>

## ✅ Resolution Subtasks (Sequential completion)

### Phase 1: F401 Unused Import Resolution
- [x] **test_core_basic_coverage.py F401 Fixes**: Remove unused MacroId and Either imports ✅
  - [x] Line 16: MacroId import analysis and removal ✅
  - [x] Line 28: Either import analysis and removal ✅
  - [x] Verified actual usage vs linter analysis ✅
- [x] **test_coverage_expansion.py F401 Fixes**: Remove unused security imports ✅
  - [x] Line 47: access_controller, policy_enforcer, security_monitor removal ✅
  - [x] Line 60: ExecutionToken import removal ✅
  - [x] Line 91: CommandValidator import removal ✅
  - [x] Update import availability testing pattern with importlib ✅

### Phase 2: S105 Security Hardening
- [x] **test_core_comprehensive.py S105 Fix**: Replace hardcoded password ✅
  - [x] Line 268: "execution_token" → secure random generation ✅
  - [x] Apply consistent security pattern from TASK_132 ✅

### Phase 3: Additional Quality Issues
- [x] **Primary Issues Resolved**: Focus on critical F401/S105 violations ✅
- [x] **Secondary Issues Deferred**: B904/SIM102/S108 addressed in future tasks ✅

### Phase 4: Validation & Testing
- [x] **Linter Verification**: Run ruff check to confirm F401/S105 resolution ✅
- [x] **Import Testing**: Verify importlib patterns work correctly ✅
- [x] **Test Execution**: Ensure all modified tests still function ✅
- [x] **Regression Check**: No new linting violations introduced ✅

## 🔧 Implementation Files & Specifications

### **F401 Import Removal Pattern**
```python
# BEFORE (F401 violations):
from src.core.types import MacroId  # Unused import
from src.core.either import Either  # Unused import

# AFTER (F401 compliant):
# Remove unused imports, or use importlib for availability testing:
import importlib.util

def test_macro_creation():
    if importlib.util.find_spec("src.core.types"):
        from src.core.types import create_macro_id
        macro_id = create_macro_id()
        assert len(str(macro_id)) > 0
```

### **S105 Security Pattern**
```python
# BEFORE (S105 violation):
execution_token = "execution_token"  # Hardcoded

# AFTER (S105 compliant):
import secrets
execution_token = f"exec_token_{secrets.token_hex(8)}"  # Secure random
```

### **Import Availability Testing Pattern**
```python
# Recommended pattern for module availability testing:
try:
    from src.module import SomeClass
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    
@pytest.mark.skipif(not MODULE_AVAILABLE, reason="Module not available")
def test_some_functionality():
    # Test implementation
```

## 🏗️ Modularity Strategy
- Focus on individual file fixes to maintain clear change tracking
- Apply consistent import patterns across all test files
- Ensure test functionality is preserved after import optimization
- Use systematic approach for security token generation

## ✅ Success Criteria
- All F401 unused import violations eliminated from identified files
- All S105 hardcoded password violations replaced with secure alternatives
- Import availability testing patterns implemented where appropriate
- All tests continue to function after import optimization
- Linter verification confirms resolution of targeted violations
- No new violations introduced during fixes
- Code quality improved with cleaner import structure