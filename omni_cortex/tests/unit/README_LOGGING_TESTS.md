# Logging Utilities Test Guide

## Quick Start

### Run Logging Tests
```bash
# Run only logging utility tests
pytest tests/unit/test_logging_utils.py -v

# Run with coverage
pytest tests/unit/test_logging_utils.py --cov=app.core.logging_utils --cov-report=term-missing

# Run specific test class
pytest tests/unit/test_logging_utils.py::TestSanitizeApiKeys -v

# Run specific test
pytest tests/unit/test_logging_utils.py::TestSanitizeApiKeys::test_sanitize_api_key_with_equals -v
```

### In Docker
```bash
# Run tests in Docker container
docker-compose exec omni-cortex python -m pytest tests/unit/test_logging_utils.py -v
```

## Test Structure

### Test Classes
1. **TestSanitizeApiKeys** - API key, token, password patterns (11 tests)
2. **TestSanitizeEnvVars** - Environment variable patterns (7 tests)
3. **TestSanitizeError** - Exception sanitization (5 tests)
4. **TestSanitizeDict** - Dictionary sanitization (12 tests)
5. **TestSafeRepr** - Object representation (6 tests)
6. **TestSanitizeLogRecord** - Log record sanitization (4 tests)
7. **TestCreateSafeErrorDetails** - Error detail creation (4 tests)
8. **TestIntegrationScenarios** - Real-world scenarios (3 tests)

**Total: 52 test cases**

## What's Tested

### API Key Patterns
- `api_key=sk-...`
- `api_key: "sk-..."`
- `api-key='sk-...'`
- `token = "eyJ..."`
- `secret: abcdef...`
- `password="MySecret..."`
- `authorization: "Bearer eyJ..."`

### Environment Variables
- `OPENAI_API_KEY=...`
- `ANTHROPIC_API_KEY=...`
- `GOOGLE_API_KEY=...`
- `OPENROUTER_API_KEY=...`

### Dictionary Keys (case-insensitive)
- `api_key`, `apikey`, `api-key`
- `token`, `auth_token`, `access_token`
- `secret`, `password`, `pwd`
- `authorization`, `auth`

## Verification Checklist

When adding new patterns or modifying sanitization logic:

- [ ] Add test case to appropriate test class
- [ ] Verify `[REDACTED]` appears in output
- [ ] Verify original sensitive value is removed
- [ ] Verify non-sensitive data is preserved
- [ ] Test with nested structures (dicts, lists)
- [ ] Test case-insensitive matching
- [ ] Test with multiple occurrences
- [ ] Update integration tests if needed

## Common Test Patterns

### Testing a New Redaction Pattern
```python
def test_new_pattern(self):
    """Test description."""
    text = 'sensitive_field=secret_value_12345678901234567890'
    result = sanitize_api_keys(text)
    assert '[REDACTED]' in result
    assert 'secret_value' not in result
```

### Testing Nested Structures
```python
def test_nested_redaction(self):
    """Test nested dict redaction."""
    data = {'outer': {'inner': {'api_key': 'sk-123456'}}}
    result = sanitize_dict(data)
    assert '[REDACTED]' in result['outer']['inner']['api_key']
```

### Testing Preservation
```python
def test_preserves_safe_data(self):
    """Test safe data is not modified."""
    data = {'safe_field': 'safe_value', 'api_key': 'sk-123456'}
    result = sanitize_dict(data)
    assert result['safe_field'] == 'safe_value'
    assert '[REDACTED]' in result['api_key']
```

## Expected Output Format

### Text Redaction
```
Input:  api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz
Output: api_key=[REDACTED]
```

### Dict Redaction (Long Values)
```
Input:  {'api_key': 'sk-1234567890'}
Output: {'api_key': 'sk-1...[REDACTED]'}
```

### Dict Redaction (Short Values ≤4 chars)
```
Input:  {'password': 'abc'}
Output: {'password': '[REDACTED]'}
```

### Env Var Redaction
```
Input:  OPENAI_API_KEY=sk-1234567890
Output: OPENAI_API_KEY="[REDACTED]"
```

## Troubleshooting

### Test Failures

**Pattern Not Matching:**
- Check regex pattern in `app/core/logging_utils.py`
- Verify minimum length requirement (usually 20 chars)
- Check case-insensitive flag

**False Positives:**
- Verify minimum length prevents short value matching
- Check pattern specificity

**Nested Data Not Redacted:**
- Ensure `sanitize_dict` is called recursively
- Check if data type is dict or list

### Debug Mode

Add print statements to see intermediate values:
```python
def test_debug_pattern(self):
    text = 'your test case'
    result = sanitize_api_keys(text)
    print(f"Input:  {text}")
    print(f"Output: {result}")
    assert '[REDACTED]' in result
```

Run with output:
```bash
pytest tests/unit/test_logging_utils.py::test_debug_pattern -v -s
```

## Security Notes

### What's Protected
- ✓ API keys from all major providers
- ✓ JWT and Bearer tokens
- ✓ Passwords and secrets
- ✓ Environment variable values
- ✓ Authorization headers

### What's NOT Protected
These require additional handling:
- ⚠️  Credit card numbers (add pattern if needed)
- ⚠️  Social security numbers (add pattern if needed)
- ⚠️  Personal identifiable information (context-specific)
- ⚠️  Private keys (different format from API keys)

### Adding New Patterns

To add a new sensitive data pattern:

1. Add pattern to `API_KEY_PATTERNS` or `ENV_VAR_PATTERNS` in `app/core/logging_utils.py`
2. Add test cases to `tests/unit/test_logging_utils.py`
3. Run tests to verify: `pytest tests/unit/test_logging_utils.py -v`
4. Update this README with the new pattern

Example:
```python
# In app/core/logging_utils.py
API_KEY_PATTERNS = [
    # ... existing patterns ...
    r'credit_card["\']?\s*[:=]\s*["\']?(\d{16})',  # New pattern
]

# In tests/unit/test_logging_utils.py
def test_sanitize_credit_card(self):
    """Test credit card redaction."""
    text = 'credit_card=1234567890123456'
    result = sanitize_api_keys(text)
    assert '[REDACTED]' in result
    assert '1234567890123456' not in result
```

## Related Files

- `/Users/justinadams/thinking-frameworks/omni_cortex/app/core/logging_utils.py` - Implementation
- `/Users/justinadams/thinking-frameworks/omni_cortex/tests/unit/test_logging_utils.py` - Tests
- `/Users/justinadams/thinking-frameworks/omni_cortex/LOGGING_TEST_RESULTS.md` - Test results documentation

## CI/CD Integration

Add to your CI pipeline:
```yaml
- name: Run Logging Security Tests
  run: |
    pytest tests/unit/test_logging_utils.py -v
    pytest tests/unit/test_logging_utils.py --cov=app.core.logging_utils --cov-report=xml
```

Require 100% pass rate for deployment.
