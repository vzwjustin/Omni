"""
Tests for logging utilities (app/core/logging_utils.py).

Verifies that API keys, tokens, and other sensitive data are properly
redacted from logs and error messages.
"""

from app.core.logging_utils import (
    create_safe_error_details,
    safe_repr,
    sanitize_api_keys,
    sanitize_dict,
    sanitize_env_vars,
    sanitize_error,
    sanitize_log_record,
)


class TestSanitizeApiKeys:
    """Test sanitize_api_keys with various API key patterns."""

    def test_sanitize_api_key_with_equals(self):
        """Test API key with equals sign."""
        text = 'api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result

    def test_sanitize_api_key_with_colon(self):
        """Test API key with colon."""
        text = 'api_key: "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz"'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'sk-proj-1234567890abcdefghijklmnopqrstuvwxyz' not in result

    def test_sanitize_api_key_with_quotes(self):
        """Test API key with quotes."""
        text = "api-key='sk_test_1234567890abcdefghij'"
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'sk_test_1234567890abcdefghij' not in result

    def test_sanitize_token_pattern(self):
        """Test token pattern."""
        text = 'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in result

    def test_sanitize_secret_pattern(self):
        """Test secret pattern."""
        text = 'secret: abcdefghijklmnopqrstuvwxyz123456'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'abcdefghijklmnopqrstuvwxyz123456' not in result

    def test_sanitize_password_pattern(self):
        """Test password pattern."""
        text = 'password="MySecretPassword123!"'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'MySecretPassword123!' not in result

    def test_sanitize_bearer_token_header_format(self):
        """Test Bearer token pattern in HTTP header format."""
        text = 'authorization: "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in result

    def test_sanitize_bearer_token_dict_key(self):
        """Test Bearer token as dict value (HTTP headers)."""
        text = "{'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'}"
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in result

    def test_sanitize_case_insensitive(self):
        """Test case-insensitive matching."""
        text = 'API_KEY=sk-1234567890abcdefghijklmnopqrstuvwxyz'
        result = sanitize_api_keys(text)
        assert '[REDACTED]' in result
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result

    def test_sanitize_multiple_keys(self):
        """Test multiple API keys in same text."""
        text = '''
        api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz
        token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
        secret: my_secret_value_123456
        '''
        result = sanitize_api_keys(text)
        assert result.count('[REDACTED]') == 3
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in result
        assert 'my_secret_value_123456' not in result

    def test_sanitize_preserves_non_sensitive(self):
        """Test that non-sensitive text is preserved."""
        text = 'Hello world! This is a normal message with no secrets.'
        result = sanitize_api_keys(text)
        assert result == text
        assert '[REDACTED]' not in result

    def test_sanitize_short_values_not_matched(self):
        """Test that short values (< 20 chars) are not matched as API keys."""
        text = 'api_key=short'
        result = sanitize_api_keys(text)
        # Short values should not be redacted by the API_KEY_PATTERNS
        assert result == text


class TestSanitizeEnvVars:
    """Test sanitize_env_vars with environment variable patterns."""

    def test_sanitize_openai_api_key(self):
        """Test OPENAI_API_KEY redaction."""
        text = 'OPENAI_API_KEY=sk-1234567890abcdefghijklmnopqrstuvwxyz'
        result = sanitize_env_vars(text)
        assert '[REDACTED]' in result
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result
        assert 'OPENAI_API_KEY="[REDACTED]"' in result

    def test_sanitize_anthropic_api_key(self):
        """Test ANTHROPIC_API_KEY redaction."""
        text = 'ANTHROPIC_API_KEY="sk-ant-1234567890"'
        result = sanitize_env_vars(text)
        assert '[REDACTED]' in result
        assert 'sk-ant-1234567890' not in result

    def test_sanitize_google_api_key(self):
        """Test GOOGLE_API_KEY redaction."""
        text = "GOOGLE_API_KEY='AIzaSyD1234567890'"
        result = sanitize_env_vars(text)
        assert '[REDACTED]' in result
        assert 'AIzaSyD1234567890' not in result

    def test_sanitize_openrouter_api_key(self):
        """Test OPENROUTER_API_KEY redaction."""
        text = 'OPENROUTER_API_KEY=sk-or-v1-1234567890'
        result = sanitize_env_vars(text)
        assert '[REDACTED]' in result
        assert 'sk-or-v1-1234567890' not in result

    def test_sanitize_multiple_env_vars(self):
        """Test multiple env vars in same text."""
        text = '''
        OPENAI_API_KEY=sk-1234567890
        ANTHROPIC_API_KEY=sk-ant-9876543210
        '''
        result = sanitize_env_vars(text)
        assert result.count('[REDACTED]') == 2
        assert 'sk-1234567890' not in result
        assert 'sk-ant-9876543210' not in result

    def test_sanitize_env_var_with_quotes(self):
        """Test env var with quotes."""
        text = 'OPENAI_API_KEY="sk-1234567890"'
        result = sanitize_env_vars(text)
        assert '[REDACTED]' in result
        assert 'sk-1234567890' not in result

    def test_sanitize_preserves_env_var_name(self):
        """Test that env var names are preserved."""
        text = 'OPENAI_API_KEY=sk-1234567890'
        result = sanitize_env_vars(text)
        assert 'OPENAI_API_KEY' in result


class TestSanitizeError:
    """Test sanitize_error with exceptions containing API keys."""

    def test_sanitize_error_with_api_key(self):
        """Test error message containing API key."""
        error = ValueError('Failed with api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz')
        result = sanitize_error(error)
        assert '[REDACTED]' in result
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result

    def test_sanitize_error_with_env_var(self):
        """Test error message containing env var."""
        error = RuntimeError('OPENAI_API_KEY=sk-1234567890 is invalid')
        result = sanitize_error(error)
        assert '[REDACTED]' in result
        assert 'sk-1234567890' not in result

    def test_sanitize_error_with_bearer_token_header(self):
        """Test error message containing Bearer token in header format."""
        error = Exception('Request failed: authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9')
        result = sanitize_error(error)
        assert '[REDACTED]' in result
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in result

    def test_sanitize_error_truncates_long_messages(self):
        """Test that very long error messages are truncated."""
        long_message = 'Error: ' + 'x' * 2000
        error = Exception(long_message)
        result = sanitize_error(error)
        assert len(result) <= 1020  # 1000 + "... [truncated]"
        assert '[truncated]' in result

    def test_sanitize_error_preserves_safe_message(self):
        """Test that safe error messages are preserved."""
        error = ValueError('Invalid input format')
        result = sanitize_error(error)
        assert 'Invalid input format' in result
        assert '[REDACTED]' not in result


class TestSanitizeDict:
    """Test sanitize_dict with sensitive key names."""

    def test_sanitize_api_key_field(self):
        """Test redacting api_key field."""
        data = {'api_key': 'sk-1234567890abcdefghijklmnopqrstuvwxyz'}
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['api_key']
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result['api_key']

    def test_sanitize_token_field(self):
        """Test redacting token field."""
        data = {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'}
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['token']

    def test_sanitize_password_field(self):
        """Test redacting password field."""
        data = {'password': 'MySecretPassword123'}
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['password']

    def test_sanitize_authorization_field(self):
        """Test redacting authorization field."""
        data = {'authorization': 'Bearer token123'}
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['authorization']

    def test_sanitize_shows_first_chars(self):
        """Test that first 4 chars are shown for long values."""
        data = {'api_key': 'sk-1234567890'}
        result = sanitize_dict(data)
        assert result['api_key'].startswith('sk-1')
        assert '[REDACTED]' in result['api_key']

    def test_sanitize_short_values(self):
        """Test that short values (<=4 chars) are fully redacted."""
        data = {'password': 'abc'}
        result = sanitize_dict(data)
        assert result['password'] == '[REDACTED]'

    def test_sanitize_case_insensitive_keys(self):
        """Test case-insensitive key matching."""
        data = {
            'API_KEY': 'sk-123456',
            'Token': 'token123',
            'SECRET': 'secret123'
        }
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['API_KEY']
        assert '[REDACTED]' in result['Token']
        assert '[REDACTED]' in result['SECRET']

    def test_sanitize_nested_dicts(self):
        """Test sanitizing nested dictionaries."""
        data = {
            'config': {
                'api_key': 'sk-123456',
                'safe_value': 'hello'
            },
            'safe_field': 'world'
        }
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['config']['api_key']
        assert result['config']['safe_value'] == 'hello'
        assert result['safe_field'] == 'world'

    def test_sanitize_list_of_dicts(self):
        """Test sanitizing lists containing dicts."""
        data = {
            'items': [
                {'api_key': 'sk-111111', 'name': 'item1'},
                {'token': 'token222', 'name': 'item2'}
            ]
        }
        result = sanitize_dict(data)
        assert '[REDACTED]' in result['items'][0]['api_key']
        assert '[REDACTED]' in result['items'][1]['token']
        assert result['items'][0]['name'] == 'item1'
        assert result['items'][1]['name'] == 'item2'

    def test_sanitize_custom_redact_keys(self):
        """Test custom redact keys."""
        data = {'custom_secret': 'my_value', 'normal_field': 'safe'}
        result = sanitize_dict(data, redact_keys=['custom_secret'])
        assert '[REDACTED]' in result['custom_secret']
        assert result['normal_field'] == 'safe'

    def test_sanitize_preserves_safe_fields(self):
        """Test that safe fields are not redacted."""
        data = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30,
            'active': True
        }
        result = sanitize_dict(data)
        assert result == data

    def test_sanitize_variations_of_sensitive_keys(self):
        """Test various forms of sensitive key names."""
        data = {
            'apikey': 'key1',
            'api-key': 'key2',
            'api_key': 'key3',
            'auth_token': 'token1',
            'access_token': 'token2',
            'pwd': 'pass1'
        }
        result = sanitize_dict(data)
        for key in data:
            assert '[REDACTED]' in result[key]


class TestSafeRepr:
    """Test safe_repr truncates long objects."""

    def test_safe_repr_short_object(self):
        """Test short object representation."""
        obj = {'key': 'value'}
        result = safe_repr(obj)
        assert 'key' in result
        assert 'value' in result

    def test_safe_repr_truncates_long_object(self):
        """Test long object is truncated."""
        obj = {'key': 'x' * 1000}
        result = safe_repr(obj, max_length=100)
        assert len(result) <= 120  # 100 + "... [truncated]"
        assert '[truncated]' in result

    def test_safe_repr_sanitizes_api_keys(self):
        """Test API keys in repr are sanitized."""
        obj = {'api_key': 'sk-1234567890abcdefghijklmnopqrstuvwxyz'}
        result = safe_repr(obj)
        assert '[REDACTED]' in result
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result

    def test_safe_repr_handles_repr_failure(self):
        """Test fallback when repr fails."""
        class BadRepr:
            def __repr__(self):
                raise ValueError("repr failed")

        obj = BadRepr()
        result = safe_repr(obj)
        assert 'BadRepr' in result
        assert 'repr failed' in result

    def test_safe_repr_custom_max_length(self):
        """Test custom max_length parameter."""
        obj = 'x' * 1000
        result = safe_repr(obj, max_length=50)
        assert len(result) <= 70  # 50 + "... [truncated]"

    def test_safe_repr_various_types(self):
        """Test safe_repr with various object types."""
        test_cases = [
            42,
            3.14,
            "hello",
            [1, 2, 3],
            (4, 5, 6),
            {7, 8, 9},
            None,
            True
        ]
        for obj in test_cases:
            result = safe_repr(obj)
            assert isinstance(result, str)
            assert '[REDACTED]' not in result  # No sensitive data


class TestSanitizeLogRecord:
    """Test sanitize_log_record with structured log records."""

    def test_sanitize_message_field(self):
        """Test message field is sanitized."""
        record = {'message': 'Using api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz'}
        result = sanitize_log_record(record)
        assert '[REDACTED]' in result['message']
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in result['message']

    def test_sanitize_dict_fields(self):
        """Test dict fields are sanitized."""
        record = {
            'message': 'Request failed',
            'context': {
                'api_key': 'sk-123456',
                'user': 'john'
            }
        }
        result = sanitize_log_record(record)
        assert '[REDACTED]' in result['context']['api_key']
        assert result['context']['user'] == 'john'

    def test_sanitize_string_fields(self):
        """Test string fields are sanitized."""
        record = {
            'message': 'Log entry',
            'error': 'Failed with token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
            'safe_field': 'no secrets here'
        }
        result = sanitize_log_record(record)
        assert '[REDACTED]' in result['error']
        assert result['safe_field'] == 'no secrets here'

    def test_sanitize_preserves_record_structure(self):
        """Test record structure is preserved."""
        record = {
            'timestamp': '2024-01-01T00:00:00Z',
            'level': 'INFO',
            'message': 'Normal log message'
        }
        result = sanitize_log_record(record)
        assert result['timestamp'] == record['timestamp']
        assert result['level'] == record['level']
        assert result['message'] == record['message']


class TestCreateSafeErrorDetails:
    """Test create_safe_error_details with exceptions."""

    def test_create_details_basic_exception(self):
        """Test basic exception details."""
        error = ValueError('Invalid input')
        details = create_safe_error_details(error)
        assert details['type'] == 'ValueError'
        assert details['message'] == 'Invalid input'

    def test_create_details_sanitizes_message(self):
        """Test exception message is sanitized."""
        error = RuntimeError('Failed with api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz')
        details = create_safe_error_details(error)
        assert '[REDACTED]' in details['message']
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in details['message']

    def test_create_details_with_details_attribute(self):
        """Test exception with details attribute."""
        class CustomError(Exception):
            def __init__(self, message, details):
                super().__init__(message)
                self.details = details

        error = CustomError(
            'Custom error',
            {'api_key': 'sk-123456', 'user': 'john'}
        )
        details = create_safe_error_details(error)
        assert details['type'] == 'CustomError'
        assert '[REDACTED]' in details['details']['api_key']
        assert details['details']['user'] == 'john'

    def test_create_details_without_details_attribute(self):
        """Test exception without details attribute."""
        error = ValueError('Standard error')
        details = create_safe_error_details(error)
        assert 'details' not in details
        assert details['type'] == 'ValueError'
        assert details['message'] == 'Standard error'


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_log_sanitization_workflow(self):
        """Test complete log sanitization workflow."""
        # Simulate a log record from a failed API call
        record = {
            'timestamp': '2024-01-01T12:00:00Z',
            'level': 'ERROR',
            'message': 'API call failed with api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz',
            'context': {
                'url': 'https://api.example.com/v1/chat',
                'headers': {
                    'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
                    'content-type': 'application/json'
                },
                'env_config': 'OPENAI_API_KEY=sk-proj-abcdefg'
            }
        }

        # Sanitize the record
        result = sanitize_log_record(record)

        # Verify all sensitive data is redacted
        assert '[REDACTED]' in result['message']
        assert 'sk-1234567890abcdefghijklmnopqrstuvwxyz' not in str(result)
        assert '[REDACTED]' in result['context']['headers']['authorization']
        assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in str(result)
        # env_config is sanitized because it contains env var pattern
        assert '[REDACTED]' in result['context']['env_config']

        # Verify safe data is preserved
        assert result['timestamp'] == '2024-01-01T12:00:00Z'
        assert result['level'] == 'ERROR'
        assert result['context']['url'] == 'https://api.example.com/v1/chat'
        assert result['context']['headers']['content-type'] == 'application/json'

    def test_error_with_nested_sensitive_data(self):
        """Test error containing nested sensitive data."""
        class APIError(Exception):
            def __init__(self, message, details):
                super().__init__(message)
                self.details = details

        error = APIError(
            'API request failed',
            {
                'request': {
                    'api_key': 'sk-1234567890',
                    'endpoint': '/v1/chat'
                },
                'response': {
                    'error': 'Unauthorized',
                    'token': 'invalid_token_xyz123456789'
                }
            }
        )

        details = create_safe_error_details(error)

        # Verify all sensitive fields are redacted
        assert '[REDACTED]' in details['details']['request']['api_key']
        assert '[REDACTED]' in details['details']['response']['token']

        # Verify safe fields are preserved
        assert details['details']['request']['endpoint'] == '/v1/chat'
        assert details['details']['response']['error'] == 'Unauthorized'

    def test_all_env_var_patterns(self):
        """Test all environment variable patterns are redacted."""
        text = '''
        export OPENAI_API_KEY=sk-1111111111
        export ANTHROPIC_API_KEY=sk-ant-2222222222
        export GOOGLE_API_KEY=AIzaSyD3333333333
        export OPENROUTER_API_KEY=sk-or-v1-4444444444
        '''
        result = sanitize_env_vars(text)

        # All keys should be redacted
        assert result.count('[REDACTED]') == 4
        assert 'sk-1111111111' not in result
        assert 'sk-ant-2222222222' not in result
        assert 'AIzaSyD3333333333' not in result
        assert 'sk-or-v1-4444444444' not in result

        # Variable names should be preserved
        assert 'OPENAI_API_KEY' in result
        assert 'ANTHROPIC_API_KEY' in result
        assert 'GOOGLE_API_KEY' in result
        assert 'OPENROUTER_API_KEY' in result
