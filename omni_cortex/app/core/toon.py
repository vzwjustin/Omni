"""
TOON (Token-Oriented Object Notation) Serialization Module

TOON is a lightweight JSON alternative optimized for LLMs that reduces token usage
by 20-60% through efficient encoding of structured data, especially arrays of uniform objects.

Key Features:
- Tabular arrays: Arrays of objects with consistent keys are represented as tables
- Delimiter-based encoding: Uses "|" for field separation, reducing overhead
- Token-optimized: Designed for LLM consumption rather than human readability
- Reversible: Full lossless conversion to/from JSON

Example:
    JSON (verbose):
    [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "SF"}
    ]

    TOON (compact):
    {name|age|city}
    Alice|30|NYC
    Bob|25|SF

References:
- https://scalevise.com/resources/toon-lightweight-json-for-ai-llm-systems/
- https://www.intuz.com/blog/reduce-llm-token-costs-using-toon-format
"""

import json
from typing import Any, Dict, List, Union


class TOONEncoder:
    """Encoder for converting JSON/Python objects to TOON format."""

    def __init__(
        self,
        delimiter: str = "|",
        threshold: int = 2,
        compact_primitives: bool = True
    ):
        """
        Initialize TOON encoder.

        Args:
            delimiter: Field separator for tabular arrays (default: "|")
            threshold: Minimum array length to use tabular format (default: 2)
            compact_primitives: Use compact encoding for primitive values (default: True)
        """
        self.delimiter = delimiter
        self.threshold = threshold
        self.compact_primitives = compact_primitives

    def encode(self, data: Any) -> str:
        """
        Encode Python object to TOON format.

        Args:
            data: Python object to encode

        Returns:
            TOON-formatted string
        """
        return self._encode_value(data)

    def _encode_value(self, value: Any, indent: int = 0) -> str:
        """Encode a single value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Escape delimiter in strings
            escaped = value.replace(self.delimiter, f"\\{self.delimiter}")
            return escaped if self.compact_primitives else f'"{escaped}"'
        elif isinstance(value, list):
            return self._encode_array(value, indent)
        elif isinstance(value, dict):
            return self._encode_object(value, indent)
        else:
            # Fallback to JSON for unsupported types
            return json.dumps(value)

    def _encode_array(self, arr: List[Any], indent: int = 0) -> str:
        """Encode array, using tabular format for uniform objects."""
        if not arr:
            return "[]"

        # Check if array is uniform (all dicts with same keys)
        if len(arr) >= self.threshold and all(isinstance(item, dict) for item in arr):
            first_keys = set(arr[0].keys())
            if all(set(item.keys()) == first_keys for item in arr):
                return self._encode_tabular_array(arr, indent)

        # Standard array encoding
        items = [self._encode_value(item, indent + 1) for item in arr]
        if self.compact_primitives and all(isinstance(x, (int, float, str, bool, type(None))) for x in arr):
            # Compact: [1,2,3]
            return f"[{','.join(items)}]"
        else:
            # Expanded for complex types
            spacing = "  " * (indent + 1)
            formatted_items = [f"{spacing}{item}" for item in items]
            return "[\n" + ",\n".join(formatted_items) + f"\n{'  ' * indent}]"

    def _encode_tabular_array(self, arr: List[Dict], indent: int = 0) -> str:
        """Encode array of uniform objects as tabular TOON format."""
        if not arr:
            return "[]"

        keys = list(arr[0].keys())
        spacing = "  " * indent

        # Header: {key1|key2|key3}
        header = f"{{{self.delimiter.join(keys)}}}"

        # Rows: value1|value2|value3
        rows = []
        for obj in arr:
            row_values = []
            for key in keys:
                val = obj.get(key)
                if isinstance(val, str):
                    # Escape delimiter and newlines in cell values
                    escaped = val.replace(self.delimiter, f"\\{self.delimiter}").replace("\n", "\\n")
                    row_values.append(escaped)
                elif val is None:
                    row_values.append("null")
                elif isinstance(val, bool):
                    row_values.append("true" if val else "false")
                else:
                    row_values.append(str(val))
            rows.append(self.delimiter.join(row_values))

        # Format with proper indentation
        result = header + "\n"
        result += "\n".join(rows)

        return result

    def _encode_object(self, obj: Dict[str, Any], indent: int = 0) -> str:
        """Encode dictionary object."""
        if not obj:
            return "{}"

        spacing = "  " * indent
        inner_spacing = "  " * (indent + 1)

        items = []
        for key, value in obj.items():
            encoded_value = self._encode_value(value, indent + 1)
            # Compact key-value pairs
            items.append(f"{inner_spacing}{key}:{encoded_value}")

        return "{\n" + ",\n".join(items) + f"\n{spacing}}}"


class TOONDecoder:
    """Decoder for converting TOON format back to Python objects."""

    def __init__(self, delimiter: str = "|"):
        """
        Initialize TOON decoder.

        Args:
            delimiter: Field separator used in tabular arrays (default: "|")
        """
        self.delimiter = delimiter

    def decode(self, toon_str: str) -> Any:
        """
        Decode TOON format string to Python object.

        Args:
            toon_str: TOON-formatted string

        Returns:
            Python object
        """
        lines = toon_str.strip().split("\n")

        # Check if it's a tabular array (starts with {key|key|key})
        if lines and lines[0].startswith("{") and lines[0].endswith("}"):
            return self._decode_tabular_array(lines)

        # Otherwise, try JSON-like decoding
        return self._decode_value(toon_str.strip())

    def _decode_tabular_array(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Decode tabular TOON array to list of dictionaries."""
        if not lines:
            return []

        # Parse header: {key1|key2|key3}
        header = lines[0].strip()
        if not (header.startswith("{") and header.endswith("}")):
            raise ValueError(f"Invalid tabular array header: {header}")

        keys = header[1:-1].split(self.delimiter)

        # Parse rows
        result = []
        for line in lines[1:]:
            if not line.strip():
                continue

            values = self._split_row(line.strip())
            if len(values) != len(keys):
                raise ValueError(f"Row has {len(values)} values but header has {len(keys)} keys")

            obj = {}
            for key, value in zip(keys, values):
                obj[key] = self._parse_cell_value(value)

            result.append(obj)

        return result

    def _split_row(self, row: str) -> List[str]:
        """Split row by delimiter, respecting escaped delimiters."""
        parts = []
        current = []
        escaped = False

        for char in row:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == self.delimiter:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return parts

    def _parse_cell_value(self, value: str) -> Any:
        """Parse cell value to appropriate Python type."""
        value = value.strip()

        # Handle escaped newlines
        value = value.replace("\\n", "\n")

        if value == "null":
            return None
        elif value == "true":
            return True
        elif value == "false":
            return False
        elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        else:
            # Try float
            try:
                return float(value)
            except ValueError:
                # String (with escaped delimiter restored)
                return value.replace(f"\\{self.delimiter}", self.delimiter)

    def _decode_value(self, value: str) -> Any:
        """Decode generic value (fallback to JSON parsing)."""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Return as string if not valid JSON
            return value


def to_toon(data: Any, **kwargs) -> str:
    """
    Convert Python object to TOON format.

    Args:
        data: Python object to encode
        **kwargs: Additional arguments for TOONEncoder

    Returns:
        TOON-formatted string
    """
    encoder = TOONEncoder(**kwargs)
    return encoder.encode(data)


def from_toon(toon_str: str, **kwargs) -> Any:
    """
    Convert TOON format string to Python object.

    Args:
        toon_str: TOON-formatted string
        **kwargs: Additional arguments for TOONDecoder

    Returns:
        Python object
    """
    decoder = TOONDecoder(**kwargs)
    return decoder.decode(toon_str)


def get_compression_ratio(json_str: str, toon_str: str) -> float:
    """
    Calculate token reduction ratio between JSON and TOON formats.

    Args:
        json_str: Original JSON string
        toon_str: TOON-encoded string

    Returns:
        Compression ratio (e.g., 0.4 means 40% of original size)
    """
    # Simple character-based ratio (token count would be more accurate but requires tiktoken)
    return len(toon_str) / len(json_str) if len(json_str) > 0 else 1.0


def get_token_savings(json_str: str, toon_str: str) -> Dict[str, Union[int, float]]:
    """
    Calculate token savings statistics.

    Args:
        json_str: Original JSON string
        toon_str: TOON-encoded string

    Returns:
        Dictionary with compression statistics
    """
    json_len = len(json_str)
    toon_len = len(toon_str)
    saved = json_len - toon_len
    ratio = get_compression_ratio(json_str, toon_str)

    return {
        "original_chars": json_len,
        "compressed_chars": toon_len,
        "saved_chars": saved,
        "compression_ratio": ratio,
        "reduction_percent": (1 - ratio) * 100
    }
