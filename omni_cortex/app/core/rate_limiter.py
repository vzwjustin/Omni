"""
Rate Limiter for Omni-Cortex MCP Tools

Provides token bucket rate limiting with configurable limits per tool category.
Uses asyncio-native design with no external dependencies.

Thread Safety Model:
--------------------
This module uses a hybrid locking strategy:

1. threading.Lock (_rate_limiter_lock): Protects singleton creation of the
   global RateLimiter instance. Used because asyncio.Lock() cannot be created
   at module level (no event loop exists yet) and because we need cross-thread
   protection for the singleton pattern.

2. asyncio.Lock (RateLimiter._lock): Used within the RateLimiter instance for
   async-safe access to token buckets during check_rate_limit(). This is created
   inside __init__ when an event loop is typically available.

This separation allows:
- Safe module-level singleton pattern (threading.Lock)
- Efficient async operations within the rate limiter (asyncio.Lock)
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import structlog

from .settings import get_settings

logger = structlog.get_logger("rate_limiter")


@dataclass
class TokenBucket:
    """Token bucket rate limiter for a single resource."""
    
    capacity: int  # Max tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    
    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate seconds until tokens will be available."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting by tool category."""
    
    # Requests per minute for each category
    llm_tools_rpm: int = 30  # Tools that call LLMs (reason, prepare_context)
    search_tools_rpm: int = 60  # RAG search tools
    memory_tools_rpm: int = 120  # Memory operations
    utility_tools_rpm: int = 120  # Health, list_frameworks, etc.
    
    # Global limit across all tools
    global_rpm: int = 200
    
    # Input size limits (bytes)
    max_input_size: int = 100_000  # 100KB max input
    max_code_size: int = 50_000  # 50KB max code for execution


# Tool categorization for rate limiting
TOOL_CATEGORIES: Dict[str, str] = {
    # LLM tools - expensive, rate limit aggressively
    "reason": "llm",
    "prepare_context": "llm",
    
    # Search tools - moderate rate limiting
    "search_documentation": "search",
    "search_frameworks_by_name": "search",
    "search_by_category": "search",
    "search_function": "search",
    "search_class": "search",
    "search_docs_only": "search",
    "search_framework_category": "search",
    
    # Memory tools - light rate limiting
    "get_context": "memory",
    "save_context": "memory",
    
    # Utility tools - light rate limiting
    "health": "utility",
    "list_frameworks": "utility",
    "recommend": "utility",
    "count_tokens": "utility",
    "compress_content": "utility",
    "detect_truncation": "utility",
    "manage_claude_md": "utility",
    "execute_code": "utility",
}


class RateLimiter:
    """
    Async-safe rate limiter for MCP tools.
    
    Uses token buckets per category and a global bucket.
    Thread-safe via asyncio locks.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._lock = asyncio.Lock()
        
        # Create buckets (tokens = capacity, refill = per second)
        self._buckets: Dict[str, TokenBucket] = {
            "llm": TokenBucket(
                capacity=self.config.llm_tools_rpm,
                refill_rate=self.config.llm_tools_rpm / 60.0
            ),
            "search": TokenBucket(
                capacity=self.config.search_tools_rpm,
                refill_rate=self.config.search_tools_rpm / 60.0
            ),
            "memory": TokenBucket(
                capacity=self.config.memory_tools_rpm,
                refill_rate=self.config.memory_tools_rpm / 60.0
            ),
            "utility": TokenBucket(
                capacity=self.config.utility_tools_rpm,
                refill_rate=self.config.utility_tools_rpm / 60.0
            ),
            "global": TokenBucket(
                capacity=self.config.global_rpm,
                refill_rate=self.config.global_rpm / 60.0
            ),
        }
        
        # Default bucket for unknown tools (think_* tools)
        self._default_category = "utility"
    
    def _get_category(self, tool_name: str) -> str:
        """Get the rate limit category for a tool."""
        # Handle think_* tools
        if tool_name.startswith("think_"):
            return "llm"  # Framework tools are treated as LLM tools
        return TOOL_CATEGORIES.get(tool_name, self._default_category)
    
    async def check_rate_limit(self, tool_name: str) -> tuple[bool, str]:
        """
        Check if a tool call is allowed under rate limits.
        
        Returns:
            (allowed, error_message) - If allowed is False, error_message explains why.
        """
        async with self._lock:
            category = self._get_category(tool_name)
            category_bucket = self._buckets[category]
            global_bucket = self._buckets["global"]
            
            # Check global limit first
            if not global_bucket.try_acquire():
                wait_time = global_bucket.time_until_available()
                logger.warning(
                    "rate_limit_exceeded",
                    tool=tool_name,
                    category="global",
                    retry_after_seconds=round(wait_time, 1)
                )
                return False, f"Global rate limit exceeded. Retry in {wait_time:.1f}s"
            
            # Check category limit
            if not category_bucket.try_acquire():
                # Refund the global token since we're rejecting
                global_bucket.tokens = min(
                    global_bucket.capacity,
                    global_bucket.tokens + 1
                )
                wait_time = category_bucket.time_until_available()
                logger.warning(
                    "rate_limit_exceeded",
                    tool=tool_name,
                    category=category,
                    retry_after_seconds=round(wait_time, 1)
                )
                return False, f"Rate limit for {category} tools exceeded. Retry in {wait_time:.1f}s"
            
            return True, ""
    
    def validate_input_size(self, arguments: Dict[str, Any], tool_name: str) -> tuple[bool, str]:
        """
        Validate that input sizes are within limits.
        
        Returns:
            (valid, error_message)
        """
        # Calculate total input size
        total_size = 0
        for key, value in arguments.items():
            if isinstance(value, str):
                total_size += len(value.encode("utf-8", errors="ignore"))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total_size += len(item.encode("utf-8", errors="ignore"))
        
        # Check general input size
        if total_size > self.config.max_input_size:
            logger.warning(
                "input_size_exceeded",
                tool=tool_name,
                size=total_size,
                max_size=self.config.max_input_size
            )
            return False, f"Input size ({total_size} bytes) exceeds limit ({self.config.max_input_size} bytes)"
        
        # Special check for code execution
        if tool_name == "execute_code":
            code = arguments.get("code", "")
            if len(code.encode("utf-8", errors="ignore")) > self.config.max_code_size:
                return False, f"Code size exceeds limit ({self.config.max_code_size} bytes)"
        
        return True, ""


# Global rate limiter instance (lazy initialization)
_rate_limiter: Optional[RateLimiter] = None
# threading.Lock for module-level singleton protection - works across threads
# and can be created at module load time (unlike asyncio.Lock which needs an event loop)
_rate_limiter_lock = threading.Lock()


async def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance (thread-safe singleton).

    Uses threading.Lock (not asyncio.Lock) for singleton protection because:
    - asyncio.Lock cannot be created at module level (no event loop yet)
    - threading.Lock provides true cross-thread protection
    - The critical section contains only synchronous operations (settings read, object creation)

    The RateLimiter instance itself uses asyncio.Lock internally for its async operations.
    """
    global _rate_limiter

    # Fast path: already initialized (no lock needed for read due to GIL)
    if _rate_limiter is not None:
        return _rate_limiter

    # Slow path: acquire lock and double-check
    with _rate_limiter_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _rate_limiter is None:
            settings = get_settings()
            config = RateLimitConfig(
                llm_tools_rpm=getattr(settings, "rate_limit_llm_rpm", 30),
                search_tools_rpm=getattr(settings, "rate_limit_search_rpm", 60),
                memory_tools_rpm=getattr(settings, "rate_limit_memory_rpm", 120),
                utility_tools_rpm=getattr(settings, "rate_limit_utility_rpm", 120),
                global_rpm=getattr(settings, "rate_limit_global_rpm", 200),
            )
            _rate_limiter = RateLimiter(config)

    return _rate_limiter
