"""
Error handling utilities for Groq API integration.

Provides custom exceptions and retry logic for robust error handling.
"""

import time
from typing import Callable, Any, TypeVar
from datetime import datetime, timedelta
from collections import deque

T = TypeVar('T')


class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass


class GroqAPIError(RAGError):
    """Groq API-specific errors."""
    pass


class GroqAuthenticationError(GroqAPIError):
    """Raised when API key is invalid or expired."""
    pass


class GroqRateLimitError(GroqAPIError):
    """Raised when rate limit is exceeded."""
    pass


class GroqServiceUnavailableError(GroqAPIError):
    """Raised when Groq service is temporarily unavailable."""
    pass


class GroqTimeoutError(GroqAPIError):
    """Raised when Groq request times out."""
    pass


class UpstashError(RAGError):
    """Upstash Vector Database errors."""
    pass


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.exponential_base = exponential_base
    
    def get_backoff_time(self, attempt: int) -> float:
        """Calculate backoff time for attempt number."""
        backoff = self.initial_backoff * (self.exponential_base ** attempt)
        return min(backoff, self.max_backoff)


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    on_retry: Callable[[int, float, Exception], None] = None,
    **kwargs
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        args: Positional arguments for func
        config: RetryConfig instance
        on_retry: Callback on each retry (attempt, wait_time, exception)
        kwargs: Keyword arguments for func
    
    Returns:
        Result from func
    
    Raises:
        Last exception if all retries fail
    """
    
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_retries):
        try:
            return func(*args, **kwargs)
        
        except (GroqAuthenticationError, ValueError) as e:
            # Don't retry auth errors
            raise
        
        except (GroqRateLimitError, GroqServiceUnavailableError, GroqTimeoutError) as e:
            last_exception = e
            
            if attempt < config.max_retries - 1:
                backoff = config.get_backoff_time(attempt)
                
                if on_retry:
                    on_retry(attempt + 1, backoff, e)
                
                time.sleep(backoff)
            else:
                raise
        
        except Exception as e:
            last_exception = e
            
            if attempt < config.max_retries - 1:
                backoff = config.get_backoff_time(attempt)
                
                if on_retry:
                    on_retry(attempt + 1, backoff, e)
                
                time.sleep(backoff)
            else:
                raise
    
    if last_exception:
        raise last_exception


def parse_groq_error(error: Exception) -> tuple[str, type]:
    """
    Parse Groq error and return error type.
    
    Returns:
        (error_message, error_class)
    """
    
    error_str = str(error).lower()
    
    if "401" in error_str or "unauthorized" in error_str or "invalid" in error_str:
        return error_str, GroqAuthenticationError
    
    elif "429" in error_str or "rate limit" in error_str:
        return error_str, GroqRateLimitError
    
    elif "503" in error_str or "service unavailable" in error_str or "temporarily unavailable" in error_str:
        return error_str, GroqServiceUnavailableError
    
    elif "timeout" in error_str or "504" in error_str:
        return error_str, GroqTimeoutError
    
    else:
        return error_str, GroqAPIError


def validate_groq_api_key(api_key: str) -> bool:
    """
    Validate Groq API key format.
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if valid format
    
    Raises:
        ValueError: If invalid format
    """
    
    if not api_key:
        raise ValueError("GROQ_API_KEY is empty")
    
    if not api_key.startswith('gsk_'):
        raise ValueError(f"Invalid GROQ_API_KEY format (should start with 'gsk_')")
    
    if len(api_key) < 20:
        raise ValueError(f"GROQ_API_KEY seems too short")
    
    return True
