"""
Rate limiting utilities for Groq API.

Implements token bucket algorithm to manage request and token limits.
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional


class RateLimiter:
    """Simple token bucket rate limiter for requests."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window (Free tier: 30/min)
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        
        # Remove requests outside window
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.window_seconds):
            self.requests.popleft()
        
        return len(self.requests) < self.max_requests
    
    def record_request(self) -> None:
        """Record a request."""
        self.requests.append(datetime.now())
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limit exceeded.
        
        Returns:
            Time waited in seconds
        """
        wait_total = 0.0
        
        while not self.is_allowed():
            oldest = self.requests[0]
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
            
            if wait_time > 0:
                print(f"⏱️  Rate limit: {len(self.requests)}/{self.max_requests} requests. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)
                wait_total += wait_time + 0.1
        
        return wait_total
    
    def get_requests_used(self) -> int:
        """Get number of requests used in current window."""
        now = datetime.now()
        
        # Remove old requests
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.window_seconds):
            self.requests.popleft()
        
        return len(self.requests)
    
    def get_requests_remaining(self) -> int:
        """Get number of requests remaining in current window."""
        return max(0, self.max_requests - self.get_requests_used())
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get when rate limit resets."""
        if not self.requests:
            return None
        
        oldest = self.requests[0]
        reset_time = oldest + timedelta(seconds=self.window_seconds)
        return reset_time


class TokenBucketLimiter:
    """Advanced rate limiter considering both requests and tokens."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 30,
        max_tokens_per_minute: int = 100
    ):
        """
        Initialize token bucket limiter.
        
        Args:
            max_requests_per_minute: Max requests per minute
            max_tokens_per_minute: Max tokens per minute
        """
        self.request_limiter = RateLimiter(
            max_requests=max_requests_per_minute,
            window_seconds=60
        )
        self.token_limiter = RateLimiter(
            max_requests=max_tokens_per_minute,
            window_seconds=60
        )
        self.token_counts = deque()  # (timestamp, token_count)
    
    def can_make_request(self, estimated_tokens: int = 100) -> bool:
        """Check if request can be made."""
        return (
            self.request_limiter.is_allowed() and
            self._can_use_tokens(estimated_tokens)
        )
    
    def _can_use_tokens(self, tokens: int) -> bool:
        """Check if tokens are available."""
        now = datetime.now()
        
        # Remove old token counts
        while self.token_counts and (now - self.token_counts[0][0]) > timedelta(seconds=60):
            self.token_counts.popleft()
        
        total_tokens = sum(count for _, count in self.token_counts)
        return total_tokens + tokens <= 100  # Simplified for free tier
    
    def record_request(self, tokens_used: int) -> None:
        """Record request and tokens used."""
        self.request_limiter.record_request()
        self.token_counts.append((datetime.now(), tokens_used))
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        now = datetime.now()
        
        # Calculate token usage
        total_tokens = 0
        for timestamp, count in self.token_counts:
            if (now - timestamp) <= timedelta(seconds=60):
                total_tokens += count
        
        return {
            "requests_used": self.request_limiter.get_requests_used(),
            "requests_limit": self.request_limiter.max_requests,
            "requests_remaining": self.request_limiter.get_requests_remaining(),
            "tokens_used": total_tokens,
            "tokens_limit": 100,
            "tokens_remaining": max(0, 100 - total_tokens),
        }
