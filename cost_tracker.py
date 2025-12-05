"""
Cost tracking utilities for Groq API usage.

Monitors token usage and estimates costs.
"""

from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict
import json
from pathlib import Path


class CostTracker:
    """Track Groq API usage and estimate costs."""
    
    # Groq pricing (approximate as of Nov 2025)
    COST_PER_1K_TOKENS = 0.0001  # $0.0001 per 1K tokens
    
    def __init__(self, log_file: str = "groq_usage.json"):
        """
        Initialize cost tracker.
        
        Args:
            log_file: File to persist usage data
        """
        self.log_file = log_file
        self.queries: List[Dict] = []
        self.load_history()
    
    def log_query(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model: str = "llama-3.1-8b-instant",
        timestamp: datetime = None
    ) -> float:
        """
        Log a query and return estimated cost.
        
        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            total_tokens: Total tokens
            model: Model used
            timestamp: Query timestamp
        
        Returns:
            Estimated cost in USD
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        cost = (total_tokens / 1000) * self.COST_PER_1K_TOKENS
        
        query = {
            "timestamp": timestamp.isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost
        }
        
        self.queries.append(query)
        self.save_history()
        
        return cost
    
    def save_history(self) -> None:
        """Save usage history to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.queries, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save usage history: {e}")
    
    def load_history(self) -> None:
        """Load usage history from file."""
        try:
            if Path(self.log_file).exists():
                with open(self.log_file, 'r') as f:
                    self.queries = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load usage history: {e}")
    
    def get_total_tokens(self, hours_back: int = 24) -> int:
        """Get total tokens used in last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        total = 0
        for query in self.queries:
            query_time = datetime.fromisoformat(query['timestamp'])
            if query_time > cutoff:
                total += query['total_tokens']
        
        return total
    
    def get_total_cost(self, hours_back: int = 24) -> float:
        """Get total cost for last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        total = 0.0
        for query in self.queries:
            query_time = datetime.fromisoformat(query['timestamp'])
            if query_time > cutoff:
                total += query['cost']
        
        return total
    
    def get_daily_cost(self, days_back: int = 1) -> float:
        """Get cost for last N days."""
        return self.get_total_cost(hours_back=days_back * 24)
    
    def get_monthly_estimate(self) -> float:
        """Estimate monthly cost based on recent activity."""
        # Use last 7 days to estimate
        daily_cost = self.get_daily_cost(days_back=7) / 7
        return daily_cost * 30
    
    def get_query_count(self, hours_back: int = 24) -> int:
        """Get number of queries in last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        count = 0
        for query in self.queries:
            query_time = datetime.fromisoformat(query['timestamp'])
            if query_time > cutoff:
                count += 1
        
        return count
    
    def get_average_tokens_per_query(self, hours_back: int = 24) -> float:
        """Get average tokens per query."""
        count = self.get_query_count(hours_back)
        if count == 0:
            return 0
        
        total = self.get_total_tokens(hours_back)
        return total / count
    
    def print_report(self, hours_back: int = 24) -> None:
        """Print usage report for last N hours."""
        if not self.queries:
            print("📊 No usage data yet")
            return
        
        total_queries = self.get_query_count(hours_back)
        total_tokens = self.get_total_tokens(hours_back)
        total_cost = self.get_total_cost(hours_back)
        avg_tokens = self.get_average_tokens_per_query(hours_back)
        monthly_estimate = self.get_monthly_estimate()
        
        print(f"\n📊 Groq API Usage Report (Last {hours_back}h)")
        print(f"   Queries: {total_queries}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Avg tokens/query: {avg_tokens:.0f}")
        print(f"   Cost: ${total_cost:.6f}")
        print(f"   Est. monthly: ${monthly_estimate:.2f}")
    
    def print_daily_report(self) -> None:
        """Print daily usage report."""
        self.print_report(hours_back=24)
    
    def print_session_report(self, session_start: datetime) -> None:
        """Print session usage report."""
        session_queries = [
            q for q in self.queries
            if datetime.fromisoformat(q['timestamp']) > session_start
        ]
        
        if not session_queries:
            print("📊 No queries in this session")
            return
        
        total_tokens = sum(q['total_tokens'] for q in session_queries)
        total_cost = sum(q['cost'] for q in session_queries)
        
        print(f"\n📊 Session Summary")
        print(f"   Queries: {len(session_queries)}")
        print(f"   Tokens: {total_tokens:,}")
        print(f"   Cost: ${total_cost:.6f}")
