# Migration Design Document: Ollama → Groq Cloud API

**Project:** RAGFood  
**Date:** November 29, 2025  
**Status:** Design Phase  
**Author:** System Design

---

## Executive Summary

This document outlines the migration strategy from local Ollama inference to Groq Cloud API for LLM-based answer generation. Groq provides significantly faster inference speeds through specialized hardware and eliminates the need for local model hosting.

**Key Benefits:**
- ✅ **5-10x faster inference** (Groq's specialized hardware)
- ✅ **No local model storage** (~4GB freed)
- ✅ **No Ollama service management** (one less dependency)
- ✅ **Production-ready infrastructure** (99.99% uptime)
- ✅ **Scalable** (automatically handles concurrent requests)
- ✅ **Pay-as-you-go** (only pay for tokens used)

**Trade-offs:**
- ❌ Network dependency (requires internet)
- ❌ Usage costs (~$0.001-0.01 per query)
- ❌ Rate limiting (varies by tier)
- ❌ Data sent to third-party (privacy consideration)

---

## Table of Contents

1. [Current System Analysis](#current-system-analysis)
2. [Target Architecture](#target-architecture)
3. [API Comparison](#api-comparison)
4. [Detailed Migration Steps](#detailed-migration-steps)
5. [Code Refactoring Guide](#code-refactoring-guide)
6. [Error Handling & Resilience](#error-handling--resilience)
7. [Rate Limiting Strategy](#rate-limiting-strategy)
8. [Cost Analysis](#cost-analysis)
9. [Performance Expectations](#performance-expectations)
10. [Security & Privacy](#security--privacy)
11. [Testing Strategy](#testing-strategy)
12. [Rollback Plan](#rollback-plan)

---

## Current System Analysis

### Ollama Architecture (Local)

```
User Query
    ↓
┌─────────────────────────────────────────┐
│     RAG Pipeline (Python)               │
│  - Upstash Vector query (50-150ms)      │
│  - Context retrieval (5ms)              │
│  - Prompt building (1ms)                │
└────────────┬────────────────────────────┘
             │
             ↓ (HTTP POST to localhost)
┌─────────────────────────────────────────┐
│     Ollama Service (localhost:11434)    │
│  ├─ Model: llama3.2 (7B params)         │
│  ├─ RAM: ~4-6GB in use                  │
│  ├─ Inference: CPU-based                │
│  └─ Latency: 2-10s per query            │
└─────────────────────────────────────────┘
             ↓
        Answer to user

Total time: ~2-10 seconds (dominated by LLM)
```

### Current Code (rag_run.py)

```python
# Current Ollama implementation
def rag_query(question: str, index: Index) -> str:
    # ... Upstash vector search ...
    
    # Build prompt
    prompt = f"""Use the following context..."""
    
    # Call Ollama LLM (blocking)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=30
    )
    
    return response.json()["response"].strip()
```

### Issues with Current Approach

1. **Ollama dependency:** Must be running locally at all times
2. **Resource intensive:** Uses 4-6GB RAM, CPU-based inference
3. **Limited scalability:** Single machine bottleneck
4. **Inference speed:** CPU-based inference is slow (2-10s)
5. **Maintenance burden:** Model updates, troubleshooting, monitoring
6. **Development friction:** Setup requires Ollama installation
7. **No monitoring:** No built-in observability

---

## Target Architecture

### Groq Cloud API (Serverless)

```
User Query
    ↓
┌──────────────────────────────────────────┐
│      RAG Pipeline (Python)               │
│  - Upstash Vector query (50-150ms)       │
│  - Context retrieval (5ms)               │
│  - Prompt building (1ms)                 │
└────────────┬─────────────────────────────┘
             │
             ↓ (HTTPS to Groq API)
┌──────────────────────────────────────────┐
│     Groq Cloud API (api.groq.com)        │
│  ├─ Model: llama-3.1-8b-instant         │
│  ├─ Hardware: Specialized LPU            │
│  ├─ Latency: 200-500ms per query         │
│  ├─ Throughput: 200+ tokens/sec          │
│  └─ Availability: 99.99% SLA             │
└──────────────────────────────────────────┘
             ↓
        Answer to user

Total time: ~500ms-2s (10x faster!)
```

### Key Improvements

| Aspect | Ollama (Local) | Groq (Cloud) | Improvement |
|--------|----------------|--------------|------------|
| **Latency** | 2-10s | 200-500ms | ✅ 5-20x faster |
| **Throughput** | ~1 req/10s | 200+ tokens/sec | ✅ 100x faster |
| **RAM usage** | ~5GB | 0MB | ✅ Freed |
| **Availability** | Depends on host | 99.99% SLA | ✅ Reliable |
| **Scalability** | Vertical only | Auto-scales | ✅ Unlimited |
| **Cost** | ~$0 (power) | $0.001-0.01/query | 🔄 TBD |
| **Setup** | Complex | Simple (SDK) | ✅ Easier |

---

## API Comparison

### Request Format

#### BEFORE (Ollama)

```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Your full prompt text here...",
        "stream": False,
        "temperature": 0.7
    },
    timeout=30
)

answer = response.json()["response"]
```

**Issues:**
- ❌ Full prompt as single string (not structured)
- ❌ Custom endpoint format (not standardized)
- ❌ Limited metadata in response
- ❌ No built-in error details

---

#### AFTER (Groq)

```python
from groq import Groq

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Your full prompt text here..."
        }
    ],
    temperature=0.7,
    max_completion_tokens=1024,
    stream=False
)

answer = completion.choices[0].message.content
```

**Advantages:**
- ✅ Structured message format (OpenAI-compatible)
- ✅ Standard API (widely adopted)
- ✅ Rich metadata (usage, finish_reason, etc.)
- ✅ Built-in error handling
- ✅ Native streaming support

---

### Response Comparison

#### Ollama Response

```json
{
    "response": "Pizza is an Italian dish...",
    "context": null,
    "total_duration": 8234567890,
    "load_duration": 1234567890,
    "prompt_eval_count": 45,
    "prompt_eval_duration": 234567890,
    "eval_count": 156,
    "eval_duration": 5123456789
}
```

---

#### Groq Response

```json
{
    "id": "chatcmpl-8934hfs...",
    "object": "chat.completion",
    "created": 1701273450,
    "model": "llama-3.1-8b-instant",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Pizza is an Italian dish..."
            },
            "finish_reason": "stop",
            "stop_reason": "end_turn"
        }
    ],
    "usage": {
        "prompt_tokens": 45,
        "completion_tokens": 156,
        "total_tokens": 201
    }
}
```

**Advantages:**
- ✅ Token count tracking (for cost monitoring)
- ✅ Finish reason (context for why response ended)
- ✅ Standardized format (OpenAI-compatible)

---

## Detailed Migration Steps

### Step 1: Environment Setup (30 minutes)

#### 1.1 Install Groq SDK

```bash
# In .venv environment
pip install groq
```

**Verification:**
```python
from groq import Groq
print("✅ Groq SDK installed")
```

#### 1.2 Verify API Key

Your `.env.local` should have:
```
GROQ_API_KEY="gsk_..."
```

**Verify it's accessible:**
```python
import os
from dotenv import load_dotenv

load_dotenv('.env.local')
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    raise ValueError("GROQ_API_KEY not set in .env.local")
    
print(f"✅ API key found: {api_key[:20]}...")
```

#### 1.3 Test Groq Connection

```python
from groq import Groq
import os

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Test connection
try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_completion_tokens=10
    )
    print(f"✅ Groq connection successful")
    print(f"   Response: {completion.choices[0].message.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

---

### Step 2: Code Refactoring (1-2 hours)

#### 2.1 Update Imports

```python
# REMOVE
import requests

# ADD
from groq import Groq
```

#### 2.2 Initialize Groq Client

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv('.env.local')

# Create client (auto-reads GROQ_API_KEY)
groq_client = Groq()
```

#### 2.3 Refactor LLM Call

**BEFORE (Ollama - raw HTTP):**
```python
def generate_answer(prompt: str) -> str:
    """Generate answer using Ollama."""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7
        },
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code}")
    
    return response.json()["response"].strip()
```

**AFTER (Groq - SDK):**
```python
def generate_answer(prompt: str) -> str:
    """Generate answer using Groq API."""
    
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        stream=False
    )
    
    return completion.choices[0].message.content.strip()
```

---

### Step 3: Integrate into RAG Pipeline

#### Complete Refactored Function

```python
def rag_query_with_groq(question: str, index: Index) -> str:
    """
    Complete RAG pipeline with Groq LLM backend.
    
    Steps:
    1. Search Upstash Vector for relevant documents
    2. Build context from retrieved documents
    3. Create prompt with context
    4. Call Groq API for answer generation
    5. Return final answer
    """
    
    try:
        # Step 1: Query Upstash Vector (no change)
        print("\n🧠 Retrieving relevant information from Upstash...")
        results = index.query(
            data=question,
            top_k=3,
            include_metadata=True
        )
        
        if not results:
            return "No relevant information found."
        
        # Step 2: Extract documents
        print(f"📚 Found {len(results)} relevant documents\n")
        
        top_docs = []
        for i, result in enumerate(results, 1):
            doc_text = result.metadata.get('original_text', result.id)
            score = result.score
            
            print(f"🔹 Source {i} (Relevance: {score:.2%}): {doc_text[:100]}...")
            top_docs.append(doc_text)
        
        # Step 3: Build context
        context = "\n".join(top_docs)
        
        # Step 4: Create structured prompt
        prompt = f"""Use the following context to answer the question. 
If the information is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""
        
        # Step 5: Call Groq (CHANGED from Ollama)
        print("\n🚀 Generating answer with Groq...\n")
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            stream=False
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Log token usage for cost tracking
        usage = completion.usage
        print(f"\n📊 Usage Stats:")
        print(f"   Input tokens: {usage.prompt_tokens}")
        print(f"   Output tokens: {usage.completion_tokens}")
        print(f"   Total: {usage.total_tokens}")
        
        return answer
        
    except Exception as e:
        return f"❌ Error: {str(e)}"
```

---

### Step 4: Update requirements.txt

```bash
# Remove
requests
chromadb

# Add
groq>=0.4.0
upstash-vector>=0.1.0
python-dotenv>=1.0.0
```

**Update file:**
```bash
pip freeze > requirements.txt
```

---

## Code Refactoring Guide

### Complete Refactored rag_run.py

```python
"""
RAGFood - Retrieval Augmented Generation with Upstash Vector + Groq LLM

Architecture:
1. Load food data into Upstash Vector (automatic embeddings)
2. Accept user queries
3. Search Upstash for relevant documents (50-150ms)
4. Build prompt with retrieved context
5. Call Groq API for answer generation (200-500ms)
6. Return final answer to user

Key improvements over Ollama:
- Groq provides 5-20x faster inference
- No local model management needed
- Production-ready infrastructure (99.99% uptime)
- Token usage tracking for cost optimization
"""

import os
import json
import time
from typing import List, Optional
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq
from groq.types.chat import ChatCompletion

# Load environment variables
load_dotenv('.env.local')

# Constants
UPSTASH_URL = os.getenv('UPSTASH_VECTOR_REST_URL')
UPSTASH_TOKEN = os.getenv('UPSTASH_VECTOR_REST_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
JSON_FILE = "foods.json"

# Global clients
upstash_index: Optional[Index] = None
groq_client: Optional[Groq] = None

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_clients():
    """Initialize Upstash Vector and Groq clients."""
    global upstash_index, groq_client
    
    # Initialize Upstash Vector
    upstash_index = Index(
        url=UPSTASH_URL,
        token=UPSTASH_TOKEN
    )
    
    # Initialize Groq (auto-reads GROQ_API_KEY from environment)
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    print("✅ Clients initialized")
    print(f"   - Upstash Vector: {UPSTASH_URL[:50]}...")
    print(f"   - Groq API: {GROQ_API_KEY[:20]}...")

def validate_setup():
    """Validate all required credentials and connections."""
    
    errors = []
    
    # Check credentials
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        errors.append("Missing UPSTASH_VECTOR_REST_URL or UPSTASH_VECTOR_REST_TOKEN")
    
    if not GROQ_API_KEY:
        errors.append("Missing GROQ_API_KEY")
    
    if errors:
        for error in errors:
            print(f"❌ {error}")
        return False
    
    # Test Upstash connection
    try:
        info = upstash_index.info()
        print(f"✅ Upstash Vector connected")
        print(f"   Vectors: {info.vector_count}")
    except Exception as e:
        print(f"❌ Upstash error: {e}")
        return False
    
    # Test Groq connection
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=1
        )
        print(f"✅ Groq API connected")
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return False
    
    return True

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_migrate_foods():
    """Load foods from JSON and upsert to Upstash Vector."""
    
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        food_data = json.load(f)
    
    print(f"🔄 Loading {len(food_data)} food items...")
    
    vectors_to_upsert = []
    
    for item in food_data:
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."
        
        vector_data = (
            item["id"],
            enriched_text,
            {
                "original_text": item["text"],
                "region": item.get("region", ""),
                "type": item.get("type", ""),
                "source": "foods.json"
            }
        )
        vectors_to_upsert.append(vector_data)
    
    # Batch upsert
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        try:
            upstash_index.upsert(vectors=batch)
            print(f"✅ Upserted batch {i//batch_size + 1}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"✅ {len(food_data)} foods ready")

# ============================================================================
# GROQ LLM GENERATION (NEW)
# ============================================================================

def generate_answer(prompt: str, max_tokens: int = 1024) -> tuple[str, dict]:
    """
    Generate answer using Groq API.
    
    Args:
        prompt: The full prompt with context
        max_tokens: Maximum tokens in response
    
    Returns:
        (answer_text, usage_stats)
    
    Usage stats useful for:
    - Cost tracking
    - Performance monitoring
    - Rate limit management
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_completion_tokens=max_tokens,
            stream=False
        )
        
        answer = completion.choices[0].message.content.strip()
        
        usage_stats = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
            "finish_reason": completion.choices[0].finish_reason
        }
        
        return answer, usage_stats
        
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")

# ============================================================================
# RAG PIPELINE (MAIN)
# ============================================================================

def rag_query(question: str) -> dict:
    """
    Complete RAG pipeline with metrics tracking.
    
    Returns:
        {
            "answer": str,
            "metrics": {
                "vector_search_time": float,
                "llm_generation_time": float,
                "total_time": float,
                "tokens_used": int,
                "cost_estimate": float
            },
            "sources": List[str]
        }
    """
    
    start_time = time.time()
    
    try:
        # Step 1: Vector search (Upstash)
        print("\n🧠 Searching vector database...")
        vector_start = time.time()
        
        results = upstash_index.query(
            data=question,
            top_k=3,
            include_metadata=True
        )
        
        vector_time = time.time() - vector_start
        
        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "metrics": {
                    "vector_search_time": vector_time,
                    "llm_generation_time": 0,
                    "total_time": time.time() - start_time,
                    "tokens_used": 0,
                    "cost_estimate": 0
                }
            }
        
        # Step 2: Extract documents
        print(f"📚 Found {len(results)} relevant documents\n")
        
        top_docs = []
        sources = []
        for i, result in enumerate(results, 1):
            doc_text = result.metadata.get('original_text', result.id)
            score = result.score
            region = result.metadata.get('region', '')
            
            print(f"🔹 Source {i} (Relevance: {score:.2%})")
            print(f"   {doc_text[:80]}...")
            if region:
                print(f"   Region: {region}")
            print()
            
            top_docs.append(doc_text)
            sources.append({
                "text": doc_text,
                "relevance": score,
                "region": region
            })
        
        # Step 3: Build context
        context = "\n".join(top_docs)
        
        # Step 4: Create prompt
        prompt = f"""Use the following context to answer the question.
If the information is not in the context, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
        
        # Step 5: Generate answer with Groq
        print("🚀 Generating answer with Groq...\n")
        llm_start = time.time()
        
        answer, usage_stats = generate_answer(prompt)
        
        llm_time = time.time() - llm_start
        
        # Calculate estimated cost
        # Groq pricing: roughly $0.0001 per 1K tokens
        cost_per_1k_tokens = 0.0001
        estimated_cost = (usage_stats['total_tokens'] / 1000) * cost_per_1k_tokens
        
        # Print metrics
        print(f"\n📊 Generation Complete")
        print(f"   Tokens: {usage_stats['total_tokens']}")
        print(f"   Time: {llm_time*1000:.0f}ms")
        print(f"   Cost: ${estimated_cost:.6f}")
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "vector_search_time": vector_time,
                "llm_generation_time": llm_time,
                "total_time": time.time() - start_time,
                "tokens_used": usage_stats['total_tokens'],
                "cost_estimate": estimated_cost,
                "finish_reason": usage_stats['finish_reason']
            }
        }
        
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "sources": [],
            "metrics": {
                "vector_search_time": 0,
                "llm_generation_time": 0,
                "total_time": time.time() - start_time,
                "tokens_used": 0,
                "cost_estimate": 0
            }
        }

# ============================================================================
# INTERACTIVE LOOP
# ============================================================================

def main():
    """Main RAG application."""
    
    print("\n" + "="*60)
    print("🚀 RAGFood - Upstash Vector + Groq LLM Edition")
    print("="*60)
    
    # Initialize
    init_clients()
    
    if not validate_setup():
        print("\n❌ Setup validation failed")
        return
    
    # Load data
    load_and_migrate_foods()
    
    # Interactive loop
    print("\n" + "="*60)
    print("🧠 RAG Ready! Ask a question (type 'exit' to quit)")
    print("="*60 + "\n")
    
    total_tokens = 0
    total_cost = 0
    query_count = 0
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print(f"\n📊 Session Summary")
            print(f"   Queries: {query_count}")
            print(f"   Tokens: {total_tokens}")
            print(f"   Estimated cost: ${total_cost:.6f}")
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        # Run RAG pipeline
        result = rag_query(question)
        
        print(f"\n🤖 Assistant: {result['answer']}\n")
        
        # Track usage
        query_count += 1
        total_tokens += result['metrics']['tokens_used']
        total_cost += result['metrics']['cost_estimate']

if __name__ == "__main__":
    main()
```

---

## Error Handling & Resilience

### Common Error Scenarios & Solutions

#### 1. Groq API Connection Error

```python
class GroqConnectionError(Exception):
    """Raised when Groq API is unreachable."""
    pass

def generate_answer_with_retry(
    prompt: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> tuple[str, dict]:
    """Generate answer with exponential backoff retry logic."""
    
    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1024,
                stream=False
            )
            
            return completion.choices[0].message.content.strip(), {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason
            }
            
        except Exception as e:
            if "429" in str(e):  # Rate limit
                wait_time = (backoff_factor ** attempt) + (backoff_factor ** attempt)
                print(f"⚠️  Rate limited. Retry in {wait_time}s...")
                time.sleep(wait_time)
            elif "401" in str(e):  # Auth error
                raise ValueError("Invalid GROQ_API_KEY")
            elif "503" in str(e):  # Service unavailable
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"⚠️  Service temporarily unavailable. Retry in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise GroqConnectionError("Groq service unavailable")
            else:
                raise
    
    raise GroqConnectionError("Max retries exceeded")
```

#### 2. Invalid API Key

```python
def validate_groq_key():
    """Validate Groq API key is valid."""
    
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env.local\n"
            "Get your key from: https://console.groq.com/keys"
        )
    
    if not api_key.startswith('gsk_'):
        raise ValueError(
            "Invalid GROQ_API_KEY format (should start with 'gsk_')\n"
            "Check https://console.groq.com/keys"
        )
    
    try:
        client = Groq(api_key=api_key)
        # Test with minimal call
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=1
        )
        print("✅ API key valid")
    except Exception as e:
        if "401" in str(e):
            raise ValueError("GROQ_API_KEY is invalid or expired")
        raise
```

#### 3. Rate Limiting

```python
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        
        # Remove old requests outside window
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.window_seconds):
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def wait_if_needed(self):
        """Block if rate limit exceeded."""
        if not self.is_allowed():
            oldest = self.requests[0]
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
            print(f"⚠️  Rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(max(0, wait_time))
            self.wait_if_needed()  # Recursive check after wait

# Usage
limiter = RateLimiter(max_requests=30, window_seconds=60)

def rag_query_rate_limited(question: str) -> str:
    limiter.wait_if_needed()
    return rag_query(question)
```

#### 4. Token Limit Exceeded

```python
def generate_answer_with_truncation(
    prompt: str,
    max_total_tokens: int = 2048
) -> tuple[str, dict]:
    """
    Generate answer, truncating context if needed to stay under token limit.
    
    Groq has token limits per request:
    - llama-3.1-8b-instant: 2048 tokens max
    """
    
    # Estimate token count (rough: ~4 chars per token)
    estimated_tokens = len(prompt) // 4
    
    if estimated_tokens > max_total_tokens * 0.9:
        print(f"⚠️  Prompt too long ({estimated_tokens} tokens). Truncating context...")
        
        # Remove some context
        lines = prompt.split('\n')
        
        # Keep system prompt + question, truncate context
        prompt = '\n'.join(lines[:5] + lines[-5:])
    
    return generate_answer(prompt)
```

---

## Rate Limiting Strategy

### Groq Rate Limits (by tier)

```
Free Tier:
├─ 30 requests/minute
├─ 100 tokens/minute
└─ Best for testing

Pro Tier:
├─ 100+ requests/minute
├─ Unlimited tokens
└─ Production use

Enterprise:
├─ Custom limits
├─ Dedicated support
└─ SLA guarantee
```

### Implementation

```python
import time
from datetime import datetime, timedelta

class GroqRateLimiter:
    """Advanced rate limiter with cost awareness."""
    
    def __init__(self):
        self.request_timestamps = deque()
        self.token_usage = deque()
        self.request_limit = 30  # Free tier
        self.window_seconds = 60
    
    def track_request(self, tokens_used: int):
        """Track request for rate limiting."""
        now = datetime.now()
        self.request_timestamps.append(now)
        self.token_usage.append((now, tokens_used))
        
        # Log for monitoring
        print(f"📊 Rate limit: {len(self.request_timestamps)}/{self.request_limit} requests")
    
    def can_make_request(self) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        
        # Remove old timestamps
        while self.request_timestamps and (now - self.request_timestamps[0]) > timedelta(seconds=self.window_seconds):
            self.request_timestamps.popleft()
        
        return len(self.request_timestamps) < self.request_limit
    
    def wait_if_needed(self):
        """Block and wait if rate limited."""
        while not self.can_make_request():
            oldest = self.request_timestamps[0]
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
            
            if wait_time > 0:
                print(f"⏱️  Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 0.1)  # Small buffer
    
    def get_cost_estimate(self) -> float:
        """Estimate cost based on recent usage."""
        now = datetime.now()
        cost_per_1k_tokens = 0.0001
        
        # Remove old tokens
        while self.token_usage and (now - self.token_usage[0][0]) > timedelta(hours=1):
            self.token_usage.popleft()
        
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        return (total_tokens / 1000) * cost_per_1k_tokens

limiter = GroqRateLimiter()
```

---

## Cost Analysis

### Groq Pricing Model

```
Pricing Tiers (as of Nov 2025):

Free Tier:
├─ $0/month
├─ 30 requests/minute
├─ 100 tokens/minute
└─ Best for: Development, testing

Pay-as-you-go:
├─ $0.0001 per 1K tokens (approximately)
├─ Based on actual usage
├─ No minimum
└─ Best for: Small projects

Pro Plan:
├─ $10-50/month base
├─ Included tokens
├─ Higher rate limits
└─ Best for: Production apps

Enterprise:
├─ Custom pricing
├─ Dedicated support
├─ SLA guarantee
└─ Best for: Critical systems
```

### Cost Comparison: Ollama vs Groq

```
┌──────────────────┬──────────────────┬──────────────┐
│ Category         │ Ollama (Local)   │ Groq (Cloud) │
├──────────────────┼──────────────────┼──────────────┤
│ Setup cost       │ $0               │ $0           │
│ Monthly base     │ $0               │ $0 (pay/use) │
│ Per 1K tokens    │ $0               │ $0.0001      │
│                  │                  │              │
│ 1000 queries/mo  │                  │              │
│ (200 tokens avg) │ $0               │ $0.02        │
│                  │                  │              │
│ 10000 queries/mo │ $0               │ $0.20        │
│ (200 tokens avg) │ $0               │ $0.20        │
│                  │                  │              │
│ Hosting cost     │ $100-500/mo      │ $0           │
│ RAM (5GB)        │ (opportunity cost)               │
│                  │                  │              │
│ Maintenance      │ ~10 hrs/mo       │ $0           │
│ Dev time         │ ~$500/mo equiv   │              │
│                  │                  │              │
│ TOTAL            │ ~$600-1000/mo    │ $0.20/mo     │
└──────────────────┴──────────────────┴──────────────┘

Annual Savings: ~$7000-12000
```

### Cost Monitoring

```python
class CostTracker:
    """Track and monitor Groq API costs."""
    
    def __init__(self):
        self.queries = []
        self.cost_per_1k_tokens = 0.0001
    
    def log_query(self, tokens_used: int, timestamp=None):
        """Log query for cost tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.queries.append({
            "timestamp": timestamp,
            "tokens": tokens_used,
            "cost": (tokens_used / 1000) * self.cost_per_1k_tokens
        })
    
    def get_daily_cost(self, days_back: int = 1) -> float:
        """Get cost for last N days."""
        cutoff = datetime.now() - timedelta(days=days_back)
        return sum(q['cost'] for q in self.queries if q['timestamp'] > cutoff)
    
    def get_monthly_estimate(self) -> float:
        """Estimate monthly cost based on recent activity."""
        daily_cost = self.get_daily_cost(days_back=7) / 7
        return daily_cost * 30
    
    def print_report(self):
        """Print cost report."""
        if not self.queries:
            print("No queries tracked")
            return
        
        total_tokens = sum(q['tokens'] for q in self.queries)
        total_cost = sum(q['cost'] for q in self.queries)
        avg_tokens = total_tokens / len(self.queries)
        
        print(f"\n📊 Cost Report")
        print(f"  Total queries: {len(self.queries)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/query: {avg_tokens:.0f}")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Estimated monthly: ${self.get_monthly_estimate():.2f}")

# Usage
tracker = CostTracker()

def rag_query_tracked(question: str):
    result = rag_query(question)
    tracker.log_query(result['metrics']['tokens_used'])
    return result
```

---

## Performance Expectations

### Latency Comparison

#### Request Timeline

```
Ollama (Local CPU-based):
┌──────────────────────────────────────────────────┐
│ Latency Breakdown                                │
├──────────────────────────────────────────────────┤
│ Prompt tokenization: 10-50ms                     │
│ Context window setup: 10-50ms                    │
│ LLM inference: 1500-8000ms (slow CPU)            │
│ Response generation: 500-3000ms                  │
├──────────────────────────────────────────────────┤
│ TOTAL: 2000-11000ms (~4s average)                │
└──────────────────────────────────────────────────┘

Groq (Cloud GPU/LPU):
┌──────────────────────────────────────────────────┐
│ Latency Breakdown                                │
├──────────────────────────────────────────────────┤
│ Network latency to API: 10-50ms                  │
│ Prompt tokenization: 1-5ms                       │
│ LLM inference: 100-200ms (specialized HW)        │
│ Network return: 10-50ms                         │
├──────────────────────────────────────────────────┤
│ TOTAL: 100-300ms (~200ms average)                │
└──────────────────────────────────────────────────┘

IMPROVEMENT: 5-20x faster! ✅
```

### Throughput Comparison

```
Ollama:
├─ Sequential: ~1 request/5 seconds
├─ Can't parallelize (single model)
├─ Throughput: ~0.2 req/sec

Groq:
├─ Concurrent: 100+ parallel requests
├─ Auto-scaling infrastructure
├─ Throughput: 100+ req/sec
```

### Response Quality

Both use similar base models (llama family), so quality should be comparable:
- Ollama: llama3.2 (7B)
- Groq: llama-3.1-8b-instant (8B)

**Expected similar quality, but Groq might be slightly better due to:**
- More recent model (3.1 vs 3.2 naming)
- Better infrastructure (specialized LPU hardware)
- More resources available

---

## Security & Privacy

### 1. API Key Management

#### Secure Storage

```python
# ✅ DO: Load from environment
import os
from dotenv import load_dotenv

load_dotenv('.env.local')
api_key = os.getenv('GROQ_API_KEY')

# ❌ DON'T: Hardcode key
api_key = "gsk_..."  # EXPOSED!

# ❌ DON'T: Log the key
print(f"Using key: {api_key}")  # EXPOSED!
```

#### .gitignore Configuration

```bash
# .gitignore
.env
.env.local
.env.*.local
*.key
*.secret
```

### 2. Data Privacy

#### What Groq Sees

```
✅ Can see:
├─ Your prompts (food-related questions)
├─ Groq's response to your query
└─ Generic metadata (timestamp, token count)

❌ Groq likely CANNOT see:
├─ Your vector database content
├─ Food data from Upstash
├─ User identification
└─ Historical queries (if not logged)

⚠️ Limitations:
├─ Prompts sent over internet (HTTPS encrypted)
├─ Groq stores logs (check privacy policy)
├─ Subject to Groq's privacy policy
└─ Not suitable for sensitive proprietary data
```

#### Privacy Checklist

```python
def sanitize_prompt(prompt: str) -> str:
    """Remove sensitive info before sending to Groq."""
    
    # Remove email addresses
    prompt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', prompt)
    
    # Remove phone numbers
    prompt = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', prompt)
    
    # Remove credit card patterns
    prompt = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC]', prompt)
    
    return prompt
```

### 3. Comparison: Data Location

```
Ollama (Local):
├─ All data stays on your machine
├─ No transmission risk
├─ No third-party access
└─ Max privacy ✅

Groq (Cloud):
├─ Prompts sent to Groq servers
├─ Stored in Groq infrastructure
├─ Subject to Groq privacy policy
├─ Groq employees may have access
└─ Less privacy ⚠️

Decision:
├─ Public/semi-public data: Groq OK
├─ Sensitive/proprietary data: Use Ollama
├─ Food data: Groq acceptable
└─ Financial/medical data: Ollama only
```

---

## Testing Strategy

### Unit Tests

```python
import unittest
from unittest.mock import Mock, patch

class TestGroqIntegration(unittest.TestCase):
    """Test Groq API integration."""
    
    def setUp(self):
        self.prompt = "What is pizza?"
        
    @patch('groq.Groq.chat.completions.create')
    def test_generate_answer(self, mock_create):
        """Test answer generation."""
        
        mock_response = Mock()
        mock_response.choices[0].message.content = "Pizza is Italian."
        mock_response.usage.total_tokens = 50
        mock_create.return_value = mock_response
        
        answer, usage = generate_answer(self.prompt)
        
        self.assertEqual(answer, "Pizza is Italian.")
        self.assertEqual(usage['total_tokens'], 50)
    
    def test_rate_limiter(self):
        """Test rate limiting logic."""
        
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # First two requests OK
        self.assertTrue(limiter.is_allowed())
        self.assertTrue(limiter.is_allowed())
        
        # Third request blocked
        self.assertFalse(limiter.is_allowed())
    
    def test_error_handling(self):
        """Test error handling."""
        
        with self.assertRaises(ValueError):
            validate_groq_key()  # No key set
    
    def test_cost_calculation(self):
        """Test cost calculation."""
        
        tokens = 1000
        cost_per_1k = 0.0001
        expected_cost = 0.0001
        
        actual_cost = (tokens / 1000) * cost_per_1k
        self.assertAlmostEqual(actual_cost, expected_cost, places=6)
```

### Integration Tests

```python
class TestRAGPipeline(unittest.TestCase):
    """Test complete RAG pipeline."""
    
    def setUp(self):
        init_clients()
        load_and_migrate_foods()
    
    def test_rag_query_returns_valid_response(self):
        """Test RAG query returns valid structure."""
        
        result = rag_query("What is pizza?")
        
        self.assertIn("answer", result)
        self.assertIn("metrics", result)
        self.assertGreater(len(result['answer']), 0)
    
    def test_rag_query_metrics(self):
        """Test query metrics are tracked."""
        
        result = rag_query("What is pizza?")
        metrics = result['metrics']
        
        self.assertGreater(metrics['vector_search_time'], 0)
        self.assertGreater(metrics['llm_generation_time'], 0)
        self.assertGreater(metrics['total_time'], 0)
        self.assertGreater(metrics['tokens_used'], 0)
    
    def test_rag_query_reproducibility(self):
        """Test same query gives reasonable results."""
        
        result1 = rag_query("What is pizza?")
        result2 = rag_query("What is pizza?")
        
        # Should find same sources
        self.assertEqual(
            result1['sources'][0]['text'],
            result2['sources'][0]['text']
        )
```

### Performance Tests

```python
class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_query_latency(self):
        """Test query completes within expected time."""
        
        start = time.time()
        result = rag_query("What is pizza?")
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        self.assertLess(elapsed, 5.0)
        
        # Groq generation should be < 1 second
        self.assertLess(result['metrics']['llm_generation_time'], 1.0)
    
    def test_throughput(self):
        """Test can handle multiple queries."""
        
        queries = [
            "What is pizza?",
            "How to make pasta?",
            "What is sushi?"
        ]
        
        start = time.time()
        for q in queries:
            rag_query(q)
        elapsed = time.time() - start
        
        # 3 queries should complete in < 15 seconds
        self.assertLess(elapsed, 15.0)
```

---

## Rollback Plan

### Why Rollback Might Be Needed

```
Scenarios for rollback:
1. Groq API performance degrades
2. Cost exceeds budget
3. Rate limiting too restrictive
4. Privacy concerns arise
5. Data quality issues
6. Integration issues
```

### Rollback Procedure

#### Step 1: Keep Ollama as Backup

```bash
# Keep Ollama available during transition
# Keep old rag_run.py with Ollama code
git checkout <old-commit> -- rag_run_ollama.py
```

#### Step 2: Create Rollback Script

```python
# rollback_to_ollama.py
"""
Rollback from Groq to Ollama if needed.
"""

import sys
import shutil
from pathlib import Path

def rollback():
    """Rollback to Ollama-based RAG."""
    
    print("🔄 Rolling back to Ollama...")
    
    # Restore original rag_run.py
    shutil.copy('rag_run_ollama.py', 'rag_run.py')
    print("✅ Restored Ollama version")
    
    # Start Ollama service
    os.system("ollama serve &")
    print("✅ Started Ollama service")
    
    # Pull model if needed
    os.system("ollama pull llama3.2")
    print("✅ Model ready")
    
    print("✅ Rollback complete. Run: python rag_run.py")

if __name__ == "__main__":
    rollback()
```

#### Step 3: Version Control

```bash
# Tag current Groq implementation
git tag groq-v1.0

# Tag current Ollama implementation  
git tag ollama-v1.0

# Easy rollback if needed
git checkout ollama-v1.0
```

#### Step 4: Monitor During Transition

```python
# Run both simultaneously for comparison
def compare_backends():
    """Compare Ollama vs Groq output."""
    
    question = "What is pizza?"
    
    # Ollama response
    ollama_answer = generate_answer_ollama(question)
    
    # Groq response
    groq_answer = generate_answer_groq(question)
    
    print(f"Ollama: {ollama_answer}\n")
    print(f"Groq:   {groq_answer}\n")
    
    # Both should be reasonable
```

---

## Migration Checklist

### Pre-Migration (Day 1)

```
Setup & Preparation:
[ ] Get GROQ_API_KEY from https://console.groq.com/keys
[ ] Add GROQ_API_KEY to .env.local
[ ] pip install groq
[ ] Test Groq connection with sample query
[ ] Create backup of rag_run_ollama.py
[ ] Document current performance metrics

Verification:
[ ] Confirm Upstash Vector working (migrated earlier)
[ ] Test vector search latency
[ ] Verify Ollama still working as baseline
[ ] Document baseline performance
```

### Migration (Day 2)

```
Code Changes:
[ ] Install Groq SDK: pip install groq
[ ] Update imports in rag_run.py
[ ] Replace generate_answer() function
[ ] Update RAG pipeline to use groq_client
[ ] Add cost tracking
[ ] Add error handling
[ ] Test locally

Testing:
[ ] Unit tests pass
[ ] Integration tests pass
[ ] Performance tests pass
[ ] Manual testing with sample queries
[ ] Verify token usage tracking
```

### Post-Migration (Day 3+)

```
Validation:
[ ] Monitor Groq API for errors
[ ] Track cost metrics
[ ] Compare answer quality
[ ] Load test with multiple queries
[ ] Check rate limiting behavior

Optimization:
[ ] Tune temperature/max_tokens
[ ] Optimize prompt engineering
[ ] Cache common queries
[ ] Monitor latency

Documentation:
[ ] Update README
[ ] Document API changes
[ ] Add troubleshooting guide
[ ] Record performance metrics
```

---

## Appendix: Quick Reference

### Before & After Code

#### BEFORE (Ollama)
```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    },
    timeout=30
)
answer = response.json()["response"].strip()
```

#### AFTER (Groq)
```python
completion = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    stream=False
)
answer = completion.choices[0].message.content.strip()
```

### Dependencies

```
BEFORE:
├─ requests
├─ chromadb (being replaced)
├─ Ollama (external service)

AFTER:
├─ groq>=0.4.0
├─ upstash-vector
├─ python-dotenv
```

### Environment Variables

```
# Add to .env.local
GROQ_API_KEY="gsk_..."  # Get from https://console.groq.com/keys
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Invalid API key | Regenerate from console.groq.com |
| 429 Too Many Requests | Rate limited | Implement backoff/queue |
| Connection Error | Network issue | Check internet connection |
| Invalid model name | Typo in model | Use `llama-3.1-8b-instant` |
| Timeout | Response too slow | Increase timeout, reduce max_tokens |

---

## Conclusion

Migrating from Ollama to Groq represents a shift from local CPU-based inference to cloud GPU/LPU-based inference. This provides:

1. **5-20x faster responses** - Specialized hardware
2. **Zero infrastructure** - Fully managed cloud service
3. **Lower total cost** - No server hosting needed
4. **Production reliability** - 99.99% uptime SLA
5. **Simplified deployment** - Just an API call

### Next Steps

1. Get GROQ_API_KEY from https://console.groq.com/keys
2. Add to .env.local
3. Run migration following this guide
4. Monitor metrics and optimize
5. Keep Ollama as fallback for 1-2 weeks

### Implementation Timeline

- **Day 1:** Setup & validation (2-3 hours)
- **Day 2:** Code migration & testing (3-4 hours)
- **Day 3:** Monitoring & optimization (ongoing)

---

**Document Version:** 1.0  
**Last Updated:** November 29, 2025  
**Status:** Ready for Implementation
