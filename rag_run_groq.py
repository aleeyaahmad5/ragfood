"""
RAGFood - Retrieval Augmented Generation with Upstash Vector + Groq LLM

Architecture:
1. Vector Search: Upstash Vector (automatic embeddings, 50-150ms)
2. LLM Generation: Groq Cloud API (5-20x faster, 200-500ms)
3. Cost Tracking: Monitor token usage and estimate costs

Migration Benefits:
✅ 5-20x faster inference (Groq specialized hardware)
✅ No local model management needed
✅ Production-ready infrastructure (99.99% uptime)
✅ Automatic scaling
✅ Token usage tracking for cost optimization
✅ Lower total cost (~$10-15/month vs $800-2000/month with Ollama)

Usage:
    python rag_run_groq.py
"""

import os
import json
import time
from typing import List, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq

# Import utility modules
from error_handling import (
    validate_groq_api_key,
    retry_with_backoff,
    RetryConfig,
    parse_groq_error,
    GroqAuthenticationError,
    GroqRateLimitError
)
from rate_limiter import TokenBucketLimiter
from cost_tracker import CostTracker

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables
load_dotenv('.env.local')

UPSTASH_URL = os.getenv('UPSTASH_VECTOR_REST_URL')
UPSTASH_TOKEN = os.getenv('UPSTASH_VECTOR_REST_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

JSON_FILE = "foods.json"
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# Global clients
upstash_index: Optional[Index] = None
groq_client: Optional[Groq] = None
rate_limiter: Optional[TokenBucketLimiter] = None
cost_tracker: Optional[CostTracker] = None

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_clients() -> bool:
    """Initialize all clients and validate setup."""
    global upstash_index, groq_client, rate_limiter, cost_tracker
    
    print("\n" + "="*60)
    print("🚀 Initializing RAGFood with Groq Backend")
    print("="*60)
    
    # Validate credentials
    try:
        validate_groq_api_key(GROQ_API_KEY)
        print("✅ GROQ_API_KEY format valid")
    except ValueError as e:
        print(f"❌ {e}")
        return False
    
    # Initialize Upstash Vector
    try:
        upstash_index = Index(
            url=UPSTASH_URL,
            token=UPSTASH_TOKEN
        )
        info = upstash_index.info()
        print(f"✅ Upstash Vector connected")
        print(f"   Vectors: {info.vector_count}")
        print(f"   Dimensions: {info.dimension}")
    except Exception as e:
        print(f"❌ Upstash connection failed: {e}")
        return False
    
    # Initialize Groq
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Test connection with minimal call
        test_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=1
        )
        print(f"✅ Groq API connected")
        print(f"   Model: {GROQ_MODEL}")
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return False
    
    # Initialize rate limiter and cost tracker
    rate_limiter = TokenBucketLimiter(
        max_requests_per_minute=30,  # Free tier limit
        max_tokens_per_minute=100
    )
    
    cost_tracker = CostTracker("groq_usage.json")
    
    print(f"✅ Rate limiter initialized (30 req/min)")
    print(f"✅ Cost tracker initialized")
    
    return True

def load_and_migrate_foods() -> bool:
    """Load foods from JSON and upsert to Upstash Vector."""
    
    if not os.path.exists(JSON_FILE):
        print(f"❌ {JSON_FILE} not found")
        return False
    
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        food_data = json.load(f)
    
    print(f"\n🔄 Loading {len(food_data)} food items...")
    
    vectors_to_upsert = []
    
    for item in food_data:
        # Enhance text with metadata
        enriched_text = item["text"]
        if "region" in item:
            enriched_text += f" This food is popular in {item['region']}."
        if "type" in item:
            enriched_text += f" It is a type of {item['type']}."
        
        # Create upsert tuple: (id, text, metadata)
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
    
    # Batch upsert for efficiency
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        try:
            upstash_index.upsert(vectors=batch)
            print(f"✅ Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
        except Exception as e:
            print(f"❌ Error upserting batch: {e}")
            return False
    
    print(f"✅ {len(food_data)} foods ready for querying")
    return True

# ============================================================================
# GROQ LLM GENERATION
# ============================================================================

def generate_answer(prompt: str) -> tuple[str, Dict]:
    """
    Generate answer using Groq API with retry logic.
    
    Args:
        prompt: Full prompt with context
    
    Returns:
        (answer_text, usage_stats)
    
    Usage stats includes:
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - finish_reason
    """
    
    def _call_groq():
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
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
    
    def on_retry(attempt: int, wait_time: float, error: Exception):
        """Callback for retry events."""
        error_msg, error_type = parse_groq_error(error)
        
        if error_type == GroqRateLimitError:
            print(f"⚠️  Rate limited (attempt {attempt}). Waiting {wait_time:.1f}s...")
        elif error_type == GroqAuthenticationError:
            print(f"❌ Authentication failed: {error_msg}")
        else:
            print(f"⚠️  Error (attempt {attempt}): {error_type.__name__}. Retrying in {wait_time:.1f}s...")
    
    # Create retry config
    config = RetryConfig(
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=30.0,
        exponential_base=2.0
    )
    
    try:
        answer, usage = retry_with_backoff(
            _call_groq,
            config=config,
            on_retry=on_retry
        )
        
        # Track usage and cost
        cost = cost_tracker.log_query(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
            model=GROQ_MODEL
        )
        
        return answer, usage
    
    except Exception as e:
        raise Exception(f"Failed to generate answer after retries: {str(e)}")

# ============================================================================
# RAG PIPELINE
# ============================================================================

def rag_query(question: str) -> Dict:
    """
    Complete RAG pipeline with metrics tracking.
    
    Args:
        question: User question
    
    Returns:
        {
            "answer": str,
            "sources": List[Dict],
            "metrics": {
                "vector_search_time": float,
                "llm_generation_time": float,
                "total_time": float,
                "tokens_used": int,
                "cost_estimate": float
            }
        }
    """
    
    start_time = time.time()
    
    try:
        # Check rate limit
        rate_limit_status = rate_limiter.get_status()
        if rate_limit_status['requests_remaining'] <= 0:
            wait_time = rate_limiter.wait_if_needed()
            print(f"ℹ️  Waited {wait_time:.1f}s for rate limit")
        
        # Step 1: Vector search
        print("\n🧠 Searching Upstash Vector database...")
        vector_start = time.time()
        
        results = upstash_index.query(
            data=question,
            top_k=3,
            include_metadata=True
        )
        
        vector_time = time.time() - vector_start
        
        if not results:
            return {
                "answer": "No relevant information found in the knowledge base.",
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
If the information is not in the context, say you don't have enough information to answer.

Context:
{context}

Question: {question}
Answer:"""
        
        # Step 5: Generate answer with Groq
        print("\n🚀 Generating answer with Groq (llama-3.1-8b-instant)...\n")
        llm_start = time.time()
        
        answer, usage_stats = generate_answer(prompt)
        
        llm_time = time.time() - llm_start
        
        # Calculate cost
        cost_estimate = (usage_stats['total_tokens'] / 1000) * 0.0001
        
        # Print metrics
        print(f"\n📊 Generation Complete")
        print(f"   Input tokens: {usage_stats['prompt_tokens']}")
        print(f"   Output tokens: {usage_stats['completion_tokens']}")
        print(f"   Total tokens: {usage_stats['total_tokens']}")
        print(f"   Generation time: {llm_time*1000:.0f}ms")
        print(f"   Estimated cost: ${cost_estimate:.6f}")
        
        # Record rate limit usage
        rate_limiter.record_request(usage_stats['total_tokens'])
        status = rate_limiter.get_status()
        print(f"   Rate limit: {status['requests_remaining']} requests remaining")
        
        return {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "vector_search_time": vector_time,
                "llm_generation_time": llm_time,
                "total_time": time.time() - start_time,
                "tokens_used": usage_stats['total_tokens'],
                "cost_estimate": cost_estimate
            }
        }
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
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
    
    # Initialize
    if not init_clients():
        print("\n❌ Initialization failed")
        return
    
    # Load data
    if not load_and_migrate_foods():
        print("\n❌ Data loading failed")
        return
    
    # Interactive loop
    print("\n" + "="*60)
    print("✨ RAGFood Ready! Ask a question (type 'exit' to quit)")
    print("="*60 + "\n")
    
    session_start = datetime.now()
    
    while True:
        try:
            question = input("You: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                # Print session summary
                cost_tracker.print_session_report(session_start)
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Run RAG pipeline
            result = rag_query(question)
            
            print(f"\n🤖 Assistant: {result['answer']}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            cost_tracker.print_session_report(session_start)
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")

if __name__ == "__main__":
    main()
