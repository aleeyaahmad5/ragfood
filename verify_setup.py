"""
Verify Groq migration setup.

Run this to test all components before running the full RAG pipeline.
"""

import os
import sys
from dotenv import load_dotenv

print("\n" + "="*60)
print("🔍 Groq Migration Setup Verification")
print("="*60 + "\n")

# Load environment variables
load_dotenv('.env.local')

# Check 1: Verify environment variables
print("1️⃣  Checking environment variables...")
upstash_url = os.getenv('UPSTASH_VECTOR_REST_URL')
upstash_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')

if not upstash_url or not upstash_token:
    print("❌ Missing Upstash credentials")
    sys.exit(1)
else:
    print(f"✅ Upstash URL: {upstash_url[:50]}...")

if not groq_api_key:
    print("❌ Missing GROQ_API_KEY")
    sys.exit(1)
else:
    print(f"✅ Groq API Key: {groq_api_key[:20]}...")

# Check 2: Verify dependencies
print("\n2️⃣  Checking dependencies...")

try:
    from upstash_vector import Index
    print("✅ upstash-vector installed")
except ImportError:
    print("❌ upstash-vector not installed")
    print("   Run: pip install upstash-vector")
    sys.exit(1)

try:
    from groq import Groq
    print("✅ groq installed")
except ImportError:
    print("❌ groq not installed")
    print("   Run: pip install groq")
    sys.exit(1)

try:
    import error_handling
    print("✅ error_handling module found")
except ImportError:
    print("❌ error_handling.py not found")
    sys.exit(1)

try:
    import rate_limiter
    print("✅ rate_limiter module found")
except ImportError:
    print("❌ rate_limiter.py not found")
    sys.exit(1)

try:
    import cost_tracker
    print("✅ cost_tracker module found")
except ImportError:
    print("❌ cost_tracker.py not found")
    sys.exit(1)

# Check 3: Test Upstash connection
print("\n3️⃣  Testing Upstash Vector connection...")
try:
    index = Index(
        url=upstash_url,
        token=upstash_token
    )
    info = index.info()
    print(f"✅ Upstash connected")
    print(f"   Vectors: {info.vector_count}")
    print(f"   Dimensions: {info.dimension}")
except Exception as e:
    print(f"❌ Upstash connection failed: {e}")
    sys.exit(1)

# Check 4: Test Groq connection
print("\n4️⃣  Testing Groq API connection...")
try:
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello"}],
        max_completion_tokens=10
    )
    print(f"✅ Groq API connected")
    print(f"   Response: {completion.choices[0].message.content[:50]}")
    print(f"   Tokens used: {completion.usage.total_tokens}")
except Exception as e:
    print(f"❌ Groq connection failed: {e}")
    sys.exit(1)

# Check 5: Verify data file
print("\n5️⃣  Checking data file...")
if os.path.exists('foods.json'):
    import json
    with open('foods.json', 'r', encoding='utf-8') as f:
        foods = json.load(f)
    print(f"✅ foods.json found ({len(foods)} items)")
else:
    print("⚠️  foods.json not found (will need to be created)")

print("\n" + "="*60)
print("✅ All checks passed! Ready to run RAG pipeline")
print("="*60)
print("\nRun: python rag_run_groq.py\n")
