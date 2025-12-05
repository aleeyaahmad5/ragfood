# GitHub Setup Guide for RAGFood

## ✅ Pre-Push Checklist

Before pushing to GitHub, make sure:

- [x] `.venv` folder is in `.gitignore` (NOT committed)
- [x] `.env.local` is in `.gitignore` (secrets NOT committed)
- [x] `chroma_db/` is in `.gitignore` (local data NOT committed)
- [x] `.vercel/` is in `.gitignore` (Vercel config NOT committed)
- [x] `node_modules/` is in `.gitignore` (not needed in repo)
- [x] Python dependencies documented in `requirements.txt`

---

## 🚀 Initial Push to GitHub

### 1. Check Git Status

```bash
git status
```

Should show:
- ✅ `requirements.txt` (tracked)
- ✅ Python files (tracked)
- ✅ Markdown docs (tracked)
- ✅ Next.js config (tracked)
- ❌ `.venv/` (NOT shown - ignored)
- ❌ `.env.local` (NOT shown - ignored)
- ❌ `chroma_db/` (NOT shown - ignored)
- ❌ `node_modules/` (NOT shown - ignored)

### 2. Add All Tracked Files

```bash
git add .
```

### 3. Create Initial Commit

```bash
git commit -m "Initial RAGFood project - Groq + Upstash Vector setup"
```

### 4. Push to GitHub

```bash
git push -u origin main
```

---

## 📋 What Gets Committed

✅ **Code Files:**
- `rag_run_groq.py` - Main RAG pipeline
- `error_handling.py` - Error handling utilities
- `rate_limiter.py` - Rate limiting
- `cost_tracker.py` - Cost tracking
- `verify_setup.py` - Setup verification
- `backup_rollback.py` - Backup utilities

✅ **Configuration:**
- `requirements.txt` - Python dependencies
- `.gitignore` - Files to ignore
- `tsconfig.json` - TypeScript config
- `next.config.js` - Next.js config
- `package.json` - Node dependencies

✅ **Documentation:**
- `README.md` - Project overview
- `IMPLEMENTATION_GUIDE.md` - Setup guide
- `IMPLEMENTATION_REPORT.md` - Implementation details
- `GROQ_MIGRATION_PLAN.md` - Migration documentation
- `MIGRATION_DESIGN.md` - Architecture documentation
- `GITHUB_SETUP.md` - This file

---

## ❌ What Does NOT Get Committed

❌ **Secrets & Environment:**
- `.env.local` - Contains API keys
- `.env` - Environment variables

❌ **Dependencies:**
- `.venv/` - Python virtual environment
- `node_modules/` - Node dependencies

❌ **Local Data:**
- `chroma_db/` - ChromaDB local database
- `groq_usage.json` - Cost tracking data

❌ **Generated Files:**
- `.vercel/` - Vercel deployment files
- `.next/` - Next.js build files
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files

---

## 👥 For New Contributors

### Clone & Setup

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ragfood.git
cd ragfood

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install Node dependencies
npm install

# 6. Create .env.local with your secrets
# Copy from the team and add your own keys:
#   - UPSTASH_VECTOR_REST_URL
#   - UPSTASH_VECTOR_REST_TOKEN
#   - UPSTASH_VECTOR_REST_READONLY_TOKEN
#   - GROQ_API_KEY
#   - VERCEL_OIDC_TOKEN

# 7. Verify setup
python verify_setup.py

# 8. Run development
python rag_run_groq.py
```

---

## 📝 Environment Variables Template

Create `.env.local` in project root:

```dotenv
# Upstash Vector Database
UPSTASH_VECTOR_REST_URL="https://YOUR_INSTANCE.upstash.io"
UPSTASH_VECTOR_REST_TOKEN="YOUR_TOKEN_HERE"
UPSTASH_VECTOR_REST_READONLY_TOKEN="YOUR_READONLY_TOKEN_HERE"

# Groq API
GROQ_API_KEY="gsk_YOUR_KEY_HERE"

# Vercel (optional, for deployment)
VERCEL_OIDC_TOKEN="YOUR_VERCEL_TOKEN_HERE"
```

**⚠️ NEVER commit `.env.local`**

---

## 🔐 Security Best Practices

1. **Never commit secrets** - Always use `.env.local`
2. **Add to `.gitignore`** - Already done in this project
3. **Rotate keys regularly** - Change API keys periodically
4. **Use environment variables** - Never hardcode credentials
5. **Document in README** - Tell users how to set up `.env.local`

---

## 📦 Project Structure

```
ragfood/
├── .venv/                          # ❌ NOT committed
├── .env.local                      # ❌ NOT committed
├── .gitignore                      # ✅ Committed
├── .vercel/                        # ❌ NOT committed
├── chroma_db/                      # ❌ NOT committed
├── node_modules/                   # ❌ NOT committed
├── .next/                          # ❌ NOT committed
│
├── Python Backend
├── ├── rag_run_groq.py            # ✅ Main pipeline
├── ├── error_handling.py          # ✅ Error utilities
├── ├── rate_limiter.py            # ✅ Rate limiting
├── ├── cost_tracker.py            # ✅ Cost tracking
├── ├── verify_setup.py            # ✅ Setup verification
├── ├── backup_rollback.py         # ✅ Backup utilities
├── ├── requirements.txt           # ✅ Python deps
├── ├── foods.json                 # ✅ Sample data
├── └── groq_usage.json            # ❌ NOT committed (generated)
│
├── Next.js Frontend
├── ├── next.config.js             # ✅ Next.js config
├── ├── tsconfig.json              # ✅ TypeScript config
├── ├── package.json               # ✅ Node dependencies
├── ├── public/                    # ✅ Static files
├── └── src/                       # ✅ React components
│
└── Documentation
  ├── README.md                    # ✅ Main documentation
  ├── GITHUB_SETUP.md              # ✅ This file
  ├── IMPLEMENTATION_GUIDE.md      # ✅ Quick start guide
  ├── IMPLEMENTATION_REPORT.md     # ✅ Implementation summary
  ├── GROQ_MIGRATION_PLAN.md       # ✅ Migration guide
  └── MIGRATION_DESIGN.md          # ✅ Architecture guide
```

---

## 🚀 GitHub Deployment

### Option 1: Deploy Next.js to Vercel

```bash
npm install -g vercel
vercel
```

### Option 2: Deploy Python Backend

Use Railway, Render, or Heroku:

```bash
# Create requirements.txt (already done)
pip freeze > requirements.txt

# Create Procfile
echo "web: python rag_run_groq.py" > Procfile

# Deploy with your platform
```

---

## ✨ Ready to Push!

Your project is now ready for GitHub. The `.venv` will stay on your local machine, secrets are safe in `.env.local`, and only the source code gets committed.

```bash
git push origin main
```

**Done! 🎉**

---

## 📚 Next Steps

- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Share `.env.local` setup guide with team (NOT the file itself!)
- [ ] Set up GitHub Actions for CI/CD
- [ ] Deploy to production (Vercel for frontend, Railway for backend)
- [ ] Monitor usage and costs

---

## 🆘 Troubleshooting

**Q: I accidentally committed `.env.local`!**
```bash
git rm --cached .env.local
git commit -m "Remove .env.local from git"
git push
# Then rotate all API keys!
```

**Q: `.venv` is being tracked?**
```bash
git rm -r --cached .venv
git commit -m "Remove .venv from git"
```

**Q: How do I update `.gitignore`?**
```bash
# Edit .gitignore
# Then:
git add .gitignore
git commit -m "Update .gitignore"
git push
```

---

**Status:** ✅ Ready for GitHub  
**Last Updated:** December 6, 2025
