# 🚀 Quick Deploy to Render - Cheat Sheet

## ⚡ Super Fast Version

### 1️⃣ Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2️⃣ Go to Render
🔗 **https://render.com** → Sign up with GitHub

### 3️⃣ Create Web Service
- Click **"New +"** → **"Web Service"**
- Connect your GitHub repo
- Fill in:

**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn app:app --bind 0.0.0.0:$PORT
```

**Plan:** FREE

### 4️⃣ Add Environment Variables

Copy-paste from your `.env` file:

```
AZURE_TRANSLATOR_KEY=your_key
AZURE_TRANSLATOR_REGION=centralindia
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com/
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 5️⃣ Deploy!
Click **"Create Web Service"** → Wait 3-5 min → DONE! ✨

---

## 📝 Your Live URL
```
https://your-app-name.onrender.com
```

---

## ⚠️ Remember

- ✅ Free tier handles 15 MB files
- ⚠️ App sleeps after 15 min (first request takes 30-60s)
- ✅ Auto-deploys when you push to GitHub
- ✅ 512 MB RAM, 750 hours/month FREE

---

## 🔥 That's It!

Your document translator is LIVE! 🎉
