# 🚀 Deploy to Render - Complete Step-by-Step Guide

## ✅ Prerequisites

1. ✅ GitHub account
2. ✅ Your code pushed to GitHub repository
3. ✅ Render account (free) - https://render.com

---

## 📋 Step-by-Step Deployment

### **Step 1: Push Your Code to GitHub**

If you haven't already:

```bash
# Initialize git (if not already done)
cd c:\Users\UmarKhan\Downloads\azure_ai_env
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Document Translator App"

# Create repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

**⚠️ IMPORTANT:** Make sure `.env` is in `.gitignore` (it already is!)

---

### **Step 2: Sign Up on Render**

1. Go to **https://render.com**
2. Click **"Get Started"**
3. Sign up with **GitHub** (recommended for easy integration)
4. Authorize Render to access your repositories

---

### **Step 3: Create a New Web Service**

1. Click **"New +"** button (top right)
2. Select **"Web Service"**
3. Click **"Connect a repository"**
4. Find your repository: `azure_ai_env` or whatever you named it
5. Click **"Connect"**

---

### **Step 4: Configure the Service**

Fill in these settings:

#### **Basic Settings:**
- **Name:** `document-translator` (or any name you like)
- **Region:** Choose closest to you (e.g., Singapore, Frankfurt)
- **Branch:** `main` (or your default branch)
- **Root Directory:** Leave empty (unless your app is in a subdirectory)

#### **Build & Deploy Settings:**
- **Runtime:** `Python 3`
- **Build Command:** 
  ```
  pip install -r requirements.txt
  ```
- **Start Command:** 
  ```
  gunicorn app:app --bind 0.0.0.0:$PORT
  ```

#### **Plan:**
- Select **"Free"** 
- ✅ 512 MB RAM
- ✅ Shared CPU

---

### **Step 5: Add Environment Variables**

Scroll down to **"Environment Variables"** section.

Click **"Add Environment Variable"** for each:

**Copy these from your `.env` file:**

```
AZURE_TRANSLATOR_KEY = your_azure_translator_key_here

AZURE_TRANSLATOR_REGION = centralindia

AZURE_TRANSLATOR_ENDPOINT = https://api.cognitive.microsofttranslator.com/

AZURE_OPENAI_API_KEY = your_azure_openai_key_here

AZURE_OPENAI_ENDPOINT = https://your-resource.openai.azure.com/

AZURE_OPENAI_DEPLOYMENT_NAME = gpt-4o-mini

AZURE_OPENAI_API_VERSION = 2024-02-15-preview
```

**Optional (if using DeepL):**
```
DEEPL_API_KEY = your_deepl_api_key_here
```

---

### **Step 6: Create Web Service**

1. Click **"Create Web Service"** button at the bottom
2. Render will start building your app
3. Wait 2-5 minutes for the first deployment

You'll see logs showing:
```
==> Installing dependencies
==> Building...
==> Starting server...
==> Your service is live! 🎉
```

---

### **Step 7: Access Your Live App**

Once deployed, you'll get a URL like:
```
https://document-translator-xxxx.onrender.com
```

Click the URL to open your app! 🎉

---

## 🎯 Testing Your Deployment

1. Open your Render URL
2. Try uploading a small PDF (test with 1-2 pages first)
3. Select languages
4. Click "Translate Document"
5. Download the result

---

## 🐛 Troubleshooting

### **Issue 1: Build Fails**
**Error:** `No module named 'X'`
**Fix:** Check `requirements.txt` has all dependencies

### **Issue 2: App Crashes on Start**
**Error:** `Application error`
**Fix:** 
1. Check logs in Render dashboard
2. Make sure environment variables are set correctly
3. Verify start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

### **Issue 3: "Cold Start" Delay**
**Behavior:** First request takes 30-60 seconds
**This is normal** on Render free tier! The app sleeps after 15 min of inactivity.

**Solutions:**
- Use a cron job to ping your app every 14 minutes
- Upgrade to paid tier ($7/month) for always-on
- Or just accept the delay (it's free!)

### **Issue 4: File Upload Fails**
**Error:** File too large
**Fix:** 
1. Check file is under 16 MB
2. Render free tier should handle 15 MB files fine
3. If issues persist, check server logs

---

## ⚙️ Advanced Configuration

### **Auto-Deploy from GitHub**

Render automatically deploys when you push to GitHub!

```bash
# Make changes locally
git add .
git commit -m "Updated feature"
git push

# Render will auto-deploy in 2-3 minutes!
```

### **Custom Domain**

1. Go to your service settings
2. Click **"Custom Domain"**
3. Add your domain (e.g., `translator.yourdomain.com`)
4. Follow DNS setup instructions
5. Free SSL included!

### **View Logs**

1. Go to your service dashboard
2. Click **"Logs"** tab
3. See real-time logs

### **Monitor Usage**

1. Dashboard shows:
   - Requests per minute
   - Memory usage
   - CPU usage
   - Deploy history

---

## 📊 Render Free Tier Limits

| Feature | Limit |
|---------|-------|
| **RAM** | 512 MB |
| **CPU** | Shared |
| **Bandwidth** | 100 GB/month |
| **Build Minutes** | 500/month |
| **Hours** | 750/month |
| **Sleep After** | 15 min inactivity |
| **File Size** | Up to 16 MB ✅ |

---

## 🚀 Your App URL Structure

After deployment:
- **Homepage:** `https://your-app.onrender.com/`
- **Health Check:** `https://your-app.onrender.com/health`
- **API Docs:** Check your Flask routes

---

## 💡 Tips for Success

1. **First deployment takes longest** (5-10 min)
2. **Subsequent deploys are faster** (2-3 min)
3. **Free tier sleeps** - First request after sleep takes 30-60s
4. **Check logs** if anything goes wrong
5. **Environment variables** must be EXACT (copy from .env)

---

## 🎉 You're Done!

Your document translator is now live and accessible worldwide for FREE! 🌍

**Share your URL and start translating documents!**

---

## 🆘 Need Help?

- **Render Docs:** https://render.com/docs
- **Community Forum:** https://community.render.com
- **Check logs** in Render dashboard for errors

---

## 💰 Upgrading (Optional)

If you need:
- ✅ No cold starts
- ✅ Always-on app
- ✅ More RAM/CPU
- ✅ Priority support

Upgrade to **Starter plan** ($7/month):
1. Go to service settings
2. Click "Change Plan"
3. Select "Starter"
4. Enjoy instant responses! ⚡
