# Railway Deployment Troubleshooting

If you're getting "Railpack could not determine how to build the app", try these solutions in order:

## 🚀 Solution 1: Use the Simple App (Recommended)

1. **Test with the simple app first:**
   ```bash
   # Railway will use simple_app.py and requirements_simple.txt
   git add .
   git commit -m "Add simple app for Railway testing"
   git push
   ```

2. **Check Railway dashboard** - it should now deploy successfully

3. **Once simple app works, switch to full app:**
   - Update `railway.json` to use the full app
   - Test again

## 🔧 Solution 2: Manual Railway Configuration

1. **In Railway dashboard:**
   - Go to your project settings
   - Set "Root Directory" to `code`
   - Set "Build Command" to `pip install -r requirements.txt`
   - Set "Start Command" to `cd webapp && gunicorn app:app --bind 0.0.0.0:$PORT`

## 🐳 Solution 3: Use Dockerfile

1. **Railway will automatically detect the Dockerfile**
2. **No additional configuration needed**
3. **Push your code and Railway will use Docker deployment**

## 📁 Solution 4: Restructure Repository

If nothing works, restructure your repository:

```
yolov8-custom/
├── app.py              # Main Flask app
├── requirements.txt    # Dependencies
├── Procfile           # Railway/Heroku config
├── runtime.txt        # Python version
├── webapp/            # Web app files
│   ├── templates/
│   ├── static/
│   └── ...
└── API/               # API files
    └── ...
```

## 🔍 Debug Steps

1. **Check Railway logs** for specific error messages
2. **Verify file structure** - all files should be in the correct locations
3. **Test locally** - run `python simple_app.py` locally first
4. **Check Python version** - ensure it matches `runtime.txt`

## 🎯 Quick Fix Commands

```bash
# Test simple app locally
cd code
python simple_app.py

# If it works locally, push to Railway
git add .
git commit -m "Fix Railway deployment"
git push
```

## 📞 If Still Not Working

1. **Check Railway documentation**: https://docs.railway.app/
2. **Try Render.com** as an alternative
3. **Use Heroku** with the Procfile approach
4. **Contact Railway support** with your specific error

## 🏆 Success Indicators

- Railway shows "Deploying..." then "Deployed"
- Your app URL shows the JSON response
- Health check endpoint `/health` returns `{"status": "healthy"}` 