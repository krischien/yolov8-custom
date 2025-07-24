# Deployment Guide for YOLOv8 Web Application

This guide will help you deploy your YOLOv8 object detection web application using GitHub and cloud hosting platforms.

## 🚀 Quick Deployment Options

### Option 1: Railway (Recommended - Free Tier Available)

1. **Create a Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with your GitHub account

2. **Connect Your Repository**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Environment Variables**
   - Add these environment variables in Railway dashboard:
     ```
     PORT=5000
     FLASK_ENV=production
     ```

4. **Deploy**
   - Railway will automatically detect the Python app
   - It will install dependencies from `requirements.txt`
   - Your app will be live at a Railway URL

### Option 2: Render (Free Tier Available)

1. **Create a Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with your GitHub account

2. **Create a Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Set build command: `cd code/webapp && pip install -r requirements.txt`
   - Set start command: `cd code/webapp && gunicorn app:app`

3. **Configure Environment**
   - Add environment variables if needed
   - Set auto-deploy to enabled

### Option 3: Heroku (Paid)

1. **Install Heroku CLI**
   ```bash
   # Windows
   winget install --id=Heroku.HerokuCLI
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## 📁 GitHub Repository Setup

### 1. Initialize Git Repository
```bash
cd yolov8-custom
git init
git add .
git commit -m "Initial commit"
```

### 2. Create GitHub Repository
- Go to [github.com](https://github.com)
- Click "New repository"
- Name it `yolov8-custom` or your preferred name
- Don't initialize with README (you already have one)

### 3. Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 🔧 GitHub Actions (Automatic Deployment)

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that will:

1. **Test** your application on every push
2. **Deploy** automatically when you push to main/master branch

### Setup GitHub Actions Secrets

For Railway deployment:
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add these secrets:
   - `RAILWAY_TOKEN`: Your Railway API token
   - `RAILWAY_SERVICE`: Your Railway service ID

For Render deployment:
1. Add these secrets:
   - `RENDER_API_KEY`: Your Render API key
   - `RENDER_SERVICE_ID`: Your Render service ID

## 🌐 Custom Domain (Optional)

### Railway
1. Go to your Railway project
2. Click on your service
3. Go to Settings → Domains
4. Add your custom domain

### Render
1. Go to your Render dashboard
2. Select your web service
3. Go to Settings → Custom Domains
4. Add your domain

## 📊 Monitoring and Logs

### Railway
- View logs in the Railway dashboard
- Monitor performance and usage

### Render
- Access logs in the Render dashboard
- Set up alerts for downtime

### GitHub Actions
- View deployment status in Actions tab
- Check for any deployment failures

## 🔒 Environment Variables

Create a `.env` file locally for development:
```env
FLASK_ENV=development
WEBAPP_HOST=0.0.0.0
WEBAPP_PORT=5000
```

For production, set these in your hosting platform's dashboard.

## 🐛 Troubleshooting

### Common Issues:

1. **Port Issues**
   - Make sure your app listens on `0.0.0.0` and uses `os.environ.get('PORT', 5000)`

2. **Dependencies**
   - Ensure all dependencies are in `requirements.txt`
   - Check for version conflicts

3. **Model Files**
   - Large model files (`.pt`) are excluded from Git
   - Upload them separately or use model hosting services

4. **Memory Issues**
   - YOLOv8 models can be memory-intensive
   - Consider using smaller models for production

### Getting Help:
- Check the deployment platform's logs
- Review GitHub Actions workflow status
- Ensure all files are properly committed

## 🎯 Next Steps

1. **Choose a deployment platform** (Railway recommended for beginners)
2. **Set up your GitHub repository**
3. **Configure environment variables**
4. **Deploy and test your application**
5. **Set up monitoring and alerts**

Your YOLOv8 web application will be live and accessible from anywhere in the world! 🌍 