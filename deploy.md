# 🚀 StockFlow Deployment Guide

Deploy your StockFlow bot to run 24/7 with auto-updates!

## Option 1: Railway (Recommended - Easiest)

### Setup Steps:
1. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "StockFlow production ready"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy to Railway**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repo
   - Add environment variables:
     - `SLACK_BOT_TOKEN`
     - `SLACK_APP_TOKEN`
     - `NEWSAPI_KEY`
     - `X_API_KEY`
     - `X_API_SECRET`
   - Deploy automatically starts!

3. **Auto-updates**: Every git push automatically redeploys! 🎉

### Benefits:
- ✅ Free tier available
- ✅ Auto-deployments on git push
- ✅ Built-in monitoring
- ✅ Easy scaling

---

## Option 2: DigitalOcean App Platform

### Setup Steps:
1. **Create DO App**:
   - Go to [DigitalOcean Apps](https://cloud.digitalocean.com/apps)
   - Connect GitHub repo
   - Choose "Worker" service type
   - Set run command: `python3 server.py`

2. **Add Environment Variables**:
   - All your Slack/API tokens
   - Set in App settings → Environment

3. **Deploy**: $5/month, auto-updates on push

---

## Option 3: AWS ECS Fargate (Advanced)

### For production-grade deployment:
```bash
# Build and push to ECR
aws ecr create-repository --repository-name stockflow
docker build -t stockflow .
docker tag stockflow:latest YOUR_ECR_URI/stockflow:latest
docker push YOUR_ECR_URI/stockflow:latest

# Deploy with CloudFormation or Terraform
```

---

## Option 4: VPS with Docker (Manual but cheap)

### On any VPS ($5-10/month):
```bash
# On your VPS
git clone YOUR_REPO
cd mcp-stockflow

# Create environment file
cat > .env << EOF
SLACK_BOT_TOKEN=your_token_here
SLACK_APP_TOKEN=your_app_token_here
NEWSAPI_KEY=your_key_here
X_API_KEY=your_key_here
X_API_SECRET=your_secret_here
EOF

# Run with Docker Compose
docker-compose up -d

# Auto-restart on reboot
echo "@reboot cd /path/to/mcp-stockflow && docker-compose up -d" | crontab -

# Auto-update script (optional)
cat > update.sh << EOF
#!/bin/bash
cd /path/to/mcp-stockflow
git pull
docker-compose down
docker-compose up -d --build
EOF
chmod +x update.sh
```

---

## Monitoring & Maintenance

### Health Checks:
- Railway/DO: Built-in monitoring
- VPS: Use `docker-compose logs -f stockflow`

### Logs:
```bash
# View logs
docker-compose logs -f stockflow

# Or on Railway/DO - check their dashboards
```

### Manual Updates (VPS only):
```bash
cd /path/to/mcp-stockflow
git pull
docker-compose down
docker-compose up -d --build
```

---

## 🎯 Recommended Flow:

1. **Start with Railway** (free, easiest)
2. **Set up GitHub repo** for auto-deployments
3. **Test thoroughly**
4. **Scale up** to paid tier if needed

**Result**: Every time you make changes locally and push to GitHub, your bot automatically updates within minutes! No more manual restarts. 🚀