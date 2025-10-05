#!/bin/bash
# Complete setup script: Git + Deployment for StockFlow

echo "🚀 StockFlow Complete Setup Script"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "stockflow.py" ]; then
    echo "❌ Error: Please run this script from the mcp-stockflow directory"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo ""

# Step 1: Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "🔧 Step 1: Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already exists"
fi

# Step 2: Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore file..."
    cat > .gitignore << 'EOF'
# Environment variables
.env
*.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Logs
*.log
logs/
stockflow_v2.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
/tmp/
*.tmp
*.temp
EOF
    echo "✅ .gitignore created"
else
    echo "✅ .gitignore already exists"
fi

# Step 3: Add all files
echo "📦 Adding files to git..."
git add .
echo "✅ Files added"

# Step 4: Make initial commit
echo "💾 Making initial commit..."
git commit -m "Initial StockFlow commit with Smart Picks and 7 Novel Analysis Techniques

Features:
- Smart Picks algorithm for optimal risk/reward options
- 7 novel analysis techniques (Fractal Volatility, Gamma Squeeze, etc.)
- Production-ready deployment with auto-restart
- Slack bot integration with enhanced commands
- Fortune 500 coverage with liquid options focus
- Advanced Monte Carlo simulations (20K+)
- Real-time option analysis and monitoring"

echo "✅ Initial commit completed"

# Step 5: Get GitHub repo URL
echo ""
echo "🌐 Step 2: GitHub Repository Setup"
echo "================================="
echo ""
echo "Now you need to create a GitHub repository:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: mcp-stockflow (or your preferred name)"
echo "3. Description: Advanced options analysis bot with Smart Picks algorithm"
echo "4. Set to Public (so deployment platforms can access it)"
echo "5. DON'T initialize with README (we already have files)"
echo "6. Click 'Create repository'"
echo ""

read -p "Have you created the GitHub repository? (y/n): " created_repo

if [ "$created_repo" != "y" ] && [ "$created_repo" != "Y" ]; then
    echo "❌ Please create the GitHub repository first, then run this script again"
    exit 1
fi

echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/username/mcp-stockflow.git): " repo_url

if [ -z "$repo_url" ]; then
    echo "❌ Repository URL is required"
    exit 1
fi

# Step 6: Add remote and push
echo "🔗 Adding remote repository..."
git remote add origin "$repo_url"

echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
else
    echo "❌ Failed to push to GitHub. Please check your repository URL and permissions."
    echo "You can try manually:"
    echo "  git remote add origin $repo_url"
    echo "  git push -u origin main"
    exit 1
fi

# Step 7: Deployment options
echo ""
echo "🚀 Step 3: Choose Deployment Platform"
echo "====================================="
echo ""
echo "Your code is now on GitHub! Choose deployment:"
echo ""
echo "1) Railway (Recommended - Free tier, easiest)"
echo "2) DigitalOcean App Platform ($5/month)"
echo "3) Heroku (Classic option)"
echo "4) I'll deploy manually later"
echo ""
read -p "Enter choice (1-4): " deploy_choice

case $deploy_choice in
    1)
        echo ""
        echo "🚂 Railway Deployment Instructions:"
        echo "=================================="
        echo ""
        echo "1. Go to: https://railway.app"
        echo "2. Sign up/login with your GitHub account"
        echo "3. Click 'New Project'"
        echo "4. Select 'Deploy from GitHub repo'"
        echo "5. Choose your repository: $(basename "$repo_url" .git)"
        echo "6. Railway will automatically detect it's a Python app"
        echo ""
        echo "7. Add these Environment Variables in Railway dashboard:"
        echo "   Variables tab → Add Variable:"
        echo ""
        echo "   SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN"
        echo "   SLACK_APP_TOKEN=$SLACK_APP_TOKEN"
        echo "   NEWSAPI_KEY=$NEWSAPI_KEY"
        echo "   X_API_KEY=$X_API_KEY"
        echo "   X_API_SECRET=$X_API_SECRET"
        echo ""
        echo "8. Click 'Deploy' - your bot will be live in ~2 minutes!"
        echo ""
        echo "✨ FUTURE UPDATES: Just run 'git push' and Railway auto-deploys!"
        ;;
    2)
        echo ""
        echo "🌊 DigitalOcean App Platform:"
        echo "1. Go to: https://cloud.digitalocean.com/apps"
        echo "2. Create App from GitHub repo"
        echo "3. Select your repository"
        echo "4. Choose 'Worker' component type"
        echo "5. Add environment variables"
        echo "6. Deploy ($5/month)"
        ;;
    3)
        echo ""
        echo "🟣 Heroku Instructions:"
        echo "1. Go to: https://dashboard.heroku.com/new-app"
        echo "2. Connect to GitHub repo"
        echo "3. Add environment variables in Settings → Config Vars"
        echo "4. Deploy"
        ;;
    4)
        echo ""
        echo "📋 Manual deployment info saved in deploy.md"
        echo "Your code is ready to deploy on any platform!"
        ;;
esac

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Git repository initialized"
echo "✅ Code pushed to GitHub: $repo_url"
echo "✅ Ready for 24/7 deployment"
echo ""
echo "📋 Next steps:"
echo "1. Deploy using the instructions above"
echo "2. Test your Smart Picks command in Slack"
echo "3. Every future code change: just 'git push' to auto-deploy!"
echo ""
echo "🎯 Your StockFlow bot with Smart Picks is ready for production!"