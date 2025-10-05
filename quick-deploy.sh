#!/bin/bash
# Quick deployment script for StockFlow

echo "🚀 StockFlow Quick Deployment Script"
echo "======================================"

# Check if git repo exists
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial StockFlow commit with Smart Picks"
else
    echo "📝 Committing latest changes..."
    git add .
    git commit -m "Updated StockFlow with Smart Picks - $(date)"
fi

echo ""
echo "🎯 Choose deployment option:"
echo "1) Railway (Recommended - Free tier, auto-deploys)"
echo "2) DigitalOcean App Platform ($5/month)"
echo "3) VPS with Docker (Manual setup)"
echo "4) Just setup GitHub (I'll deploy manually)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚂 Railway Deployment:"
        echo "1. Go to: https://railway.app"
        echo "2. Sign up/login with GitHub"
        echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
        echo "4. Select this repository"
        echo "5. Add these environment variables:"
        echo "   - SLACK_BOT_TOKEN (your bot token)"
        echo "   - SLACK_APP_TOKEN (your app token)"
        echo "   - NEWSAPI_KEY (your news API key)"
        echo "   - X_API_KEY (your X API key)"
        echo "   - X_API_SECRET (your X API secret)"
        echo "6. Deploy!"
        echo ""
        echo "✨ After setup: Every git push will auto-deploy!"
        ;;
    2)
        echo ""
        echo "🌊 DigitalOcean App Platform:"
        echo "1. Go to: https://cloud.digitalocean.com/apps"
        echo "2. Create new app from GitHub repo"
        echo "3. Choose 'Worker' service type"
        echo "4. Set run command: python3 server.py"
        echo "5. Add environment variables (same as Railway)"
        echo "6. Deploy ($5/month)"
        ;;
    3)
        echo ""
        echo "🐳 VPS Docker Setup:"
        echo "Run these commands on your VPS:"
        echo ""
        echo "git clone YOUR_REPO_URL"
        echo "cd mcp-stockflow"
        echo "# Create .env file with your tokens"
        echo "docker-compose up -d"
        echo ""
        echo "📋 Full instructions in deploy.md"
        ;;
    4)
        echo ""
        echo "📦 GitHub Setup Only"
        if [ -z "$(git remote get-url origin 2>/dev/null)" ]; then
            echo "⚠️  You need to set up a GitHub repository first:"
            echo "1. Create repo on GitHub"
            echo "2. Run: git remote add origin YOUR_REPO_URL"
            echo "3. Run: git push -u origin main"
        else
            echo "📤 Pushing to GitHub..."
            git push
            echo "✅ Pushed to GitHub!"
            echo "Now you can deploy manually using any platform."
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment setup complete!"
echo "📖 See deploy.md for detailed instructions"
echo "🔧 Your StockFlow bot will run 24/7 with auto-updates!"