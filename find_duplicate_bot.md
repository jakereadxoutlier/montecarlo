# Finding the Duplicate Bot

## 1. Check Railway Dashboard

Go to https://railway.app/dashboard and look for:

- **Multiple services** in your project (you might have both `stockflow` and `montecarlo` services)
- **Multiple projects** (you might have deployed to different projects)
- **Multiple environments** (production, staging, etc.)

Look for anything running with these names:
- stockflow
- montecarlo
- slack-bot
- trading-bot
- options-bot

## 2. Check Railway CLI

If you have Railway CLI installed:

```bash
# List all projects
railway list

# Check all services in current project
railway status

# Check deployments
railway logs
```

## 3. Check Slack App Configuration

Go to https://api.slack.com/apps and:

1. Find your app (probably called "StockFlow" or "MonteCarlo")
2. Go to "OAuth & Permissions"
3. Check if you have MULTIPLE apps with the same Bot Token
4. Check "Event Subscriptions" - see if multiple URLs are registered

## 4. Check Other Platforms

The old bot might be running on:

- **Heroku**: Check https://dashboard.heroku.com
- **Replit**: Check https://replit.com
- **Render**: Check https://dashboard.render.com
- **Fly.io**: Check https://fly.io/dashboard
- **Another computer/server** running the code

## 5. Nuclear Option - Regenerate Slack Tokens

If you can't find the duplicate:

1. Go to https://api.slack.com/apps
2. Select your app
3. Go to "OAuth & Permissions"
4. Click "Regenerate" for both tokens
5. Update ONLY the Railway deployment with new tokens
6. The old bot will stop working

## 6. Quick Test

After deployment, when you type `help` in Slack:

- **CORRECT BOT**: Shows "MonteCarlo UNIFIED v2" with deployment info
- **OLD BOT**: Shows "StockFlow Bot Commands:"

## 7. Railway-Specific Check

In Railway dashboard:
1. Click on your project
2. Check the "Deployments" tab
3. Look for multiple ACTIVE deployments
4. Check if you have multiple services running

Common issues:
- You might have TWO Railway projects
- You might have renamed a service (old one still running)
- You might have multiple environments (prod/dev)