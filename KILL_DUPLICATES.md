# NUCLEAR OPTION - KILL ALL DUPLICATES

## Step 1: Check Railway Dashboard NOW

Go to https://railway.app/dashboard

### DELETE these if they exist:
- Any service named `stockflow`
- Any service named `standalone-slack-app`
- Any service named `montecarlo` (if different from current)
- Any OLD deployments still running

### Check for multiple projects:
- You might have TWO Railway projects
- Delete the old one completely

## Step 2: Regenerate Slack Tokens (GUARANTEED TO WORK)

1. Go to https://api.slack.com/apps
2. Select your app
3. Go to "OAuth & Permissions"
4. Click "Regenerate" for:
   - **Bot User OAuth Token** (starts with xoxb-)
   - **App-Level Token** (starts with xapp-)
5. Save the new tokens

## Step 3: Update ONLY the Railway Deployment

1. In Railway, go to your service
2. Go to "Variables"
3. Update:
   - `SLACK_BOT_TOKEN` with the new xoxb- token
   - `SLACK_APP_TOKEN` with the new xapp- token
4. Deploy

## Step 4: Verify

Type `help` in Slack:
- Should see ONLY ONE response
- Should show "MonteCarlo UNIFIED v2"
- Should show deployment info

## Files We've DELETED (can't run anymore):
- ✅ standalone_slack_app.py
- ✅ standalone_slack_app_backup.py
- ✅ standalone_slack_app_fixed.py
- ✅ debug_slack_app.py
- ✅ test_bot_info.py
- ✅ All old server files

## Only Files That Can Run:
- ✅ montecarlo_unified.py (via server_railway_unified.py)
- That's it. Nothing else.

## If Still Having Issues:

The duplicate is running somewhere else:
- Check Heroku
- Check Replit
- Check local machines
- Check if someone else deployed it

**REGENERATING THE TOKENS WILL KILL ALL DUPLICATES INSTANTLY**