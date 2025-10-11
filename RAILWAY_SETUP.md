# 🚂 Railway Deployment Guide - Tradier + Perplexity Integration

## ✅ What's New

Your StockFlow bot now uses:
- **Tradier API** ($10/month) - Reliable market data, no rate limit headaches
- **Perplexity AI** ($20/month) - Superior sentiment analysis
- **Adaptive monitoring** - 30s checks during market hours, 5min after-hours
- **EOD caching** - Full 24/7 analysis capability

**Total cost**: $30/month (Tradier + Perplexity)

---

## 🔧 Railway Environment Variables Setup

### Required (Existing - Keep These)
```
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-level-token-here
```

### **NEW - Add These to Railway Dashboard**

#### 1. Tradier API ($10/month - REQUIRED)
```
TRADIER_API_KEY=your_tradier_api_key_here
```

**How to get**:
1. Sign up at https://tradier.com/products/market-data
2. Choose "Market Data API" plan ($10/month)
3. Get API key from dashboard
4. Add to Railway: Settings → Environment Variables → Add Variable

#### 2. Perplexity AI ($20/month - Recommended)
```
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

**How to get**:
1. Sign up at https://www.perplexity.ai/
2. Subscribe to Pro plan ($20/month)
3. Get API key from https://www.perplexity.ai/settings/api
4. Add to Railway: Settings → Environment Variables → Add Variable

**Note**: If you don't add this, system falls back to NewsAPI (lower quality)

### Optional (Existing - Keep if You Have Them)
```
NEWSAPI_KEY=your_newsapi_key_here          # Fallback sentiment source
X_API_KEY=your_x_api_key_here              # Twitter trends (supplementary)
X_API_SECRET=your_x_api_secret_here
ALPHA_VANTAGE_API_KEY=your_av_key_here     # Future use
FRED_API_KEY=your_fred_key_here            # Economic data (future)
```

### Optional AI Enhancement (Advanced Users)
```
OPENAI_API_KEY=your_openai_key_here        # For GPT-4 analysis layer
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here  # For Claude analysis layer
```

**Note**: AI enhancement adds an extra intelligence layer on top of your mathematical analysis. Not required for core functionality.

---

## 📋 Step-by-Step Railway Setup

### 1. **Get Your API Keys**

| Service | Price | Purpose | Required | Link |
|---------|-------|---------|----------|------|
| **Tradier** | $10/mo | Market data | ✅ YES | https://tradier.com/products/market-data |
| **Perplexity** | $20/mo | Sentiment | ⚠️ Recommended | https://www.perplexity.ai/ |
| NewsAPI | Free tier | Sentiment fallback | ❌ Optional | https://newsapi.org/ |

### 2. **Add to Railway**

1. Go to your Railway project dashboard
2. Click on your deployment
3. Go to **Settings** tab
4. Scroll to **Environment Variables** section
5. Click **Add Variable**
6. Add each key-value pair:
   ```
   Variable Name: TRADIER_API_KEY
   Value: [paste your key]
   ```
7. Click **Add** for each variable
8. Railway will automatically redeploy with new variables

### 3. **Verify Setup**

After deployment, check logs for:
```
✅ Success indicators:
💰 Tradier API ENABLED - Reliable market data active
🧠 AI Features ENABLED - Perplexity: True
🚀 Starting ADAPTIVE continuous option monitoring system

❌ Warning indicators:
⚠️  Tradier API DISABLED - Using yfinance fallback
```

---

## 🎯 How It Works Now

### **Market Hours (9:30 AM - 4:00 PM ET)**
```
Every 30 seconds:
  ├─ Tradier API: Get real-time prices
  ├─ Calculate Greeks
  └─ Check sell signals

Every 1 hour:
  ├─ Perplexity: Comprehensive sentiment analysis
  └─ Update ITM probabilities
```

### **After Hours (4:00 PM - 9:30 AM ET)**
```
Every 5 minutes:
  ├─ Use cached EOD data (from 4:00 PM close)
  ├─ Calculate Greeks
  └─ Check sell signals

No sentiment updates (use last market close data)
```

### **Your Workflow**
```
Monday 8 PM (you):
  └─ Run "Smart Picks" command in Slack

Bot (using Monday 4 PM EOD cache):
  ├─ Analyzes 20 liquid symbols
  ├─ 20K Monte Carlo simulations per option
  ├─ Perplexity sentiment analysis
  └─ Returns top 8 picks

Tuesday 9 AM (you):
  └─ Place order for TSLA $430 call

Bot (auto-monitoring):
  ├─ 30-second price checks
  ├─ Hourly sentiment updates
  └─ Sends Slack alert when to sell
```

---

## 💡 Cost Breakdown

| Item | Cost | What You Get |
|------|------|--------------|
| **Tradier** | $10/mo | - 120 requests/min<br>- Real-time quotes<br>- ORATS Greeks<br>- EOD caching |
| **Perplexity** | $20/mo | - 600 requests/day<br>- AI sentiment analysis<br>- Multi-source aggregation<br>- Citation-backed insights |
| **Railway** | $5/mo | - 24/7 hosting<br>- Auto-restarts<br>- Easy deployment |
| **TOTAL** | **$35/mo** | Professional-grade options trading bot |

**Alternative (Budget)**:
- Skip Perplexity ($20 saved)
- Uses NewsAPI fallback (lower quality)
- Total: $15/month

---

## 🔍 Troubleshooting

### "Tradier API error 401"
- **Cause**: Invalid API key
- **Fix**: Double-check key in Railway env vars, regenerate if needed

### "Perplexity rate limit hit"
- **Cause**: >600 requests/day or >1 request/second
- **Fix**: System auto-falls back to NewsAPI, no action needed

### "Using yfinance fallback"
- **Cause**: Tradier key not set or invalid
- **Fix**: Add `TRADIER_API_KEY` to Railway environment variables

### "After hours - using cached EOD"
- **Status**: Normal behavior
- **Note**: Bot uses 4 PM close data until next market open

---

## 📊 Monitoring Your Bot

### Check Logs in Railway
```bash
# Good indicators:
✅ Tradier: Got quotes for 5 symbols
🧠 Updating sentiment for 2 symbols
✅ Perplexity sentiment for TSLA: 0.45
📈 MARKET OPEN - Monitoring 3 options

# After hours (normal):
🌙 AFTER HOURS - Monitoring 3 options
📦 Using cached EOD for TSLA: $420.50
```

### Slack Commands to Test
```
Smart Picks              # Should return 8 picks with scores
Pick TSLA $430          # Should add to monitoring
Status                  # Should show monitored positions
```

---

## 🚀 Next Steps

1. **Deploy**: Push code to Railway (auto-deploys on git push)
2. **Add keys**: Set environment variables in Railway dashboard
3. **Test**: Run `Smart Picks` in Slack during market hours
4. **Monitor**: Check Railway logs for success indicators
5. **Trade**: Use recommendations for swing trades (7-30 days)

---

## 📞 Support

- **Tradier support**: techsupport@tradier.com
- **Perplexity docs**: https://docs.perplexity.ai/
- **Railway docs**: https://docs.railway.app/

---

**🎉 Your bot is now enterprise-grade! Enjoy reliable data and better sell signals.**
