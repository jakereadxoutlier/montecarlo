# 🎯 Complete Smart Picks System Flow

## Overview

Your Smart Picks system is a **two-layer analysis engine** that combines mathematical probability modeling with optional AI-powered intelligence to find the best risk/reward options trades.

---

## 📊 LAYER 1: Mathematical Foundation (ALWAYS RUNS)

### **Step 1: Data Collection** (montecarlo_unified.py:4727-4850)

```python
When you run "Smart Picks" in Slack:

1. Analyzes 20 liquid symbols (LIQUID_OPTIONS_SYMBOLS):
   TSLA, AAPL, MSFT, NVDA, GOOGL, AMZN, META, etc.

2. For each symbol:
   ├─ Get current stock price (Polygon.io)
   ├─ Get available expirations within 30 days
   ├─ Fetch options chains for each expiration
   └─ Filter for OTM calls (strike > current price)

API Calls:
├─ 20 quotes = 20 calls
├─ 20 expirations = 20 calls
├─ ~3-5 chains per symbol = 60-100 calls
└─ TOTAL: ~140 calls (Polygon.io handles this easily)
```

### **Step 2: Options Filtering** (lines 5050-5150)

```python
For each option contract found:

Minimum Criteria:
├─ Volume ≥ 50 (liquid enough to trade)
├─ Implied Volatility ≥ 0.10 (10%)
├─ Bid > 0 and Ask > 0 (tradeable spread)
├─ OTM (Strike > Current Price)
└─ Days to expiration ≤ 30

This filters 1000s of contracts down to ~200-300 viable candidates
```

### **Step 3: Probability Analysis** (lines 1920-2055)

For each viable option:

#### **A. Monte Carlo Simulation (20,000 paths)**
```python
monte_carlo_itm_probability_enhanced():
├─ Simulates 20,000 possible price paths
├─ Uses: Current price, IV, time to expiration
├─ Models: Geometric Brownian Motion + jump diffusion
├─ Sentiment adjustment: ±20% based on news
└─ Output: ITM probability (0-100%)

Example:
TSLA at $420, $430 strike, 21 days, 30% IV, +0.05 sentiment boost
→ 52% probability of finishing ITM
```

#### **B. Black-Scholes Greeks Calculation** (lines 3425-3457)
```python
calculate_black_scholes_greeks():
├─ Delta: Price sensitivity (0-1 for calls)
├─ Gamma: Delta change rate
├─ Theta: Time decay per day
├─ Vega: Volatility sensitivity
└─ Rho: Interest rate sensitivity

Used for: Sell signal detection during monitoring
```

#### **C. 7 Novel Analysis Techniques** (lines 128-139)
```python
AdvancedOptionsEngine.analyze_with_novel_techniques():

1. Fractal Volatility: Multi-timeframe vol patterns
2. Gamma Squeeze Detection: Dealer hedging pressure
3. Options Flow Momentum: Unusual activity scoring
4. Market Maker Impact: MM positioning analysis
5. Cross-Asset Correlation: SPY/VIX relationship
6. Volatility Surface Analysis: IV skew patterns
7. Multi-Dimensional Monte Carlo: Enhanced probability model

Output: Additional confidence scores and risk adjustments
```

### **Step 4: Composite Scoring** (lines 2680-2760)

```python
For each option, calculate composite score (0-10):

Components:
├─ ITM Probability: 35% weight
│  └─ 0.70+ ITM = 10 points
│  └─ 0.45-0.70 = 5-9 points
│  └─ <0.45 = 0-4 points
│
├─ Profit Potential: 30% weight
│  └─ Expected gain if ITM
│  └─ (Strike - Current) / Current
│
├─ Risk Level: 20% weight (inverted)
│  └─ Based on: IV, moneyness, time decay
│  └─ Lower risk = higher score
│
├─ Time to Expiration: 10% weight
│  └─ Sweet spot: 14-21 days
│  └─ Too soon = theta decay risk
│  └─ Too far = capital tied up
│
└─ Advanced Bonuses: 5% weight
   ├─ Gamma squeeze potential
   ├─ Options flow momentum
   └─ Unusual volume

Final Score: 0-10 (top options ≥ 8.0)
```

### **Step 5: Ranking & Filtering** (lines 2760-2850)

```python
From ~200-300 analyzed options:

1. Remove duplicates (same symbol/expiration)
2. Filter: Score ≥ target (usually 7.5+)
3. Filter: ITM Prob ≥ 0.45 (45%)
4. Filter: Profit Potential ≥ 0.15 (15%)
5. Sort by: Composite score (descending)
6. Take top 20 options

Output: Top 20 mathematically optimal options
```

---

## 🧠 LAYER 2: AI Enhancement (OPTIONAL - If API Keys Configured)

### **When AI is Enabled** (lines 2681-2730)

```python
AI_ENABLED = bool(PERPLEXITY_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY)

If AI enabled:
  Take top 20 options from Layer 1
  ↓
  Enhance each with AI intelligence:
```

### **AI Enhancement Components:**

#### **A. Perplexity Sentiment Analysis** (lines 194-331)

```python
perplexity_client.analyze_sentiment(symbol):

Query: "Analyze current market sentiment for {symbol} in last 24-48 hours"

AI analyzes:
├─ Recent news (earnings, products, exec changes)
├─ Analyst upgrades/downgrades
├─ Social media (Twitter, Reddit, forums)
├─ Price action and momentum
└─ Unusual catalysts

Returns:
├─ Sentiment Score: -1.0 (bearish) to +1.0 (bullish)
├─ Sentiment Boost: ±20% ITM probability adjustment
├─ Key Factors: 3-5 bullet points
├─ Confidence: low/medium/high
└─ Citations: Source URLs

Example Output:
{
  "sentiment_score": 0.65,
  "boost": 0.13,  # +13% ITM probability boost
  "key_factors": [
    "Strong earnings beat with 25% revenue growth",
    "Morgan Stanley upgrade to overweight",
    "Reddit sentiment turned bullish post-earnings"
  ],
  "confidence": "high"
}

This boosts TSLA $430 call from 52% → 65% ITM probability
```

#### **B. LLM Analysis** (lines 236-333) - Optional

```python
If OPENAI_API_KEY or ANTHROPIC_API_KEY configured:

llm_client.analyze(prompt, context):

System Prompt:
"You are an elite options trader with 20 years at Goldman Sachs.
Analyze with institutional-grade insights. Provide conviction scores 1-10."

Context includes:
├─ Option details (symbol, strike, expiration)
├─ Mathematical analysis (ITM prob, Greeks, score)
├─ Sentiment data from Perplexity
├─ Market regime (bullish/bearish/choppy)
└─ Volume and open interest

LLM provides:
├─ Conviction Score: 1-10
├─ Key Insight: Human-readable summary
├─ Entry Guidance: "Buy at $2.50-$2.70"
├─ Exit Target: "Sell at $3.50-$4.00 (15-20 days)"
├─ Risk Warning: "Watch for Fed announcement Wednesday"
└─ Unusual Activity Flag: 🔥 if detected

Example:
"Conviction: 8/10 🔥
Strong bullish setup with earnings momentum and technical breakout.
Unusual call volume (3x average) suggests institutional accumulation.
Entry: $2.50-$2.70, Target: $3.80-$4.20 in 2-3 weeks."
```

#### **C. Market Intelligence** (lines 732-776) - Optional

```python
If Perplexity enabled:

get_ai_market_intelligence():

Queries:
1. "What are the top market-moving events this week?"
2. "What sectors are institutional investors buying?"
3. "Any unusual options activity in tech stocks?"

Provides macro context for option selection:
├─ Upcoming catalysts (FOMC, earnings, FDA)
├─ Sector rotation trends
├─ Institutional flow data
└─ Risk-on vs risk-off sentiment

Used to adjust scoring weights
```

### **AI Re-Ranking** (lines 2690-2710)

```python
After AI enhancement:

Take top 20 options + AI insights
↓
Re-rank based on:
├─ Original math score: 70% weight
├─ AI conviction: 20% weight
├─ Sentiment adjustment: 10% weight
↓
Final top 8-10 options with AI layer
```

---

## 📤 OUTPUT FORMAT

### **Without AI** (Math only):
```
🎯 TSLA $430 Call (10/31)
Score: 8.5/10
ITM Probability: 52%
Profit Potential: 25%
Risk Level: 6/10
Days to Expiry: 21

Greeks:
Delta: 0.48 | Gamma: 0.03 | Theta: -0.15
Vega: 0.12 | Current Price: $2.65

Entry: $2.50-$2.80
Target: 15-25% profit in 2-3 weeks
```

### **With AI Enhancement**:
```
🎯 TSLA $430 Call (10/31)
Score: 8.5/10
ITM Probability: 65% ⬆️ +13% (sentiment boost)
Profit Potential: 25%
Risk Level: 6/10
Days to Expiry: 21

🧠 AI Analysis:
• Conviction: 8/10 🔥 Unusual Activity
• Sentiment: Bullish (High confidence)
• Key Factors:
  - Strong earnings beat with 25% revenue growth
  - Morgan Stanley upgrade to overweight
  - Call volume 3.2x average (institutional flow)
• Entry: $2.50-$2.70 ideal
• Target: $3.80-$4.20 (15-20 days)
• Risk: Watch Fed announcement Wednesday

Greeks:
Delta: 0.48 | Gamma: 0.03 | Theta: -0.15
Current Price: $2.65

⚡ Smart Money Signal: Heavy call buying detected
```

---

## 🔄 CONTINUOUS MONITORING (After You Buy)

### **When You Execute a "Pick" Command** (lines 2099-2132)

```python
When you say "Pick TSLA $430":

1. Runs full analysis (Layer 1 + AI if enabled)
2. Shows recommendation (BUY/SELL/HOLD)
3. Automatically adds to monitoring
4. Starts continuous monitoring loop

Monitoring Status:
├─ Position: TSLA $430 Call 10/31
├─ Entry Price: $420.00 stock, $2.65 premium
├─ Current: Updated every 30s (market) or 5min (after-hours)
├─ ITM Probability: Tracked continuously
└─ Sell Signals: Multi-factor analysis
```

### **Monitoring Loop** (lines 3461-3695)

```python
ADAPTIVE MONITORING:

Market Hours (9:30 AM - 4:00 PM ET):
Every 30 seconds:
  ├─ Polygon.io: Get current stock price (15-min delayed OK)
  ├─ Calculate Greeks with new price
  ├─ Run 5K Monte Carlo simulation
  └─ Check sell signals

Every 1 hour:
  ├─ Perplexity: Update sentiment (if enabled)
  ├─ Adjust ITM probability
  └─ Check if sentiment-based sell trigger

After Hours (4:00 PM - 9:30 AM):
Every 5 minutes:
  ├─ Use cached EOD data (Polygon 4 PM snapshot)
  ├─ Calculate Greeks
  └─ Check sell signals (no sentiment updates)
```

### **Sell Signal Calculation** (lines 3707-3870)

```python
Multi-Factor Sell Score (0-10):

1. Profit Target (0-3 points):
   ├─ 40%+ profit = 3 points → STRONG SELL
   ├─ 25-40% profit = 2 points → TAKE PROFITS
   └─ 15-25% profit = 1 point → MONITOR

2. ITM Probability Change (0-2 points):
   ├─ Dropped >10% = 2 points → RED FLAG
   └─ Any decline = 1 point → CAUTION

3. Theta Decay Risk (0-2 points):
   ├─ <7 days to expiry = 2 points → URGENT
   └─ <14 days = 1 point → WATCH

4. Sentiment Reversal (0-2 points):
   ├─ Score < -0.3 = 2 points → BEARISH TURN
   └─ Score < 0 = 1 point → WEAKENING

5. Delta Weakness (0-1 point):
   └─ Delta < 0.4 = 1 point → LOSING MOMENTUM

Sell Score Thresholds:
├─ 7-10 points: "🚨 STRONG SELL - Take profits now"
├─ 5-6 points: "💰 TAKE PROFITS - Good exit opportunity"
├─ 3-4 points: "⚠️ MONITOR CLOSELY - Prepare for exit"
└─ 0-2 points: "✅ HOLD - Position still strong"
```

### **Slack Alert Example**:

```
🚨 SELL SIGNAL: TSLA $430 Call

Current Status:
├─ Stock Price: $445.00 (+5.95%)
├─ Option Value: $3.85 (Entry: $2.65)
├─ Profit: +45.3% 🎯
├─ Days Held: 14 days
└─ Days to Expiry: 7 days

Sell Score: 8/10 (STRONG SELL)

Factors:
✅ Profit target exceeded (45% > 40% target)
⚠️ Theta acceleration (7 days remaining)
⚠️ Sentiment weakening (-0.15 from +0.45)
✅ Delta still strong (0.72)

🎯 Recommendation: SELL NOW
Take your 45% profit before theta decay accelerates.

Historical ITM: 65% → Current: 78% ✅
```

---

## 🎯 COMPLETE USER WORKFLOW

### **Evening Planning (8 PM)**
```
You: "Smart Picks"

Bot analyzes:
├─ 20 symbols, 3-5 expirations each
├─ ~200-300 option contracts
├─ Math Layer: 20K Monte Carlo per option
├─ AI Layer: Perplexity sentiment + LLM analysis (if enabled)
└─ Returns: Top 8 optimal trades

Time: ~30-60 seconds

You review results, pick 2-3 trades for tomorrow
```

### **Morning Execution (9:30 AM)**
```
Market opens

You place orders:
├─ TSLA $430 Call @ $2.65 (limit order)
├─ NVDA $520 Call @ $8.20
└─ AAPL $185 Call @ $3.45

Bot auto-monitors all positions
```

### **During Trading Day**
```
Bot monitors every 30 seconds:
├─ Updates prices (Polygon.io, 15-min delayed)
├─ Recalculates probabilities
├─ Checks sell signals
└─ Sends Slack alerts when thresholds hit

You get alert at 2:15 PM:
"💰 TAKE PROFITS: TSLA $430 - Up 28%, sentiment still strong"

You decide: Hold for 40% target or take 28% now
```

### **Sell Decision (Your Choice)**
```
Option A: Follow bot recommendation, sell at 28%
Option B: Wait for 40% target
Option C: Partial sell (50% position at 28%, hold rest)

Bot continues monitoring until you:
├─ Mark as sold: "Sold TSLA $430"
├─ Stop monitoring: "Stop"
└─ Or option expires
```

---

## 💰 COST BREAKDOWN

### **Current Setup**:
```
Polygon.io Options Starter: $29/month
├─ Unlimited API calls
├─ Greeks & IV included
├─ 15-minute delayed (perfect for swing trades)
└─ All options data

Railway Hosting: $5/month
└─ 24/7 uptime

TOTAL BASE: $34/month
```

### **Optional Add-Ons**:
```
Perplexity Pro: $20/month
├─ AI-powered sentiment
├─ Multi-source aggregation
├─ Key factors extraction
└─ Better than NewsAPI

NewsAPI: Free tier or current plan
└─ Fallback if Perplexity not enabled

TOTAL WITH AI: $54/month
```

---

## 🚀 SUMMARY

**Your Smart Picks system is:**

1. **Two-Layer Intelligence**:
   - Math Foundation: 20K Monte Carlo, Greeks, 7 novel techniques
   - AI Enhancement: Perplexity sentiment + optional LLM analysis

2. **Comprehensive Scanning**:
   - 20 symbols × ~50-100 contracts each
   - Filters to top 8-10 optimal risk/reward plays

3. **Adaptive Monitoring**:
   - 30s during market hours
   - 5min after hours
   - Multi-factor sell signals

4. **Purpose-Built for Swing Trading**:
   - 7-30 day holds
   - 15-25% profit targets
   - Theta-aware exit timing

**Bottom Line**: Professional-grade options analysis for $34-54/month, optimized for evening analysis → morning execution → automated monitoring.
