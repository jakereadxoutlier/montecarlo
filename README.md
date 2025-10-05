# 🎯 StockFlow - Advanced Options Analysis Bot

**The ultimate Slack bot for institutional-grade options analysis with 7 novel techniques and Smart Picks algorithm.**

## 🚀 Key Features

- **Smart Picks**: AI-powered optimal risk/reward options scanner (THE HACK!)
- **7 Novel Analysis Techniques**: Fractal volatility, gamma squeeze, options flow momentum, market maker impact, cross-asset correlation, volatility surface reconstruction, multi-dimensional Monte Carlo
- **Real-time Analysis**: 20K Monte Carlo simulations + advanced probability calculations
- **Auto-monitoring**: Automatic sell alerts for optimal profit taking
- **Fortune 500 Coverage**: 100 most liquid stocks for instant execution
- **Production Ready**: Auto-restarts, health checks, 24/7 uptime

## 💬 Slack Commands

- `Smart Picks` - Find optimal risk/reward options ≤30 days
- `Pick TSLA $430` - Analyze specific option with buy/sell advice
- `Options for 10/24/2025` - Best options for specific expiration date
- `Help` - Show all commands

## 🛠 Quick Deployment (24/7 Auto-Updates)

**Option 1: Railway (Recommended - Free tier)**
```bash
./quick-deploy.sh  # Choose option 1
```

**Option 2: DigitalOcean ($5/month)**
```bash
./quick-deploy.sh  # Choose option 2
```

**Option 3: VPS with Docker**
```bash
docker-compose up -d
```

## 📋 Environment Variables Required

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
NEWSAPI_KEY=your-news-api-key
X_API_KEY=your-x-api-key
X_API_SECRET=your-x-api-secret
```

---

## 📈 Smart Picks Algorithm (The "Hack")

Finds the perfect balance between high probability and high profit by combining:
- **ITM Probability** (from 7 novel techniques)
- **Profit Potential** (volatility-based scenarios)
- **Risk Level** (composite 1-10 scale)
- **Time Decay Optimization** (14-21 day sweet spot)
- **Advanced Bonuses** (gamma squeeze, options flow momentum)

**Result**: Options that are both highly likely to profit AND highly profitable when they do.

---

## Original MCP Server Documentation

## Features

### Stock Data
- Real-time stock prices and key metrics
- Historical price data with OHLC values
- Company fundamentals and financial statements
- Market indicators and ratios

### Options Analysis
- Complete options chain data
- Greeks (delta, gamma, theta, vega)
- Volume and open interest tracking
- Options strategy analysis

## Installation

```bash
# Install dependencies
pip install mcp yfinance

# Clone the repository
git clone https://github.com/twolven/stockflow
cd stockflow
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/twolven/mcp-stockflow.git
cd mcp-stockflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add to your Claude configuration:
In your `claude-desktop-config.json`, add the following to the `mcpServers` section:

```json
{
    "mcpServers": {
        "stockflow": {
            "command": "python",
            "args": ["path/to/stockflow.py"]
        }
    }
}
```

Replace "path/to/stockflow.py" with the full path to where you saved the stockflow.py file.

## Usage Prompt for Claude

When working with Claude, you can use this prompt to help it understand the available tools:

"I've enabled the stockflow tools which give you access to stock market data. You can use these three main functions:

1. `get_stock_data` - Get comprehensive stock info:
```python
{
    "symbol": "AAPL",
    "include_financials": true,  # optional
    "include_analysis": true,    # optional
    "include_calendar": true     # optional
}
```

2. `get_historical_data` - Get price history and technical indicators:
```python
{
    "symbol": "AAPL",
    "period": "1y",        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    "interval": "1d",      # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    "prepost": false       # optional - include pre/post market data
}
```

3. `get_options_chain` - Get options data:
```python
{
    "symbol": "AAPL",
    "expiration_date": "2024-12-20",  # optional - uses nearest date if not specified
    "include_greeks": true            # optional
}
```

All responses include current price data, error handling, and comprehensive market information."

### Running the Server

```bash
python stockflow.py
```

### Using with MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["stockflow.py"]
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get current stock data
            result = await session.call_tool(
                "get-stock-data", 
                arguments={"symbol": "AAPL"}
            )
            
            # Get options chain
            options = await session.call_tool(
                "get-options-chain",
                arguments={
                    "symbol": "AAPL",
                    "expiration_date": "2024-12-20"
                }
            )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

## Available Tools

1. `get-stock-data`
   - Current price and volume
   - Market cap and P/E ratio
   - 52-week high/low

2. `get-historical-data`
   - OHLC prices
   - Configurable time periods
   - Volume data

3. `get-options-chain`
   - Calls and puts
   - Strike prices
   - Greeks and IV
   - Volume and open interest

## Available Resources

1. `company-info://{symbol}`
   - Company description
   - Sector and industry
   - Employee count
   - Website

2. `financials://{symbol}`
   - Income statement
   - Balance sheet
   - Cash flow statement

## Prompts

1. `analyze-options`
   - Options strategy analysis
   - Risk/reward evaluation
   - Market condition assessment

## Requirements

- Python 3.12+
- mcp
- yfinance

## Limitations

- Data is sourced from Yahoo Finance and may have delays
- Options data availability depends on market hours
- Rate limits apply based on Yahoo Finance API restrictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Todd Wolven - (https://github.com/twolven)

## Acknowledgments

## Acknowledgments

- Built with the Model Context Protocol (MCP) by Anthropic
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Developed for use with Anthropic's Claude
