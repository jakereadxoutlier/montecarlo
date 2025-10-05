#!/usr/bin/env python3
"""
Standalone Slack App that runs independently and connects to StockFlow MCP server.
This keeps the Slack connection alive while making MCP calls as needed.
"""
import asyncio
import json
import os
import logging
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Slack configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("standalone-slack-app")

# Initialize Slack App
app = AsyncApp(token=SLACK_BOT_TOKEN)

# MCP Server parameters
MCP_SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
)

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Helper function to call MCP tools."""
    try:
        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

                if result.content and hasattr(result.content[0], 'text'):
                    return json.loads(result.content[0].text)
                return None
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name}: {e}")
        return None

@app.message("help")
async def handle_help(message, say):
    """Handle help command."""
    help_text = """**StockFlow Bot Commands:**

🎯 **Core Commands:**
- `Pick [SYMBOL] $[STRIKE]` - Get buy/sell advice for an option (auto-monitors for sell alerts)
- `Analyze [SYMBOL] $[STRIKE]` - Same as Pick
- `Buy [SYMBOL] $[STRIKE]` - Same as Pick
- `Options for [MM/DD/YYYY]` - Find best options for a specific expiration date

🧠 **Advanced Commands:**
- `Smart Picks` - Find optimal risk/reward options ≤30 days (THE HACK!)

⚙️ **Control Commands:**
- `Cancel` or `Stop` - Stop monitoring
- `Help` - Show this help message

**Examples:**
- `Pick TSLA $430` - Analyzes and starts monitoring
- `Smart Picks` - Shows optimal high-probability, high-profit options
- `Options for 10/24/2025` - Shows best options expiring that Friday
- `Analyze AAPL $200`

🚀 **Advanced Analysis Features:**
- 7 Novel Analysis Techniques (Fractal Volatility, Gamma Squeeze, etc.)
- ITM probability (20K Monte Carlo + Advanced Monte Carlo)
- Greek analysis (Delta, Gamma, Theta, Vega)
- Market sentiment and news analysis
- Composite risk/reward scoring
- Automatic sell alert notifications"""

    await say(help_text)

@app.message("options for")
async def handle_options_for_date(message, say):
    """Handle 'Options for [date]' command."""
    try:
        text = message['text']
        logger.info(f"Received options for date command: {text}")

        # Parse date from message
        import re
        date_pattern = r'options for (\d{1,2})/(\d{1,2})/(\d{4})'
        match = re.search(date_pattern, text.lower())

        if not match:
            await say("Please use format: `Options for MM/DD/YYYY` (e.g., `Options for 10/24/2025`)")
            return

        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))

        # Format date
        date_str = f"{year}-{month:02d}-{day:02d}"

        await say(f"🔍 Finding best call options for {month}/{day}/{year}... Please wait.")

        # Call MCP tool
        result = await call_mcp_tool(
            'analyze_fortune500_otm_calls',
            {
                'expiration_date': date_str,
                'symbols': ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'V', 'MA', 'JPM'],
                'min_volume': 25,
                'min_iv': 0.15,
                'probability_threshold': 0.35,
                'alert_threshold': 0.65
            }
        )

        if result and result.get('success'):
            analysis = result['data']
            top_options = analysis['top_10_otm_calls'][:5]  # Show top 5

            if top_options:
                response = f"**📈 Best Call Options for {month}/{day}/{year}**\n\n"
                response += f"**Top {len(top_options)} Opportunities:**\n"

                for i, opt in enumerate(top_options, 1):
                    response += f"\n{i}. **{opt['symbol']} ${opt['strike']} Call**\n"
                    response += f"   - ITM Probability: {opt['itm_probability']:.1%}\n"
                    response += f"   - Current Price: ${opt['current_price']:.2f}\n"
                    response += f"   - Option Price: ${opt['option_price']:.2f}\n"
                    response += f"   - Volume: {opt['volume']:,}\n"
                    response += f"   - Days to Expiry: {opt['days_to_expiration']}\n"

                response += f"\n💡 **To select an option:** Reply with `Pick [SYMBOL] $[STRIKE]`\n"
                response += f"📊 Example: `Pick {top_options[0]['symbol']} ${top_options[0]['strike']}`"

                await say(response)
            else:
                await say(f"No suitable call options found for {month}/{day}/{year}. Try a different expiration date.")
        else:
            await say(f"Error analyzing options for {month}/{day}/{year}. Please try again later.")

    except Exception as e:
        logger.error(f"Error handling options for date: {e}")
        await say("Sorry, I encountered an error processing your request. Please try again.")

@app.message("cancel")
async def handle_cancel_monitoring(message, say):
    """Handle cancel monitoring command."""
    try:
        text = message['text']
        logger.info(f"Received cancel command: {text}")

        # Call MCP tool to stop monitoring
        result = await call_mcp_tool('stop_continuous_monitoring', {})

        if result and result.get('success'):
            await say("✅ **Monitoring Cancelled**\n📊 All option monitoring has been stopped.\n💡 Use `Pick [SYMBOL] $[STRIKE]` to start monitoring new options.")
        else:
            await say("❌ **Error stopping monitoring**\nPlease try again or check if monitoring was already stopped.")

    except Exception as e:
        logger.error(f"Error handling cancel command: {e}")
        await say("Sorry, I encountered an error cancelling monitoring. Please try again.")

@app.message("stop")
async def handle_stop_monitoring(message, say):
    """Handle stop monitoring command."""
    await handle_cancel_monitoring(message, say)

@app.message("smart picks")
async def handle_smart_picks_command(message, say):
    """Handle 'Smart Picks' command for optimal risk/reward options ≤30 days."""
    await _handle_smart_picks_internal(message, say)

@app.message("smart pick")
async def handle_smart_pick_command(message, say):
    """Handle 'Smart Pick' command (singular)."""
    await _handle_smart_picks_internal(message, say)

@app.message("smartpicks")
async def handle_smartpicks_command(message, say):
    """Handle 'smartpicks' command (no space)."""
    await _handle_smart_picks_internal(message, say)

async def _handle_smart_picks_internal(message, say):
    """Handle 'Smart Picks' command for optimal risk/reward options ≤30 days."""
    try:
        text = message['text']
        logger.info(f"Received Smart Picks command: {text}")

        await say(f"🧠 **Smart Picks Analysis Starting...**\n🔍 Finding optimal risk/reward options ≤30 days using advanced techniques...\n⏳ This may take 30-60 seconds to analyze all Fortune 500 options.")

        # Call MCP tool for Smart Picks analysis
        result = await call_mcp_tool(
            'smart_picks_optimal_options',
            {
                'max_days_to_expiry': 30,
                'min_profit_potential': 0.15,  # 15% minimum profit potential
                'min_probability': 0.45,       # 45% minimum ITM probability
                'max_risk_level': 6,           # Medium risk tolerance
                'max_results': 8               # Top 8 picks
            }
        )

        if result and result.get('success'):
            analysis = result['data']['smart_picks_analysis']
            optimal_options = analysis['optimal_options']
            summary = analysis['summary_stats']

            if optimal_options:
                response = f"**🎯 Smart Picks - Optimal Risk/Reward Balance**\n\n"
                response += f"**📊 Analysis Summary:**\n"
                response += f"- Options Analyzed: {analysis['total_options_found']:,}\n"
                response += f"- Avg ITM Probability: {summary['average_itm_probability']:.1%}\n"
                response += f"- Avg Profit Potential: {summary['average_profit_potential']:.1%}\n"
                response += f"- Avg Risk Level: {summary['average_risk_level']:.1f}/10\n"
                response += f"- Avg Days to Expiry: {summary['average_days_to_expiration']:.0f}\n\n"

                response += f"**🏆 Top {len(optimal_options)} Optimal Options:**\n"

                for i, opt in enumerate(optimal_options[:6], 1):  # Show top 6
                    response += f"\n{i}. **{opt['symbol']} ${opt['strike']} Call** (Exp: {opt['expiration']})\n"
                    response += f"   • Composite Score: **{opt['composite_score']:.4f}** (Higher = Better)\n"
                    response += f"   • ITM Probability: **{opt['itm_probability']:.1%}**\n"
                    response += f"   • Profit Potential: **{opt['profit_potential']:.1%}**\n"
                    response += f"   • Risk Level: **{opt['risk_level']:.1f}/10**\n"
                    response += f"   • Days to Expiry: **{opt['days_to_expiration']}**\n"
                    response += f"   • Option Price: **${opt['option_price']:.2f}**\n"
                    response += f"   • Volume: **{opt['volume']:,}**\n"

                if len(optimal_options) > 6:
                    response += f"\n... and {len(optimal_options) - 6} more options\n"

                response += f"\n💡 **To select an option:** Reply with `Pick [SYMBOL] $[STRIKE]`\n"
                response += f"🎯 **Example:** `Pick {optimal_options[0]['symbol']} ${optimal_options[0]['strike']}`\n\n"
                response += f"🧬 **Advanced Analysis Features:**\n"
                response += f"- Fractal Volatility Analysis\n"
                response += f"- Gamma Squeeze Probability\n"
                response += f"- Options Flow Momentum\n"
                response += f"- Market Maker Impact Analysis\n"
                response += f"- Multi-Dimensional Risk Scoring"

                await say(response)
            else:
                await say(f"❌ **No Optimal Options Found**\n\nNo options met the Smart Picks criteria. Try adjusting parameters or try again later when market conditions improve.")
        else:
            await say(f"❌ **Smart Picks Analysis Failed**\n\nThere was an error running the advanced analysis. Please try again later.")

    except Exception as e:
        logger.error(f"Error handling Smart Picks command: {e}")
        await say("Sorry, I encountered an error with Smart Picks analysis. Please try again later.")

@app.message("pick")
async def handle_pick_command(message, say):
    """Handle pick command."""
    try:
        text = message['text']
        logger.info(f"Received pick command: {text}")

        # Parse pick command
        import re
        pick_pattern = r'pick\s+([a-z]{1,5})\s+\$?(\d+(?:\.\d+)?)'
        match = re.search(pick_pattern, text.lower())

        if not match:
            await say("Please use format: `Pick [SYMBOL] $[STRIKE]` (e.g., `Pick TSLA $430`)")
            return

        symbol = match.group(1).upper()
        strike = float(match.group(2))

        await say(f"Analyzing {symbol} ${strike} call... Please wait.")

        # Call MCP tool for real-time analysis
        result = await call_mcp_tool(
            'analyze_option_realtime',
            {
                'symbol': symbol,
                'strike': strike,
                'expiration_date': '2025-10-17'  # Default expiration
            }
        )

        if result and result.get('success'):
            analysis = result['data']
            option_data = analysis['option_data']
            advice = analysis['advice']

            # Format response
            response = f"**{symbol} ${strike} Call Analysis**\n\n"
            response += f"**Current Status:**\n"
            response += f"- Stock Price: ${option_data['current_price']:.2f}\n"
            response += f"- Option Price: ${option_data['option_price']:.2f}\n"
            response += f"- ITM Probability: {option_data['itm_probability']:.1%}\n"
            response += f"- Days to Expiry: {option_data['time_to_expiration_days']:.0f}\n\n"

            response += f"**Recommendation: {advice['recommendation']}**\n"
            response += f"Confidence: {advice['confidence']} ({advice['net_score']:+d} score)\n\n"

            response += f"**Key Factors:**\n"
            for factor in advice['factors'][:6]:
                response += f"- {factor}\n"

            if len(advice['factors']) > 6:
                response += f"- ... and {len(advice['factors']) - 6} more factors\n"

            await say(response)

            # Auto-select for monitoring
            select_result = await call_mcp_tool(
                'select_option_for_monitoring',
                {
                    'symbol': symbol,
                    'strike': strike,
                    'expiration_date': '2025-10-17',
                    'notes': f"Auto-selected from Slack Pick command"
                }
            )

            if select_result and select_result.get('success'):
                # Auto-start monitoring
                monitor_result = await call_mcp_tool('start_continuous_monitoring', {})

                if monitor_result and monitor_result.get('success'):
                    await say(f"✅ {symbol} ${strike} added to monitoring and auto-monitoring started!\n📊 You'll get alerts when to sell for optimal profit.")
                else:
                    await say(f"✅ {symbol} ${strike} added to monitoring list. Use MCP tools to start monitoring for sell alerts.")

        else:
            await say(f"Error analyzing {symbol} ${strike}. Please try again.")

    except Exception as e:
        logger.error(f"Error handling pick command: {e}")
        await say("Sorry, I encountered an error processing your request. Please try again.")

@app.message()
async def handle_default_message(message, say):
    """Handle all other messages."""
    text = message.get('text', '').strip().lower()

    if any(word in text for word in ['help', 'commands', 'usage']):
        await handle_help(message, say)
    elif 'options for' in text:
        await handle_options_for_date(message, say)
    elif 'smart picks' in text or 'smart pick' in text or 'smartpicks' in text or 'give me smart' in text:
        await _handle_smart_picks_internal(message, say)
    elif any(word in text for word in ['pick', 'buy', 'analyze']):
        await handle_pick_command(message, say)
    else:
        await say("I didn't understand that command. Type `help` to see available commands or try:\n- `Pick TSLA $430`\n- `Smart Picks`\n- `Options for 10/24/2025`")

async def main():
    """Start the Slack App."""
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        logger.error("Missing Slack tokens. Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env file")
        return

    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    logger.info("🚀 Starting StockFlow Slack App...")
    logger.info("Ready to receive messages!")
    logger.info("Try: 'Options for 10/24/2025' or 'Pick V $350'")

    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Slack App stopped by user")
    except Exception as e:
        logger.error(f"Slack App error: {e}")