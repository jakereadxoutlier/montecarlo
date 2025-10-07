#!/usr/bin/env python3
"""
Standalone Slack App that runs independently and connects to StockFlow MCP server.
This keeps the Slack connection alive while making MCP calls as needed.
CLEAN VERSION - ALL OLD HANDLERS REMOVED
"""
import asyncio
import json
import os
import logging
import re
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

# MCP Server parameters
MCP_SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=["/app/stockflow.py"]
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

async def _handle_smart_picks_internal(message, say):
    """Handle 'Smart Picks' command for optimal risk/reward options ≤30 days."""
    try:
        text = message['text']
        logger.info(f"Received Smart Picks command: {text}")

        await say(f"🧠 **Smart Picks Analysis Starting...**\n🔍 Finding optimal risk/reward options ≤30 days using advanced techniques...\n⏳ This may take 30-60 seconds to analyze all Fortune 500 options.")

        # Call ENHANCED MCP tool for Smart Picks analysis
        result = await call_mcp_tool(
            'smart_picks_optimal_options_enhanced',
            {
                'max_days_to_expiry': 30,
                'target_profit_potential': 0.15,  # 15% target profit potential (adaptive)
                'target_probability': 0.45,       # 45% target ITM probability (adaptive)
                'target_risk_level': 6,           # Medium risk tolerance (adaptive)
                'max_results': 8,                 # Top 8 picks
                'always_show_results': True       # Always show best available options
            }
        )

        if result and result.get('success'):
            analysis = result['data']['smart_picks_analysis']
            optimal_options = analysis['optimal_options']
            summary = analysis['summary_stats']

            if optimal_options:
                market_context = analysis.get('market_context', {})

                response = f"**🎯 ENHANCED Smart Picks - Institutional Grade Analysis**\n\n"

                # Market Context
                response += f"**📈 Market Context:**\n"
                response += f"- Regime: **{market_context.get('regime', 'Unknown').title()}** "
                response += f"({market_context.get('regime_confidence', 0):.0%} confidence)\n"
                response += f"- {market_context.get('message', 'Analysis complete')}\n\n"

                # Enhanced Analysis Summary
                response += f"**🔬 Institutional Analysis Summary:**\n"
                response += f"- Options Analyzed: **{analysis['total_options_analyzed']:,}**\n"
                response += f"- Ideal Criteria Met: **{analysis.get('ideal_criteria_met', 0)}** options\n"
                response += f"- Sentiment Analyzed: **{market_context.get('sentiment_analyzed_symbols', 0)}** symbols\n"
                response += f"- Options Flow Analyzed: **{market_context.get('flow_analyzed_symbols', 0)}** symbols\n"
                response += f"- Processing Time: **{analysis.get('performance_metrics', {}).get('processing_time_seconds', 0):.1f}s**\n\n"

                # Results Summary
                response += f"**📊 Results Summary:**\n"
                response += f"- Avg ITM Probability: **{summary['average_itm_probability']:.1%}**\n"
                response += f"- Avg Profit Potential: **{summary['average_profit_potential']:.1%}**\n"
                response += f"- Avg Risk Level: **{summary['average_risk_level']:.1f}/10**\n"
                response += f"- Avg Days to Expiry: **{summary['average_days_to_expiration']:.0f}**\n\n"

                response += f"**🏆 Top {len(optimal_options)} Options Found:**\n"

                for i, opt in enumerate(optimal_options[:6], 1):  # Show top 6
                    category = opt.get('category', 'acceptable')
                    category_emoji = "🎯" if category == "ideal" else "📈" if category == "adapted" else "⚖️"

                    response += f"\n{i}. **{opt['symbol']} ${opt['strike']} Call** {category_emoji} (Exp: {opt['expiration']})\n"
                    response += f"   • **Enhanced Score: {opt['composite_score']:.4f}** (Higher = Better)\n"
                    response += f"   • **ITM Probability: {opt['itm_probability']:.1%}**"

                    # Show sentiment adjustment if significant
                    sentiment_adj = opt.get('sentiment_adjustment', 0)
                    if abs(sentiment_adj) >= 0.01:
                        response += f" (Sentiment +{sentiment_adj:+.1%})"
                    response += "\n"

                    response += f"   • **Profit Potential: {opt['profit_potential']:.1%}**\n"
                    response += f"   • **Risk Level: {opt['risk_level']:.1f}/10**\n"
                    response += f"   • **Days to Expiry: {opt['days_to_expiration']}**\n"
                    response += f"   • **Option Price: ${opt['option_price']:.2f}**\n"
                    response += f"   • **Volume: {opt['volume']:,}**"

                    # Show flow sentiment if available
                    flow_data = opt.get('flow_data', {})
                    if flow_data and 'flow_sentiment' in flow_data:
                        flow_sentiment = flow_data['flow_sentiment']
                        if flow_sentiment != 'neutral':
                            flow_emoji = "🟢" if flow_sentiment == 'bullish' else "🔴"
                            response += f" | Flow: {flow_emoji}{flow_sentiment.title()}"
                    response += "\n"

                if len(optimal_options) > 6:
                    response += f"\n... and {len(optimal_options) - 6} more options\n"

                response += f"\n💡 **To select an option:** Reply with `Pick [SYMBOL] $[STRIKE]`\n"
                response += f"🎯 **Example:** `Pick {optimal_options[0]['symbol']} ${optimal_options[0]['strike']}`\n\n"
                response += f"🏦 **Institutional-Grade Features (NEW):**\n"
                response += f"- Real-time Market Sentiment Analysis\n"
                response += f"- Market Regime Detection & Adaptive Criteria\n"
                response += f"- Options Flow & Unusual Activity Detection\n"
                response += f"- 7 Novel Analysis Techniques (Fractal Volatility, Gamma Squeeze, etc.)\n"
                response += f"- **ALWAYS Shows Best Available Options** (Never \"None Found\")\n"
                response += f"- Enhanced Composite Scoring with Sentiment Multipliers"

                await say(response)
            else:
                # This should rarely happen with enhanced version since it always shows results
                market_context = analysis.get('market_context', {})
                response = f"⚠️ **No Options Data Available**\n\n"
                response += f"**Market Status:**\n"
                response += f"- Regime: {market_context.get('regime', 'Unknown').title()}\n"
                response += f"- {market_context.get('message', 'Markets may be closed or data unavailable')}\n\n"
                response += f"**This is unusual with Enhanced Smart Picks** - we normally always show best available options.\n"
                response += f"Please try again in a few minutes or when markets are open."
                await say(response)
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
            await say(f"❌ **Enhanced Smart Picks Analysis Failed**\n\n**Error:** {error_msg}\n\n**Possible causes:**\n- Markets closed (weekends/holidays)\n- API rate limits or timeouts\n- Network connectivity issues\n\n**Try:**\n- Wait a few minutes and try again\n- Try during market hours (9:30 AM - 4:00 PM ET)\n- Use `Pick [SYMBOL] $[STRIKE]` for individual analysis\n\n*Note: Enhanced Smart Picks includes real-time sentiment, market regime detection, and options flow analysis.*")

    except Exception as e:
        logger.error(f"Error handling Smart Picks command: {e}")
        await say("Sorry, I encountered an error with Smart Picks analysis. Please try again later.")

def setup_message_handlers(app):
    """Setup all Slack message handlers after app initialization - CLEAN VERSION."""

    # SMART PICKS MUST BE FIRST to avoid regex conflicts!
    @app.message(re.compile(r'^smart\s*picks?', re.IGNORECASE))
    async def handle_smart_picks_regex(message, say):
        await _handle_smart_picks_internal(message, say)

    # More specific Pick command regex that WON'T match "smart picks"
    @app.message(re.compile(r'^(pick|analyze|buy)\s+[A-Z]{2,5}\s+\$?\d+(\.\d+)?$', re.IGNORECASE))
    async def handle_pick_command(message, say):
        """Handle pick command with specific pattern."""
        try:
            text = message['text']
            logger.info(f"Received pick command: {text}")

            # Parse pick command
            pick_pattern = r'(pick|analyze|buy)\s+([A-Z]{2,5})\s+\$?(\d+(?:\.\d+)?)'
            match = re.search(pick_pattern, text, re.IGNORECASE)

            if not match:
                await say("Please use format: `Pick [SYMBOL] $[STRIKE]` (e.g., `Pick TSLA $430`)")
                return

            symbol = match.group(2).upper()
            strike = float(match.group(3))

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
                auto_monitoring = analysis.get('auto_monitoring', {})

                # Format response
                response = f"**{symbol} ${strike} Call Analysis**\n\n"
                response += f"**Current Status:**\n"
                response += f"- Stock Price: ${option_data['current_price']:.2f}\n"
                response += f"- Option Price: ${option_data['option_price']:.2f}\n"
                response += f"- ITM Probability: {option_data['itm_probability']:.1%}\n"
                response += f"- Days to Expiry: {option_data['time_to_expiration_days']:.0f}\n\n"

                response += f"**Recommendation: {advice['recommendation']}**\n"
                response += f"Confidence: {advice['confidence']} ({advice['net_score']:+d} score)\n\n"

                # Show monitoring status
                if auto_monitoring.get('enabled'):
                    response += f"✅ **AUTO-MONITORING ENABLED**\n"
                    response += f"📊 Total positions monitored: {auto_monitoring.get('total_monitored', 0)}\n"
                    response += f"🔔 You'll receive sell alerts when profit targets are hit!\n\n"

                response += f"**Key Factors:**\n"
                for factor in advice['factors'][:6]:
                    response += f"- {factor}\n"

                if len(advice['factors']) > 6:
                    response += f"- ... and {len(advice['factors']) - 6} more factors\n"

                # Add monitoring instructions for non-buy recommendations
                if not auto_monitoring.get('enabled'):
                    response += f"\n💡 **Tip:** Only BUY recommendations are auto-monitored for sell alerts."

                await say(response)
            else:
                await say(f"Error analyzing {symbol} ${strike}. Please try again.")

        except Exception as e:
            logger.error(f"Error handling pick command: {e}")
            await say("Sorry, I encountered an error processing your request. Please try again.")

    @app.message()
    async def handle_default_message(message, say):
        """Handle all other messages - SINGLE CLEAN HANDLER."""
        text = message.get('text', '').strip().lower()

        # Help command
        if any(word in text for word in ['help', 'commands', 'usage']):
            help_text = """**🎯 StockFlow Bot - Institutional Grade Options Analysis**

**📈 Core Commands:**
- `Pick [SYMBOL] $[STRIKE]` - Get buy/sell advice + auto-monitoring for sell alerts
- `Analyze [SYMBOL] $[STRIKE]` - Same as Pick
- `Buy [SYMBOL] $[STRIKE]` - Same as Pick

**🧠 Advanced Commands:**
- `Smart Picks` - Find optimal risk/reward options ≤30 days (INSTITUTIONAL GRADE!)

**📊 Monitoring Commands:**
- `Status` or `Positions` - Check monitored positions
- `Sold [SYMBOL] $[STRIKE]` - Mark option as sold (stops alerts)
- `Stop` or `Cancel` - Stop all monitoring
- `Start Monitoring` - Resume monitoring

**🔧 System Commands:**
- `Help` - Show this help message

**📋 Examples:**
- `Pick TSLA $430` - Analyzes TSLA $430 call + starts monitoring
- `Smart Picks` - Shows top institutional-grade options
- `Status` - Check all monitored positions
- `Sold TSLA $430` - Mark TSLA $430 as sold (stops alerts)
- `Stop` - Stop monitoring all positions

**⚡ NEW: Institutional Features (Phase 2):**
- **Alpha Vantage Integration**: Real earnings calendar, market data
- **FRED Economic Data**: Government bond yields, economic indicators
- **5 Advanced Analytics**: Multi-scenario Monte Carlo, pattern recognition, volatility forecasting, event analysis, cross-asset correlation
- **Auto-monitoring**: BUY recommendations automatically monitored
- **Smart sell alerts**: Real-time profit target notifications

**🎯 Perfect for $1K+ trading capital with professional-grade analysis!**"""
            await say(help_text)

        # Smart Picks (fallback)
        elif 'smart picks' in text or 'smart pick' in text or 'smartpicks' in text or 'give me smart' in text:
            await _handle_smart_picks_internal(message, say)

        # Pick command (fallback for non-regex matches)
        elif re.match(r'^(pick|buy|analyze)\s+[A-Z]{2,5}\s+\$?\d+(\.\d+)?', text, re.IGNORECASE):
            await handle_pick_command(message, say)

        # Monitoring control commands
        elif any(word in text for word in ['cancel', 'stop', 'stop monitoring']):
            result = await call_mcp_tool('stop_continuous_monitoring', {})
            if result and result.get('success'):
                await say("🛑 **Monitoring Stopped**\n\nAll position monitoring has been stopped. You will no longer receive sell alerts.")
            else:
                await say("❌ Error stopping monitoring. Please try again.")

        elif any(word in text for word in ['status', 'monitoring status', 'positions']):
            result = await call_mcp_tool('list_selected_options', {})
            if result and result.get('success'):
                data = result['data']
                if data.get('selected_options'):
                    active_count = 0
                    sold_count = 0
                    position_details = ""

                    for opt_key, opt_data in data['selected_options'].items():
                        if opt_data.get('status') == 'sold':
                            sold_count += 1
                            position_details += f"• **{opt_data['symbol']} ${opt_data['strike']}** ✅ SOLD (Exp: {opt_data['expiration_date']})\n"
                            if opt_data.get('final_pnl'):
                                position_details += f"  P&L: {opt_data['final_pnl']} | Sold: {opt_data.get('sold_at', '')[:10]}\n"
                            else:
                                position_details += f"  Sold: {opt_data.get('sold_at', '')[:10]}\n"
                        else:
                            active_count += 1
                            position_details += f"• **{opt_data['symbol']} ${opt_data['strike']}** 🔔 MONITORING (Exp: {opt_data['expiration_date']})\n"
                            position_details += f"  Added: {opt_data['selected_at'][:10]} | Alerts sent: {opt_data.get('alerts_sent', 0)}\n"

                    response = f"**📊 Monitoring Status**\n\n"
                    response += f"**Active Monitoring:** {active_count} positions\n"
                    response += f"**Sold Positions:** {sold_count} completed\n"
                    response += f"**System Status:** {'✅ Active' if data.get('monitoring_active') else '❌ Stopped'}\n\n"
                    response += position_details

                    await say(response)
                else:
                    await say("📊 **No Active Positions**\n\nUse `Pick [SYMBOL] $[STRIKE]` to start monitoring an option.")
            else:
                await say("❌ Error getting monitoring status. Please try again.")

        # Start monitoring command
        elif any(word in text for word in ['start monitoring', 'resume monitoring']):
            result = await call_mcp_tool('start_continuous_monitoring', {})
            if result and result.get('success'):
                await say("✅ **Monitoring Started**\n\nContinuous monitoring is now active. You'll receive sell alerts when profit targets are hit!")
            else:
                await say("❌ Error starting monitoring. Please try again.")

        # Sold command - mark option as sold to stop alerts
        elif text.lower().startswith('sold '):
            # Parse sold command: "Sold TSLA $430"
            pattern = r'sold\s+([A-Za-z]+)\s*\$?(\d+\.?\d*)'
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                await say("Please use format: `Sold [SYMBOL] $[STRIKE]` (e.g., `Sold TSLA $430`)")
                return

            symbol = match.group(1).upper()
            strike = float(match.group(2))

            await say(f"Marking {symbol} ${strike} as sold and stopping alerts...")

            # Call MCP tool to mark as sold
            result = await call_mcp_tool(
                'mark_option_sold',
                {
                    'symbol': symbol,
                    'strike': strike,
                    'expiration_date': '2025-10-17'  # Default - will find any matching
                }
            )

            if result and result.get('success'):
                data = result['data']
                await say(f"✅ **{symbol} ${strike} Marked as SOLD**\n\n📊 Sell alerts stopped for this position.\n💰 Profit/Loss: {data.get('final_pnl', 'Not calculated')}\n\nGood trade! 🎉")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                await say(f"❌ Could not mark {symbol} ${strike} as sold: {error_msg}\n\nTry `Status` to see your monitored positions.")

        # Default response
        else:
            await say("I didn't understand that command. Type `help` to see available commands or try:\n- `Pick TSLA $430` (analyze & monitor)\n- `Smart Picks` (find opportunities)\n- `Status` (check positions)\n- `Sold TSLA $430` (mark as sold)\n- `Stop` (stop monitoring)")

async def main():
    """Start the Slack App."""
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        logger.error("Missing Slack tokens. Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in environment variables")
        logger.error(f"SLACK_BOT_TOKEN present: {'Yes' if SLACK_BOT_TOKEN else 'No'}")
        logger.error(f"SLACK_APP_TOKEN present: {'Yes' if SLACK_APP_TOKEN else 'No'}")
        return

    # Initialize Slack App with tokens
    logger.info("🔧 Initializing Monte Carlo Slack App...")
    app = AsyncApp(token=SLACK_BOT_TOKEN)

    # Register all message handlers
    setup_message_handlers(app)

    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    logger.info("🚀 Starting Monte Carlo Slack App...")
    logger.info("Ready to receive messages!")
    logger.info("Try: 'Smart Picks' or 'Pick TSLA $430'")

    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monte Carlo App stopped by user")
    except Exception as e:
        logger.error(f"Monte Carlo App error: {e}")