#!/usr/bin/env python3
"""
PRODUCTION SERVER - ACTUALLY WORKS ON RAILWAY
"""
import asyncio
import logging
import os
import re
from datetime import datetime
from aiohttp import web
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production-server")

# Slack tokens
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')

# MCP Client Session (global)
mcp_session = None

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call MCP tool and return results"""
    global mcp_session

    try:
        if not mcp_session:
            # Initialize MCP connection
            server_params = StdioServerParameters(
                command="python3",
                args=["stockflow.py"]
            )

            transport = await stdio_client(server_params)
            async with ClientSession(transport[0], transport[1]) as session:
                mcp_session = session
                await mcp_session.initialize()

        # Call the tool
        result = await mcp_session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else "No result"
    except Exception as e:
        logger.error(f"MCP call failed: {e}")
        return f"Error: {str(e)}"

async def create_working_slack_app():
    """Create the ACTUAL working Slack app with all commands"""
    try:
        logger.info("🚀 Creating production Slack app...")

        if not (SLACK_BOT_TOKEN and SLACK_APP_TOKEN):
            logger.error("❌ Slack tokens missing!")
            return None

        # Create Slack app
        app = AsyncApp(token=SLACK_BOT_TOKEN)

        # SMART PICKS COMMAND
        @app.message(re.compile(r"^smart\s*picks?", re.IGNORECASE))
        async def handle_smart_picks(message, say):
            await say("🔍 Finding optimal call options... This may take 3-6 minutes...")

            try:
                result = await call_mcp_tool('find_smart_picks', {})
                await say(result)
            except Exception as e:
                await say(f"❌ Smart Picks failed: {str(e)}")

        # PICK COMMAND - Pattern: Pick TSLA $430
        @app.message(re.compile(r"^pick\s+([A-Z]{2,5})\s+\$?(\d+(?:\.\d+)?)", re.IGNORECASE))
        async def handle_pick(message, say):
            text = message['text']
            match = re.search(r"pick\s+([A-Z]{2,5})\s+\$?(\d+(?:\.\d+)?)", text, re.IGNORECASE)

            if match:
                symbol = match.group(1).upper()
                strike = float(match.group(2))

                await say(f"📊 Analyzing {symbol} ${strike} call option...")

                try:
                    result = await call_mcp_tool('analyze_specific_otm_call', {
                        'symbol': symbol,
                        'strike_price': strike,
                        'days_to_expiry': 30
                    })
                    await say(result)
                except Exception as e:
                    await say(f"❌ Analysis failed: {str(e)}")

        # STATUS COMMAND
        @app.message(re.compile(r"^status", re.IGNORECASE))
        async def handle_status(message, say):
            try:
                result = await call_mcp_tool('list_selected_options', {})
                await say(result)
            except Exception as e:
                await say(f"❌ Status check failed: {str(e)}")

        # HELP COMMAND
        @app.message(re.compile(r"^help", re.IGNORECASE))
        async def handle_help(message, say):
            help_text = """
📚 **Available Commands:**

• **Smart Picks** - Find optimal call options
• **Pick [SYMBOL] $[STRIKE]** - Analyze specific option (e.g., Pick TSLA $430)
• **Status** - View monitored positions
• **Help** - Show this message

💡 Example: `Pick NVDA $145`
"""
            await say(help_text)

        # TEST COMMAND
        @app.message("test")
        async def handle_test(message, say):
            await say("✅ Bot is working! Railway + Slack connection successful!")

        # Create Socket Mode handler
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)

        logger.info("✅ Production Slack app created with all commands")

        # Start as background task (non-blocking)
        asyncio.create_task(handler.start_async())

        logger.info("🎯 Slack Socket Mode started in background!")
        return True

    except Exception as e:
        logger.error(f"💥 Slack app creation failed: {e}")
        return False

async def create_http_server():
    """Create HTTP server for Railway health checks"""
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "service": "StockFlow Bot Production",
            "timestamp": str(datetime.now())
        })

    async def root(request):
        return web.json_response({
            "message": "StockFlow Bot is running!",
            "commands": ["smart picks", "pick TSLA $430", "status", "help"],
            "status": "active"
        })

    app = web.Application()
    app.router.add_get('/', root)
    app.router.add_get('/health', health)

    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🌐 Starting HTTP server on port {port}")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    logger.info("✅ HTTP server started successfully")
    return runner

async def main():
    """Main production server"""
    logger.info("🎯 StockFlow Production Server Starting...")
    logger.info("📊 Features: Smart Picks, Pick Analysis, Monitoring")

    try:
        # Start HTTP server first (for Railway)
        http_runner = await create_http_server()

        # Wait a bit for HTTP to stabilize
        await asyncio.sleep(2)

        # Create and start Slack app
        slack_success = await create_working_slack_app()

        if not slack_success:
            logger.error("⚠️ Slack app failed to start, but HTTP server is running")

        # Keep running with health checks
        while True:
            logger.info("🟢 Production server running...")
            await asyncio.sleep(60)  # Every minute

    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())