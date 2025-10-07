#!/usr/bin/env python3
"""
RAILWAY PRODUCTION SERVER - Uses existing standalone_slack_app.py
"""
import asyncio
import logging
import os
from datetime import datetime
from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("railway-server")

async def create_http_server():
    """Create HTTP server for Railway health checks"""
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "service": "StockFlow Bot",
            "timestamp": str(datetime.now())
        })

    async def root(request):
        return web.json_response({
            "message": "StockFlow Bot is running!",
            "commands": ["Smart Picks", "Pick [SYMBOL] $[STRIKE]", "Status", "Help"],
            "info": "Bot responds in Slack workspace"
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

    logger.info("✅ HTTP server started for Railway")
    return runner

async def start_slack_app():
    """Start the existing standalone Slack app"""
    try:
        logger.info("🚀 Starting Slack Socket Mode app...")

        # Import and run the existing standalone_slack_app
        from standalone_slack_app import main as run_slack_app

        # Run as background task so it doesn't block
        asyncio.create_task(run_slack_app())

        logger.info("✅ Slack app started in background")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start Slack app: {e}")
        return False

async def main():
    """Main entry point"""
    logger.info("🎯 StockFlow Railway Server Starting...")
    logger.info("📊 Using existing standalone_slack_app.py with all commands")

    try:
        # Start HTTP server first (for Railway)
        http_runner = await create_http_server()

        # Wait for HTTP to stabilize
        await asyncio.sleep(2)

        # Start the existing Slack app
        slack_success = await start_slack_app()

        if not slack_success:
            logger.warning("⚠️ Slack app failed, but HTTP server is running")

        # Keep running with periodic health checks
        while True:
            logger.info("🟢 Railway server running...")
            await asyncio.sleep(300)  # Every 5 minutes

    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())