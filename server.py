#!/usr/bin/env python3
"""
Production StockFlow Server - Railway compatible with HTTP + Socket Mode
"""
import asyncio
import logging
import signal
import sys
import time
import os
from datetime import datetime
from aiohttp import web
from standalone_slack_app import main as run_slack_app

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/stockflow-server.log", mode='a')
    ]
)
logger = logging.getLogger("stockflow-production-server")

async def create_http_server():
    """Create HTTP server for Railway (same pattern as working minimal app)"""
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "service": "StockFlow Bot",
            "uptime": str(datetime.now())
        })

    async def root(request):
        return web.json_response({
            "message": "StockFlow Bot is running!",
            "service": "Options Trading Slack Bot",
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

async def run_slack_in_background():
    """Run Slack app in background"""
    try:
        logger.info("🚀 Starting Slack Socket Mode app...")
        await run_slack_app()
    except Exception as e:
        logger.error(f"Slack app error: {e}")
        # Don't crash the whole server if Slack fails

async def main():
    """Production server main - HTTP server only, test Slack separately"""
    logger.info("🎯 StockFlow Production Server Starting...")

    try:
        # Start HTTP server first (for Railway)
        http_runner = await create_http_server()

        # DON'T start Slack app yet - test if HTTP server alone works
        logger.info("🚫 Slack Socket Mode disabled for testing")

        # Keep running with periodic health checks
        while True:
            logger.info("🟢 HTTP server still running...")
            await asyncio.sleep(60)  # Every minute for testing

    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 StockFlow Server stopped by user")
    except Exception as e:
        logger.error(f"🚨 Fatal startup error: {e}")
        sys.exit(1)