#!/usr/bin/env python3
"""
Clean server - add Slack step by step
"""
import asyncio
import logging
import os
from datetime import datetime
from aiohttp import web

# Simple logging (no file logging) - FIRST!
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clean-server")

# Test 1: Add minimal Slack imports only
try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    import dotenv
    dotenv.load_dotenv()

    SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
    SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
    slack_imports_ok = True
    logger.info("✅ Slack imports successful")
except Exception as e:
    slack_imports_ok = False
    logger.error(f"❌ Slack imports failed: {e}")

async def health(request):
    return web.json_response({
        "status": "healthy",
        "service": "StockFlow Clean Test",
        "uptime": str(datetime.now())
    })

async def root(request):
    return web.json_response({
        "message": "Clean server is running!",
        "slack_imports": slack_imports_ok,
        "tokens_present": bool(SLACK_BOT_TOKEN and SLACK_APP_TOKEN) if slack_imports_ok else False
    })

async def main():
    """Clean server - exact copy of working minimal pattern"""
    logger.info("🚀 Starting CLEAN server...")

    app = web.Application()
    app.router.add_get('/', root)
    app.router.add_get('/health', health)

    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🌐 Starting on port {port}")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    logger.info("✅ Clean server started successfully")

    # Keep running forever
    while True:
        logger.info("🟢 Clean server still running...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Clean server stopped")
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        exit(1)