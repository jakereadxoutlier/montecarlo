#!/usr/bin/env python3
"""
Clean server - no Slack imports at all
"""
import asyncio
import logging
import os
from datetime import datetime
from aiohttp import web

# Simple logging (no file logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clean-server")

async def health(request):
    return web.json_response({
        "status": "healthy",
        "service": "StockFlow Clean Test",
        "uptime": str(datetime.now())
    })

async def root(request):
    return web.json_response({
        "message": "Clean server is running!",
        "no_slack": True
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