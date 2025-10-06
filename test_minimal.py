#!/usr/bin/env python3
"""
Minimal test app to debug Railway deployment issues
"""
import asyncio
import logging
import os
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimal-test")

async def health(request):
    return web.json_response({"status": "ok", "message": "Minimal app running"})

async def root(request):
    return web.json_response({"message": "Hello Railway!", "port": os.environ.get('PORT', '8080')})

async def main():
    logger.info("🚀 Starting minimal test app...")

    app = web.Application()
    app.router.add_get('/', root)
    app.router.add_get('/health', health)

    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🌐 Starting on port {port}")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    logger.info("✅ Minimal app started successfully")

    # Keep running forever
    while True:
        logger.info("🟢 Still running...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Stopped by user")
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        exit(1)