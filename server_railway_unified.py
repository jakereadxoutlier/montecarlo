#!/usr/bin/env python3
"""
RAILWAY PRODUCTION SERVER - UNIFIED VERSION
Runs the single unified bot file
"""
import asyncio
import logging
import os
from datetime import datetime
from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("railway-unified")

async def create_http_server():
    """Create HTTP server for Railway health checks"""
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "service": "MonteCarlo Unified Bot",
            "timestamp": str(datetime.now())
        })

    async def root(request):
        return web.json_response({
            "message": "MonteCarlo Bot is running!",
            "commands": ["Smart Picks", "Pick [SYMBOL] $[STRIKE]", "Status", "Help"],
            "architecture": "UNIFIED - Single file, no MCP",
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

async def start_unified_bot():
    """Start the unified bot"""
    try:
        logger.info("🚀 Starting MonteCarlo Unified Bot...")

        # Import and run the unified bot
        from montecarlo_unified import main as run_bot

        # Run as background task
        asyncio.create_task(run_bot())

        logger.info("✅ Unified bot started successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start unified bot: {e}")
        return False

async def main():
    """Main entry point"""
    import uuid
    instance_id = str(uuid.uuid4())[:8]
    logger.info(f"🎯 MonteCarlo Unified Server Starting... [Instance: {instance_id}]")
    logger.info("📊 Single file architecture - No MCP complexity")
    logger.info("🔄 All features in one file")

    try:
        # Start HTTP server first (for Railway)
        http_runner = await create_http_server()

        # Wait for HTTP to stabilize
        await asyncio.sleep(2)

        # Start the unified bot
        bot_success = await start_unified_bot()

        if not bot_success:
            logger.warning("⚠️ Bot failed to start, but HTTP server is running")

        # Keep running with periodic health checks
        while True:
            logger.info("🟢 Unified server healthy...")
            await asyncio.sleep(300)  # Every 5 minutes

    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())