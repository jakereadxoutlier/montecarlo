#!/usr/bin/env python3
"""
Production StockFlow Server - Always-on deployment ready
Auto-restarts on crashes, includes health checks, and proper logging
"""
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
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

class ProductionServer:
    def __init__(self):
        self.running = True
        self.restart_count = 0
        self.max_restarts = 10
        self.last_restart = None

    async def health_check(self):
        """Simple health check loop"""
        while self.running:
            try:
                logger.info(f"🟢 StockFlow Server Health Check - Uptime: {datetime.now()}")
                await asyncio.sleep(300)  # Health check every 5 minutes
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)

    async def run_with_auto_restart(self):
        """Run Slack app with automatic restart on failures"""
        while self.running and self.restart_count < self.max_restarts:
            try:
                logger.info(f"🚀 Starting StockFlow Slack App (Attempt {self.restart_count + 1})")

                # Start health check in background
                health_task = asyncio.create_task(self.health_check())

                # Run the main Slack app
                await run_slack_app()

            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                self.running = False
                health_task.cancel()
                break

            except Exception as e:
                self.restart_count += 1
                self.last_restart = datetime.now()

                logger.error(f"💥 StockFlow crashed: {e}")
                logger.info(f"🔄 Auto-restarting in 10 seconds... (Restart {self.restart_count}/{self.max_restarts})")

                if health_task:
                    health_task.cancel()

                if self.restart_count < self.max_restarts:
                    await asyncio.sleep(10)  # Wait before restart
                else:
                    logger.error(f"❌ Max restarts ({self.max_restarts}) reached. Shutting down.")
                    self.running = False

        logger.info("🔚 StockFlow Server shutdown complete")

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}")
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

async def main():
    """Production server main entry point"""
    logger.info("🎯 StockFlow Production Server Starting...")
    logger.info("📊 Features: Smart Picks, Advanced Analysis, 7 Novel Techniques")

    server = ProductionServer()
    server.setup_signal_handlers()

    try:
        await server.run_with_auto_restart()
    except Exception as e:
        logger.error(f"💀 Production server fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 StockFlow Server stopped by user")
    except Exception as e:
        logger.error(f"🚨 Fatal startup error: {e}")
        sys.exit(1)