#!/usr/bin/env python3
"""
Debug version of Slack App with extensive logging to identify connection issues.
"""
import asyncio
import json
import os
import logging
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Slack configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug-slack-app")

# Initialize Slack App with debug logging
app = AsyncApp(token=SLACK_BOT_TOKEN, logger=logger)

@app.event("app_mention")
async def handle_app_mention(event, say, logger):
    """Handle app mentions."""
    logger.info(f"🎯 App mentioned! Event: {event}")
    text = event.get('text', '')
    await say(f"I received your mention: {text}")

@app.event("message")
async def handle_message_events(event, say, logger):
    """Handle all message events."""
    logger.info(f"📩 Message event received! Event: {event}")

    # Skip bot messages
    if event.get('bot_id') or event.get('subtype') == 'bot_message':
        logger.info("Skipping bot message")
        return

    text = event.get('text', '').strip()
    channel = event.get('channel')
    user = event.get('user')

    logger.info(f"Processing message: '{text}' from user {user} in channel {channel}")

    # Respond to any message for debugging
    await say(f"✅ I received your message: '{text}' - StockFlow bot is working!")

@app.message("hello")
async def handle_hello(message, say, logger):
    """Handle hello messages."""
    logger.info(f"👋 Hello message received: {message}")
    await say("Hello! StockFlow bot is responding!")

@app.message()
async def handle_all_messages(message, say, logger):
    """Catch-all message handler."""
    logger.info(f"🔍 Catch-all handler triggered: {message}")
    text = message.get('text', '')
    await say(f"🤖 StockFlow received: '{text}' - I'm alive and working!")

# Add error handler
@app.error
async def error_handler(error, body, logger):
    logger.error(f"❌ Error occurred: {error}")
    logger.error(f"Request body: {body}")

async def main():
    """Start the debug Slack App."""
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        logger.error("❌ Missing Slack tokens. Please check .env file")
        return

    logger.info("🔧 DEBUGGING SLACK APP CONFIGURATION")
    logger.info(f"Bot Token: {SLACK_BOT_TOKEN[:20]}...")
    logger.info(f"App Token: {SLACK_APP_TOKEN[:20]}...")

    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)

    logger.info("🚀 Starting DEBUG StockFlow Slack App...")
    logger.info("📱 TESTING INSTRUCTIONS:")
    logger.info("1. Send a DM to the bot")
    logger.info("2. Mention the bot in a channel (@stockflow)")
    logger.info("3. Try typing 'hello'")
    logger.info("4. Try 'Options for 10/24/2025'")
    logger.info("📊 Watch this console for activity logs")

    try:
        await handler.start_async()
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Debug Slack App stopped by user")
    except Exception as e:
        logger.error(f"💥 Debug Slack App crashed: {e}")
        import traceback
        traceback.print_exc()