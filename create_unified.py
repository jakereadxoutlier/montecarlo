#!/usr/bin/env python3
"""Create the unified montecarlo bot"""

import re

# Read stockflow.py
with open('stockflow.py', 'r') as f:
    stockflow_content = f.read()

# Read standalone_slack_app.py
with open('standalone_slack_app.py', 'r') as f:
    slack_content = f.read()

# Extract imports from stockflow (lines 1-105 approximately)
stockflow_lines = stockflow_content.split('\n')
imports_section = []
for line in stockflow_lines[:105]:
    # Skip MCP-related imports
    if 'from mcp' not in line and 'import mcp' not in line:
        imports_section.append(line)

# Extract all the trading functions from stockflow (skip MCP server code)
# Find functions section
functions_start = stockflow_content.find('async def fetch_alpha_vantage_data')
if functions_start == -1:
    functions_start = stockflow_content.find('def fetch_alpha_vantage_data')

# Find where MCP server starts (we'll stop here)
mcp_start = stockflow_content.find('class Server:')
if mcp_start == -1:
    mcp_start = stockflow_content.find('async def main():')

# Extract functions
trading_functions = stockflow_content[functions_start:mcp_start]

# Clean up conflicting functions
trading_functions = trading_functions.replace('async def handle_message_events', 'async def OLD_handle_message_events')
trading_functions = trading_functions.replace('async def start_slack_app', 'async def OLD_start_slack_app')
trading_functions = trading_functions.replace('async def stop_slack_app', 'async def OLD_stop_slack_app')

# Extract Slack handlers from standalone_slack_app.py
handlers_start = slack_content.find('async def _handle_smart_picks_internal')
handlers_end = slack_content.find('if __name__ == "__main__"')
slack_handlers = slack_content[handlers_start:handlers_end]

# Build the unified file
unified = '''#!/usr/bin/env python3
"""
MONTECARLO UNIFIED BOT - All-in-one trading bot with Slack
No MCP, no inter-process communication, just one file
"""

'''

# Add imports
unified += '\n'.join(imports_section[:60])  # Core imports from stockflow

# Add Slack imports
unified += '''

# Slack imports
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

'''

# Add configuration section
unified += '''
# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment
from dotenv import load_dotenv
load_dotenv()

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Slack Tokens
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("montecarlo-unified")

'''

# Add global variables
unified += '''
# ============================================================================
# GLOBAL STATE
# ============================================================================

# Selected options for monitoring
selected_options = {}

# Monitoring system state
monitoring_active = False
monitoring_task = None

# Monitoring intervals
PRICE_CHECK_INTERVAL = 60  # 1 minute for price/Greeks
NEWS_CHECK_INTERVAL = 43200  # 12 hours for news/sentiment
last_news_check = {}

# Fortune 500 symbols
FORTUNE_500_SYMBOLS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
    'WMT', 'JPM', 'PG', 'MA', 'UNH', 'DIS', 'HD', 'PYPL', 'BAC', 'CMCSA',
    'NFLX', 'ADBE', 'CRM', 'PFE', 'TMO', 'ABBV', 'NKE', 'PEP', 'KO', 'ABT'
]

'''

# Add all trading functions
unified += '''
# ============================================================================
# TRADING FUNCTIONS (from stockflow.py)
# ============================================================================

'''
unified += trading_functions

# Add Slack handlers
unified += '''

# ============================================================================
# SLACK HANDLERS (from standalone_slack_app.py)
# ============================================================================

'''
unified += slack_handlers

# Add main function
unified += '''

# ============================================================================
# MAIN APPLICATION
# ============================================================================

async def main():
    """Main entry point"""

    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        logger.error("Missing Slack tokens!")
        return

    logger.info("🚀 Starting MonteCarlo Unified Bot...")

    # Initialize Slack App
    app = AsyncApp(token=SLACK_BOT_TOKEN)

    # Setup message handlers
    setup_message_handlers(app)

    # Start Socket Mode handler
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)

    logger.info("✅ MonteCarlo Bot Ready!")
    logger.info("Commands: Smart Picks, Pick TSLA $430, Status, Help")

    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as e:
        logger.error(f"Error: {e}")
'''

# Save the unified file
with open('montecarlo_unified_complete.py', 'w') as f:
    f.write(unified)

print("Created montecarlo_unified_complete.py")
print(f"File size: {len(unified)} characters")