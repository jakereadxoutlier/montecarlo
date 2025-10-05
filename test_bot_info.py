#!/usr/bin/env python3
"""
Test bot info and permissions to debug Slack setup.
"""
import asyncio
import os
from slack_sdk.web.async_client import AsyncWebClient
import dotenv

dotenv.load_dotenv()

async def test_bot_info():
    client = AsyncWebClient(token=os.getenv('SLACK_BOT_TOKEN'))

    try:
        # Test auth
        print("🔍 TESTING BOT AUTHENTICATION")
        print("=" * 40)
        auth_response = await client.auth_test()
        print(f"✅ Bot authenticated successfully")
        print(f"User ID: {auth_response['user_id']}")
        print(f"Bot Name: {auth_response['user']}")
        print(f"Team: {auth_response['team']}")
        print()

        # Test bot info
        print("📋 BOT INFORMATION")
        print("=" * 40)
        bot_info = await client.bots_info(bot=auth_response['bot_id'])
        print(f"Bot ID: {bot_info['bot']['id']}")
        print(f"Bot Name: {bot_info['bot']['name']}")
        print(f"App ID: {bot_info['bot']['app_id']}")
        print(f"User ID: {bot_info['bot']['user_id']}")
        print()

        # List channels/DMs bot has access to
        print("🏠 CHANNELS BOT CAN ACCESS")
        print("=" * 40)
        try:
            conversations = await client.conversations_list(types="public_channel,private_channel,mpim,im")
            for conv in conversations['channels'][:5]:  # Show first 5
                print(f"- {conv['name']} (ID: {conv['id']}, Type: {conv.get('conversation_host_id', 'unknown')})")
        except Exception as e:
            print(f"❌ Could not list conversations: {e}")

        print()
        print("🎯 DEBUGGING SUGGESTIONS:")
        print("1. Make sure you've invited the bot to your workspace")
        print("2. Try sending a DM directly to the bot user")
        print(f"3. Try mentioning @{auth_response['user']} in a channel")
        print("4. Check if Socket Mode is enabled in your Slack App settings")

    except Exception as e:
        print(f"❌ Bot authentication failed: {e}")
        print()
        print("💡 TROUBLESHOOTING:")
        print("1. Check your SLACK_BOT_TOKEN in .env file")
        print("2. Make sure the token starts with 'xoxb-'")
        print("3. Verify the app is installed in your workspace")

if __name__ == "__main__":
    asyncio.run(test_bot_info())