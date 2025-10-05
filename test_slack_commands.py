#!/usr/bin/env python3
"""
Test script to verify Smart Picks command integration
"""
import asyncio
import sys
import os
sys.path.append('/Users/jakeread/mcp-stockflow')

from standalone_slack_app import _handle_smart_picks_internal, handle_help

class MockMessage:
    def __init__(self, text):
        self.text = text

    def get(self, key, default=''):
        if key == 'text':
            return self.text
        return default

    def __getitem__(self, key):
        if key == 'text':
            return self.text
        return ''

class MockSay:
    def __init__(self):
        self.messages = []

    async def __call__(self, text):
        print(f"SLACK RESPONSE:\n{text}\n" + "="*50)
        self.messages.append(text)

async def test_smart_picks_command():
    print("🧪 Testing Smart Picks Command Integration")
    print("="*50)

    # Test help command
    print("1. Testing Help Command...")
    mock_message = MockMessage("help")
    mock_say = MockSay()

    try:
        await handle_help(mock_message, mock_say)
        help_response = mock_say.messages[0]
        if "Smart Picks" in help_response and "THE HACK" in help_response:
            print("✅ Help command includes Smart Picks")
        else:
            print("❌ Help command missing Smart Picks")
            print(f"Help response: {help_response[:200]}...")
    except Exception as e:
        print(f"❌ Help command failed: {e}")

    print("\n2. Testing Smart Picks Command...")
    mock_message = MockMessage("smart picks")
    mock_say = MockSay()

    try:
        # This will call the MCP server, so it should work
        await _handle_smart_picks_internal(mock_message, mock_say)
        smart_response = mock_say.messages[0]
        if "Smart Picks Analysis Starting" in smart_response:
            print("✅ Smart Picks command responding correctly")
        else:
            print("❌ Smart Picks unexpected response")
            print(f"Smart response: {smart_response[:200]}...")
    except Exception as e:
        print(f"❌ Smart Picks command failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_smart_picks_command())