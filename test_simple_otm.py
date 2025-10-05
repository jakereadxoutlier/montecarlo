#!/usr/bin/env python3
"""
Simple test to debug the OTM call analysis functionality.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_simple():
    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Session initialized")

                # Test with a very simple request first
                result = await session.call_tool(
                    "analyze_fortune500_otm_calls",
                    arguments={
                        "expiration_date": "2025-10-17",
                        "symbols": ["AAPL"],  # Just one symbol
                        "min_volume": 10,     # Very low threshold
                        "min_iv": 0.1,        # Very low threshold
                        "probability_threshold": 0.5  # Low threshold
                    }
                )

                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)
                            if data.get('success'):
                                print("✓ Analysis completed successfully")
                                summary = data['data']['summary']
                                print(f"Found {summary['total_options_found']} options")
                                return True
                            else:
                                print(f"❌ Error: {data.get('error')}")
                                return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return False

if __name__ == "__main__":
    result = asyncio.run(test_simple())
    if result:
        print("✅ Simple test passed")
    else:
        print("❌ Simple test failed")