#!/usr/bin/env python3
"""
Debug test for OTM call analysis functionality.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_debug():
    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Session initialized")

                # First, let's check what expiration dates are available for AAPL
                print("\n1. Checking available expiration dates for AAPL...")
                result = await session.call_tool(
                    "get_options_chain_v2",
                    arguments={"symbol": "AAPL"}
                )

                if result.content and hasattr(result.content[0], 'text'):
                    data = json.loads(result.content[0].text)
                    if data.get('success'):
                        exp_dates = data['data']['available_expiration_dates']
                        print(f"Available expiration dates: {exp_dates[:10]}...")

                        # Use the nearest expiration for testing
                        test_exp_date = exp_dates[1] if len(exp_dates) > 1 else exp_dates[0]
                        print(f"Using expiration date: {test_exp_date}")

                        # Now test the OTM call analysis with very permissive criteria
                        print(f"\n2. Testing OTM call analysis with {test_exp_date}...")
                        result2 = await session.call_tool(
                            "analyze_fortune500_otm_calls",
                            arguments={
                                "expiration_date": test_exp_date,
                                "symbols": ["AAPL"],
                                "min_volume": 1,      # Very low
                                "min_iv": 0.01,       # Very low
                                "probability_threshold": 0.01  # Very low
                            }
                        )

                        if result2.content and hasattr(result2.content[0], 'text'):
                            data2 = json.loads(result2.content[0].text)
                            print(f"Response: {json.dumps(data2, indent=2)}")

                            if data2.get('success'):
                                print("✓ OTM analysis completed successfully")
                                return True
                            else:
                                print(f"❌ OTM analysis error: {data2.get('error')}")
                                return False
                    else:
                        print(f"❌ Failed to get options chain: {data.get('error')}")
                        return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return False

if __name__ == "__main__":
    result = asyncio.run(test_debug())
    if result:
        print("✅ Debug test passed")
    else:
        print("❌ Debug test failed")