#!/usr/bin/env python3
"""
Test script to communicate with the stockflow MCP server
and get TSLA options chain with Greeks for nearest expiration.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_tsla_options():
    """Connect to the stockflow MCP server and get TSLA options chain with Greeks."""

    # Server parameters - assuming the server is running as stockflow.py
    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    try:
        print("Connecting to stockflow MCP server...")

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✓ Connected and initialized session")

                # List available tools to confirm
                tools = await session.list_tools()
                print(f"✓ Available tools: {[tool.name for tool in tools.tools]}")

                # Call the get_options_chain_v2 tool with TSLA symbol and Greeks enabled
                print("\nCalling get_options_chain_v2 for TSLA with Greeks...")

                result = await session.call_tool(
                    "get_options_chain_v2",
                    arguments={
                        "symbol": "TSLA",
                        "include_greeks": True
                        # Note: expiration_date is optional - will use nearest if not provided
                    }
                )

                print("✓ Successfully retrieved TSLA options data")

                # Parse and display the results
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)

                            if data.get('success'):
                                options_data = data['data']
                                print(f"\n{'='*60}")
                                print(f"TSLA OPTIONS CHAIN - {options_data['expiration_date']}")
                                print(f"{'='*60}")
                                print(f"Underlying Price: ${options_data['underlying_price']:.2f}")
                                print(f"Days to Expiration: {options_data['days_to_expiration']}")
                                print(f"Total Volume: {options_data['summary']['total_volume']:,}")
                                print(f"Put/Call Ratio: {options_data['summary']['put_call_ratio']:.3f}")

                                # Show some call options with highest volume
                                print(f"\nHIGHEST VOLUME CALL OPTIONS:")
                                print(f"{'Strike':<8} {'Volume':<10} {'Bid':<8} {'Ask':<8} {'IV':<8}")
                                print("-" * 50)

                                # Sort calls by volume and show top 5
                                calls = sorted(options_data['calls'],
                                             key=lambda x: x.get('volume', 0) or 0,
                                             reverse=True)[:5]

                                for call in calls:
                                    strike = call.get('strike', 0)
                                    volume = call.get('volume', 0) or 0
                                    bid = call.get('bid', 0) or 0
                                    ask = call.get('ask', 0) or 0
                                    iv = call.get('impliedVolatility', 0) or 0
                                    print(f"${strike:<7} {volume:<10} ${bid:<7.2f} ${ask:<7.2f} {iv:<7.3f}")

                                # Show some put options with highest volume
                                print(f"\nHIGHEST VOLUME PUT OPTIONS:")
                                print(f"{'Strike':<8} {'Volume':<10} {'Bid':<8} {'Ask':<8} {'IV':<8}")
                                print("-" * 50)

                                # Sort puts by volume and show top 5
                                puts = sorted(options_data['puts'],
                                            key=lambda x: x.get('volume', 0) or 0,
                                            reverse=True)[:5]

                                for put in puts:
                                    strike = put.get('strike', 0)
                                    volume = put.get('volume', 0) or 0
                                    bid = put.get('bid', 0) or 0
                                    ask = put.get('ask', 0) or 0
                                    iv = put.get('impliedVolatility', 0) or 0
                                    print(f"${strike:<7} {volume:<10} ${bid:<7.2f} ${ask:<7.2f} {iv:<7.3f}")

                                # Show available expiration dates
                                print(f"\nAVAILABLE EXPIRATION DATES:")
                                exp_dates = options_data.get('available_expiration_dates', [])
                                for i, date in enumerate(exp_dates[:10]):  # Show first 10
                                    marker = " ← Current" if date == options_data['expiration_date'] else ""
                                    print(f"  {date}{marker}")
                                if len(exp_dates) > 10:
                                    print(f"  ... and {len(exp_dates) - 10} more dates")

                                return options_data

                            else:
                                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                                return None

    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {str(e)}")
        print("Make sure the stockflow server is running!")
        return None

if __name__ == "__main__":
    print("TSLA Options Chain Retrieval Test")
    print("=" * 40)

    result = asyncio.run(get_tsla_options())

    if result:
        print(f"\n✅ Successfully retrieved options data for TSLA")
        print(f"   Expiration: {result['expiration_date']}")
        print(f"   Total contracts: {len(result['calls']) + len(result['puts'])}")
    else:
        print("\n❌ Failed to retrieve options data")