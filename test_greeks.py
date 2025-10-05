#!/usr/bin/env python3
"""
Test script to test the new calculate_greeks function with TSLA $430 call expiring 2025-10-10.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_calculate_greeks():
    """Test the calculate_greeks function with TSLA $430 call."""

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    try:
        print("Testing calculate_greeks function...")
        print("=" * 50)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✓ Connected and initialized session")

                # List available tools to confirm calculate_greeks is available
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]
                print(f"✓ Available tools: {tool_names}")

                if "calculate_greeks" not in tool_names:
                    print("❌ calculate_greeks tool not found!")
                    return None

                # Test the calculate_greeks tool with TSLA $430 call expiring 2025-10-10
                print("\nTesting calculate_greeks for TSLA $430 call expiring 2025-10-10...")

                result = await session.call_tool(
                    "calculate_greeks",
                    arguments={
                        "symbol": "TSLA",
                        "strike": 430.0,
                        "expiration_date": "2025-10-10",
                        "option_type": "call",
                        "volatility": 0.54,  # Use approximate IV from our earlier data
                        "risk_free_rate": 0.05
                        # current_price will be fetched from yfinance
                    }
                )

                print("✓ Successfully calculated Greeks for TSLA $430 call")

                # Parse and display the results
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)

                            if data.get('success'):
                                greeks_data = data['data']
                                print(f"\n{'='*60}")
                                print(f"TSLA $430 CALL GREEKS - {greeks_data['option_details']['expiration_date']}")
                                print(f"{'='*60}")
                                print(f"Current Stock Price: ${greeks_data['market_data']['current_price']:.2f}")
                                print(f"Strike Price: ${greeks_data['option_details']['strike']:.2f}")
                                print(f"Days to Expiration: {greeks_data['option_details']['days_to_expiration']}")
                                print(f"Implied Volatility: {greeks_data['market_data']['volatility']:.1%}")
                                print(f"Risk-free Rate: {greeks_data['market_data']['risk_free_rate']:.1%}")

                                print(f"\nBLACK-SCHOLES GREEKS:")
                                print("-" * 30)
                                greeks = greeks_data['greeks']
                                print(f"Delta:        {greeks['delta']:8.4f}")
                                print(f"Gamma:        {greeks['gamma']:8.4f}")
                                print(f"Theta:        {greeks['theta']:8.4f}")
                                print(f"Vega:         {greeks['vega']:8.4f}")
                                print(f"Rho:          {greeks['rho']:8.4f}")
                                print(f"Option Price: ${greeks['option_price']:7.2f}")

                                print(f"\nMONEYNESS ANALYSIS:")
                                print("-" * 30)
                                moneyness = greeks_data['moneyness']
                                print(f"Spot/Strike Ratio: {moneyness['spot_to_strike_ratio']:.4f}")
                                print(f"In the Money: {'Yes' if moneyness['in_the_money'] else 'No'}")
                                print(f"Intrinsic Value: ${moneyness['intrinsic_value']:.2f}")
                                print(f"Time Value: ${moneyness['time_value']:.2f}")

                                return greeks_data

                            else:
                                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                                return None

    except Exception as e:
        print(f"❌ Failed to test calculate_greeks: {str(e)}")
        return None

async def test_edge_cases():
    """Test edge cases for the calculate_greeks function."""

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    print(f"\n{'='*60}")
    print("TESTING EDGE CASES")
    print(f"{'='*60}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test 1: Put option
            print("\n1. Testing TSLA $430 put with same parameters...")
            result = await session.call_tool(
                "calculate_greeks",
                arguments={
                    "symbol": "TSLA",
                    "strike": 430.0,
                    "expiration_date": "2025-10-10",
                    "option_type": "put",
                    "volatility": 0.54,
                    "risk_free_rate": 0.05
                }
            )

            if result.content and hasattr(result.content[0], 'text'):
                data = json.loads(result.content[0].text)
                if data.get('success'):
                    greeks = data['data']['greeks']
                    print(f"   Put Delta: {greeks['delta']:8.4f} (should be negative)")
                    print(f"   Put Gamma: {greeks['gamma']:8.4f} (should be positive)")
                    print(f"   Put Price: ${greeks['option_price']:7.2f}")
                else:
                    print(f"   ❌ Error: {data.get('error')}")

            # Test 2: Different strike (out of the money)
            print("\n2. Testing TSLA $500 call (likely OTM)...")
            result = await session.call_tool(
                "calculate_greeks",
                arguments={
                    "symbol": "TSLA",
                    "strike": 500.0,
                    "expiration_date": "2025-10-10",
                    "option_type": "call",
                    "volatility": 0.67,  # Higher IV for OTM option
                    "risk_free_rate": 0.05
                }
            )

            if result.content and hasattr(result.content[0], 'text'):
                data = json.loads(result.content[0].text)
                if data.get('success'):
                    greeks = data['data']['greeks']
                    moneyness = data['data']['moneyness']
                    print(f"   OTM Call Delta: {greeks['delta']:8.4f} (should be < 0.5)")
                    print(f"   In the Money: {'Yes' if moneyness['in_the_money'] else 'No'}")
                    print(f"   Intrinsic Value: ${moneyness['intrinsic_value']:.2f}")
                else:
                    print(f"   ❌ Error: {data.get('error')}")

if __name__ == "__main__":
    print("CALCULATE_GREEKS FUNCTION TEST")
    print("=" * 40)

    # Test main functionality
    result = asyncio.run(test_calculate_greeks())

    if result:
        print(f"\n✅ Successfully tested calculate_greeks function")

        # Test edge cases
        asyncio.run(test_edge_cases())

        print(f"\n🎯 All tests completed!")
    else:
        print("\n❌ Failed to test calculate_greeks function")