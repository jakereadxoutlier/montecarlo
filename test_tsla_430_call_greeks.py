#!/usr/bin/env python3
"""
Simple test script specifically for TSLA $430 call expiring 2025-10-10 Greeks calculation.
This demonstrates the new calculate_greeks MCP function.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_tsla_430_call():
    """Test calculate_greeks with TSLA $430 call expiring 2025-10-10."""

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call the new calculate_greeks function
            result = await session.call_tool(
                "calculate_greeks",
                arguments={
                    "symbol": "TSLA",
                    "strike": 430.0,
                    "expiration_date": "2025-10-10",
                    "option_type": "call",
                    "volatility": 0.54,  # 54% IV (from real options data)
                    "risk_free_rate": 0.05  # 5% risk-free rate
                }
            )

            if result.content and hasattr(result.content[0], 'text'):
                data = json.loads(result.content[0].text)
                if data.get('success'):
                    return data['data']

    return None

if __name__ == "__main__":
    print("TSLA $430 Call Greeks Calculator")
    print("================================")
    print("Expiration: 2025-10-10")
    print("Using Black-Scholes model with 54% IV")
    print()

    result = asyncio.run(test_tsla_430_call())

    if result:
        print(f"Current TSLA Price: ${result['market_data']['current_price']:.2f}")
        print(f"Strike Price: ${result['option_details']['strike']:.2f}")
        print(f"Days to Expiration: {result['option_details']['days_to_expiration']}")
        print()
        print("Greeks:")
        greeks = result['greeks']
        print(f"  Delta: {greeks['delta']:6.4f}  (price sensitivity)")
        print(f"  Gamma: {greeks['gamma']:6.4f}  (delta sensitivity)")
        print(f"  Theta: {greeks['theta']:6.4f}  (time decay per day)")
        print(f"  Vega:  {greeks['vega']:6.4f}  (volatility sensitivity)")
        print(f"  Rho:   {greeks['rho']:6.4f}  (interest rate sensitivity)")
        print()
        print(f"Theoretical Option Price: ${greeks['option_price']:.2f}")
    else:
        print("Failed to calculate Greeks")