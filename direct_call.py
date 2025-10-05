#!/usr/bin/env python3
import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stockflow import call_tool

async def get_tsla_options():
    try:
        # Call the options chain function directly
        result = await call_tool("get_options_chain_v2", {
            "symbol": "TSLA",
            "include_greeks": True
        })

        print("TSLA Options Chain with Greeks (Nearest Expiration):")
        print("=" * 60)

        # Extract and print the response
        if result and len(result) > 0:
            import json
            response_data = json.loads(result[0].text)
            print(json.dumps(response_data, indent=2))
        else:
            print("No data returned")

    except Exception as e:
        print(f"Error getting TSLA options: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(get_tsla_options())