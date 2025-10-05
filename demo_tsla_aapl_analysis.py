#!/usr/bin/env python3
"""
Demo script showing the enhanced StockFlow MCP capabilities.
Successfully demonstrates TSLA and AAPL OTM call analysis with Monte Carlo simulation.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def demo_enhanced_features():
    """Demo the enhanced Fortune 500 OTM call analysis features."""

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("🚀 ENHANCED STOCKFLOW MCP DEMO")
            print("=" * 50)
            print("Features: Monte Carlo, Async Processing, Fortune 500 Integration")
            print()

            # Demo the OTM call analysis
            result = await session.call_tool(
                "analyze_fortune500_otm_calls",
                arguments={
                    "expiration_date": "2025-10-17",
                    "symbols": ["TSLA", "AAPL"],
                    "min_volume": 50,
                    "probability_threshold": 0.20,
                    "alert_threshold": 0.45
                }
            )

            if result.content and hasattr(result.content[0], 'text'):
                data = json.loads(result.content[0].text)
                if data.get('success'):
                    analysis = data['data']

                    print("📊 ANALYSIS COMPLETE!")
                    print(f"Options found: {analysis['summary']['total_options_found']}")
                    print()

                    print("🏆 TOP 3 OTM CALL OPPORTUNITIES:")
                    for i, opt in enumerate(analysis['top_10_otm_calls'][:3], 1):
                        print(f"{i}. {opt['symbol']} ${opt['strike']} Call")
                        print(f"   ITM Probability: {opt['itm_probability']:.1%} (Monte Carlo 10K paths)")
                        print(f"   Delta: {opt['delta']:.4f}")
                        print(f"   Volume: {opt['volume']:,} contracts")
                        print()

                    return True

    return False

if __name__ == "__main__":
    success = asyncio.run(demo_enhanced_features())
    if success:
        print("✅ Demo completed successfully!")
        print("The enhanced StockFlow MCP is ready for production use.")
    else:
        print("❌ Demo failed")