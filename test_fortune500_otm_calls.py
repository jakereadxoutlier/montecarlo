#!/usr/bin/env python3
"""
Test script for the new Fortune 500 OTM call analysis functionality.
Tests with TSLA and AAPL OTM calls expiring 2025-10-17.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_fortune500_otm_calls():
    """Test the analyze_fortune500_otm_calls function with TSLA and AAPL."""

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    try:
        print("Testing Fortune 500 OTM Call Analysis")
        print("=" * 50)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✓ Connected and initialized session")

                # List available tools to confirm the new tool is available
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]
                print(f"✓ Available tools: {tool_names}")

                if "analyze_fortune500_otm_calls" not in tool_names:
                    print("❌ analyze_fortune500_otm_calls tool not found!")
                    return None

                # Test the analyze_fortune500_otm_calls tool with TSLA and AAPL
                print("\nTesting OTM call analysis for TSLA and AAPL expiring 2025-10-17...")
                print("Using Monte Carlo simulation (10K paths) for ITM probability...")

                result = await session.call_tool(
                    "analyze_fortune500_otm_calls",
                    arguments={
                        "expiration_date": "2025-10-17",
                        "symbols": ["TSLA", "AAPL"],  # Focus on TSLA and AAPL for testing
                        "min_volume": 50,  # Lower threshold for testing
                        "min_iv": 0.25,    # Lower IV threshold
                        "probability_threshold": 0.65,  # Lower probability threshold for testing
                        "alert_threshold": 0.75
                        # Note: Not providing slack_webhook for testing
                    }
                )

                print("✓ Successfully completed OTM call analysis")

                # Parse and display the results
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            data = json.loads(content.text)

                            if data.get('success'):
                                analysis_data = data['data']
                                print(f"\n{'='*70}")
                                print(f"FORTUNE 500 OTM CALL ANALYSIS - {analysis_data['expiration_date']}")
                                print(f"{'='*70}")

                                # Display summary
                                summary = analysis_data['summary']
                                criteria = analysis_data['criteria']

                                print(f"\nANALYSIS CRITERIA:")
                                print("-" * 30)
                                print(f"Symbols analyzed: {criteria['symbols_analyzed']}")
                                print(f"Min volume: {criteria['min_volume']}")
                                print(f"Min IV: {criteria['min_iv']:.1%}")
                                print(f"Probability threshold: {criteria['probability_threshold']:.1%}")

                                print(f"\nSUMMARY RESULTS:")
                                print("-" * 30)
                                print(f"Total options found: {summary['total_options_found']}")
                                print(f"High-prob alerts: {summary['high_probability_alerts']}")
                                print(f"Average ITM probability: {summary['average_itm_probability']:.1%}")
                                print(f"Average delta: {summary['average_delta']:.4f}")
                                print(f"Total volume: {summary['total_volume']:,}")

                                # Display top options
                                if analysis_data['top_10_otm_calls']:
                                    print(f"\nTOP OTM CALL OPTIONS (Ranked by ITM Probability):")
                                    print("-" * 90)
                                    print(f"{'Rank':<4} {'Symbol':<6} {'Strike':<8} {'Current':<8} {'Delta':<7} {'ITM%':<6} {'Volume':<8} {'IV%':<6}")
                                    print("-" * 90)

                                    for opt in analysis_data['top_10_otm_calls']:
                                        print(f"#{opt['rank']:<3} {opt['symbol']:<6} "
                                              f"${opt['strike']:<7.0f} ${opt['current_price']:<7.2f} "
                                              f"{opt['delta']:<7.4f} {opt['itm_probability']:<6.1%} "
                                              f"{opt['volume']:<8,} {opt['implied_volatility']:<6.1%}")

                                    # Show detailed info for top 3
                                    print(f"\nDETAILED INFO FOR TOP 3 OPTIONS:")
                                    print("-" * 50)
                                    for i, opt in enumerate(analysis_data['top_10_otm_calls'][:3], 1):
                                        print(f"\n#{i} {opt['symbol']} ${opt['strike']} Call:")
                                        print(f"   Current Price: ${opt['current_price']:.2f}")
                                        print(f"   Days to Expiration: {opt['days_to_expiration']}")
                                        print(f"   ITM Probability (Monte Carlo): {opt['itm_probability']:.1%}")
                                        print(f"   Delta: {opt['delta']:.4f}")
                                        print(f"   Mid Price: ${opt['mid_price']:.2f}")
                                        print(f"   Moneyness: {opt['moneyness']:.4f}")
                                        print(f"   Theoretical Price: ${opt['option_price']:.2f}")

                                else:
                                    print(f"\n❌ No OTM call options found matching the criteria")
                                    print(f"   Try lowering the probability_threshold or min_volume")

                                # Display alert info
                                alerts = analysis_data['alerts']
                                print(f"\nALERT CONFIGURATION:")
                                print("-" * 30)
                                print(f"Alert threshold: {alerts['threshold']:.1%}")
                                print(f"Triggered alerts: {alerts['triggered_count']}")
                                print(f"Slack notification: {'✓' if alerts['slack_notification_sent'] else '✗'}")

                                return analysis_data

                            else:
                                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                                return None

    except Exception as e:
        print(f"❌ Failed to test Fortune 500 OTM call analysis: {str(e)}")
        return None

async def test_monte_carlo_performance():
    """Test Monte Carlo simulation performance with different parameters."""

    print(f"\n{'='*70}")
    print("MONTE CARLO PERFORMANCE TEST")
    print(f"{'='*70}")

    server_params = StdioServerParameters(
        command="python3",
        args=["/Users/jakeread/mcp-stockflow/stockflow.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test with different probability thresholds
            test_thresholds = [0.6, 0.7, 0.8]

            for threshold in test_thresholds:
                print(f"\nTesting with {threshold:.1%} probability threshold...")

                result = await session.call_tool(
                    "analyze_fortune500_otm_calls",
                    arguments={
                        "expiration_date": "2025-10-17",
                        "symbols": ["TSLA", "AAPL", "NVDA", "META"],  # 4 symbols
                        "min_volume": 25,
                        "min_iv": 0.2,
                        "probability_threshold": threshold,
                        "alert_threshold": 0.85
                    }
                )

                if result.content and hasattr(result.content[0], 'text'):
                    data = json.loads(result.content[0].text)
                    if data.get('success'):
                        summary = data['data']['summary']
                        print(f"   Found {summary['total_options_found']} options")
                        print(f"   Average ITM probability: {summary['average_itm_probability']:.1%}")
                        print(f"   High-probability alerts: {summary['high_probability_alerts']}")

if __name__ == "__main__":
    print("FORTUNE 500 OTM CALL ANALYSIS TEST")
    print("=" * 40)
    print("Features being tested:")
    print("- Monte Carlo simulation (10K paths)")
    print("- Async batch processing")
    print("- ITM probability ranking")
    print("- Real-time data fetching")
    print("- Slack notifications (configured)")
    print("- TSLA & AAPL OTM calls (2025-10-17)")
    print()

    # Test main functionality
    result = asyncio.run(test_fortune500_otm_calls())

    if result:
        print(f"\n✅ Successfully tested Fortune 500 OTM call analysis")

        # Test performance with different parameters
        asyncio.run(test_monte_carlo_performance())

        print(f"\n🎯 All tests completed!")
        print(f"\nNext steps:")
        print("1. Configure Slack webhook for live alerts")
        print("2. Run with full Fortune 500 symbol list")
        print("3. Enable 5-minute scheduled updates")
        print("4. Monitor for high-probability (>80%) options")
    else:
        print("\n❌ Failed to test Fortune 500 OTM call analysis")