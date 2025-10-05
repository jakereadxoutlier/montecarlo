#!/usr/bin/env python3
"""
Fortune 500 Options Coverage Engine - Ensures comprehensive coverage and highest probability filtering.
"""

# Comprehensive Fortune 500 symbols for options analysis
FORTUNE_500_SYMBOLS = [
    # Top 50 by Market Cap (Highest Liquidity)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'BRK-A', 'BRK-B', 'UNH', 'JNJ',
    'META', 'NVDA', 'JPM', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
    'AVGO', 'XOM', 'LLY', 'KO', 'TMO', 'COST', 'WMT', 'MRK', 'CSCO', 'CVX',
    'ABT', 'ACN', 'NFLX', 'DHR', 'VZ', 'ADBE', 'CRM', 'NKE', 'TXN', 'NEE',
    'AMD', 'QCOM', 'RTX', 'PM', 'HON', 'SPGI', 'UNP', 'IBM', 'UPS', 'SCHW',

    # High-Volume Options Stocks (Next 50)
    'INTC', 'GS', 'INTU', 'CAT', 'LOW', 'MS', 'ISRG', 'GE', 'AXP', 'AMAT',
    'DE', 'MU', 'BKNG', 'TJX', 'MDLZ', 'SYK', 'T', 'ADP', 'GILD', 'ADI',
    'LRCX', 'PANW', 'AMT', 'CB', 'MMM', 'SBUX', 'CME', 'DUK', 'SO', 'ELV',
    'ITW', 'CL', 'ZTS', 'BSX', 'BMY', 'EOG', 'PLD', 'AON', 'ICE', 'APD',
    'EQIX', 'SHW', 'CSX', 'NSC', 'WM', 'MCD', 'KLAC', 'FCX', 'EMR', 'PNC',

    # Tech & Growth (High Volatility = High Option Premiums)
    'CRM', 'NOW', 'SNOW', 'DDOG', 'ZM', 'ROKU', 'SQ', 'PYPL', 'SHOP', 'UBER',
    'LYFT', 'PINS', 'TWTR', 'SNAP', 'ZS', 'OKTA', 'CRWD', 'NET', 'FSLY', 'ESTC',

    # Financial Services (High Volume)
    'WFC', 'C', 'USB', 'TFC', 'PNC', 'COF', 'SCHW', 'BLK', 'SPGI', 'ICE',
    'CME', 'MCO', 'AON', 'MMC', 'AJG', 'CB', 'TRV', 'PGR', 'ALL', 'AIG',

    # Healthcare & Biotech (High Volatility)
    'MRNA', 'BNTX', 'REGN', 'BIIB', 'VRTX', 'AMGN', 'CVS', 'CI', 'HUM', 'ANTM',
    'AET', 'WLP', 'ESRX', 'CAH', 'MCK', 'ABC', 'JNJ', 'PFE', 'MRK', 'LLY',

    # Energy (High Volatility)
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'HES', 'DVN',
    'FANG', 'APA', 'MRO', 'OXY', 'HAL', 'BKR', 'NOV', 'FTI', 'RIG', 'HP',

    # Consumer & Retail
    'AMZN', 'WMT', 'HD', 'LOW', 'TGT', 'COST', 'DG', 'DLTR', 'KR', 'WBA',
    'CVS', 'BBBY', 'M', 'JWN', 'KSS', 'DKS', 'ULTA', 'LULU', 'NKE', 'ADDYY',

    # Industrial & Materials
    'GE', 'CAT', 'DE', 'MMM', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
    'GD', 'BA', 'EMR', 'ITW', 'ETN', 'PH', 'CMI', 'FLR', 'JCI', 'IR'
]

def get_fortune500_symbols_by_priority():
    """
    Return Fortune 500 symbols prioritized by options activity and liquidity.
    """
    # Tier 1: Highest priority (most active options)
    tier1 = FORTUNE_500_SYMBOLS[:50]

    # Tier 2: High priority
    tier2 = FORTUNE_500_SYMBOLS[50:100]

    # Tier 3: Medium priority
    tier3 = FORTUNE_500_SYMBOLS[100:150]

    # Tier 4: Lower priority but still included
    tier4 = FORTUNE_500_SYMBOLS[150:]

    return {
        'tier1': tier1,
        'tier2': tier2,
        'tier3': tier3,
        'tier4': tier4,
        'all': FORTUNE_500_SYMBOLS
    }

def get_optimized_symbol_list(max_symbols: int = 20, focus_area: str = 'all'):
    """
    Get optimized symbol list for analysis based on focus area.

    Args:
        max_symbols: Maximum number of symbols to return
        focus_area: 'tech', 'finance', 'healthcare', 'energy', 'all'

    Returns:
        List of optimized symbols for analysis
    """
    if focus_area == 'tech':
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
                  'CRM', 'ADBE', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'LRCX', 'ADI', 'KLAC', 'MU']
    elif focus_area == 'finance':
        symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                  'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'V', 'MA', 'AXP', 'SCHW', 'CB']
    elif focus_area == 'healthcare':
        symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY',
                  'GILD', 'AMGN', 'REGN', 'VRTX', 'BIIB', 'CVS', 'CI', 'HUM', 'ZTS', 'SYK']
    elif focus_area == 'energy':
        symbols = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'HES', 'DVN',
                  'FANG', 'APA', 'MRO', 'OXY', 'HAL', 'BKR', 'NOV', 'NEE', 'DUK', 'SO']
    else:  # 'all' - diversified high-activity options
        symbols = ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V', 'SPY',
                  'QQQ', 'IWM', 'BAC', 'AMD', 'NFLX', 'DIS', 'PYPL', 'SQ', 'ROKU', 'ZM']

    return symbols[:max_symbols]

# Probability threshold configuration
PROBABILITY_THRESHOLDS = {
    'conservative': 0.65,    # Only show 65%+ ITM probability
    'moderate': 0.55,        # Show 55%+ ITM probability
    'aggressive': 0.45,      # Show 45%+ ITM probability
    'experimental': 0.35     # Show 35%+ ITM probability (for research)
}

def get_analysis_config(mode: str = 'moderate'):
    """
    Get analysis configuration based on trading mode.
    """
    return {
        'probability_threshold': PROBABILITY_THRESHOLDS[mode],
        'min_volume': 100 if mode == 'conservative' else 50 if mode == 'moderate' else 25,
        'min_iv': 0.20 if mode == 'conservative' else 0.15 if mode == 'moderate' else 0.10,
        'max_results': 5 if mode == 'conservative' else 10 if mode == 'moderate' else 15,
        'days_range': [7, 45] if mode == 'conservative' else [1, 60] if mode == 'moderate' else [1, 90]
    }

if __name__ == "__main__":
    symbols = get_fortune500_symbols_by_priority()
    print(f"Total Fortune 500 symbols: {len(symbols['all'])}")
    print(f"Tier 1 (highest priority): {len(symbols['tier1'])}")

    tech_focus = get_optimized_symbol_list(20, 'tech')
    print(f"Tech-focused analysis: {tech_focus}")

    config = get_analysis_config('conservative')
    print(f"Conservative config: {config}")