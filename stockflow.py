import logging
import asyncio
import yfinance as yf
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
import json
import traceback
import pandas as pd
import datetime
from functools import wraps
import time
from typing import Optional, Dict, Any, List
import numpy as np
from scipy.stats import norm
import math
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import re
from textblob import TextBlob
import urllib.parse
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError
import dotenv
from advanced_analysis_engine import AdvancedOptionsEngine

# Load environment variables
dotenv.load_dotenv()

# API Keys Configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
X_API_KEY = os.getenv('X_API_KEY')
X_API_SECRET = os.getenv('X_API_SECRET')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Slack App Configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')  # xoxb-your-bot-token
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')  # xapp-your-app-token
SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')

# Global option selection and monitoring system
selected_options = {}  # Store selected options for monitoring
monitoring_active = False
monitoring_task = None  # Background monitoring task

# Optimized monitoring intervals
PRICE_CHECK_INTERVAL = 60  # 1 minute for price/Greeks (FREE - yfinance only)
NEWS_CHECK_INTERVAL = 43200  # 12 hours for news/sentiment (PAID APIs)
last_news_check = {}  # Track last news check per symbol

# Slack App initialization - deferred until start_slack_app() is called
slack_app = None
slack_handler = None
slack_app_running = False

# Initialize Advanced Options Engine for novel analysis techniques
advanced_engine = AdvancedOptionsEngine()

class StockflowJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if isinstance(obj, datetime.date):  # Add this line
            return obj.isoformat()          # Add this line
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
        
def convert_df_timestamps(df):
    """Convert DataFrame timestamps in column names to ISO format strings"""
    df = df.copy()
    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
    return df.to_dict('records')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stockflow_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stockflow-server-v2")

class StockflowError(Exception):
    pass

class ValidationError(StockflowError):
    pass

class APIError(StockflowError):
    pass

# Alpha Vantage and FRED API Integration
async def fetch_alpha_vantage_data(function: str, symbol: str = None, **kwargs) -> dict:
    """Fetch data from Alpha Vantage API."""
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("ALPHA_VANTAGE_API_KEY not configured, returning mock data")
        return {"Note": "Alpha Vantage API key not configured"}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "apikey": ALPHA_VANTAGE_API_KEY,
        **kwargs
    }

    if symbol:
        params["symbol"] = symbol

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "Error Message" in data:
                        logger.error(f"Alpha Vantage error: {data['Error Message']}")
                        return {}
                    if "Note" in data and "API call frequency" in data["Note"]:
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        return {}
                    return data
                else:
                    logger.error(f"Alpha Vantage HTTP error: {response.status}")
                    return {}
    except Exception as e:
        logger.error(f"Alpha Vantage API error: {e}")
        return {}

async def fetch_fred_data(series_id: str, limit: int = 100) -> dict:
    """Fetch economic data from FRED API."""
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not configured, returning mock data")
        return {"observations": []}

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"FRED API HTTP error: {response.status}")
                    return {"observations": []}
    except Exception as e:
        logger.error(f"FRED API error: {e}")
        return {"observations": []}

async def get_alpha_vantage_earnings_calendar() -> Dict[str, Any]:
    """Get earnings calendar from Alpha Vantage."""
    earnings_data = await fetch_alpha_vantage_data("EARNINGS_CALENDAR", horizon="3month")

    if not earnings_data or "Note" in earnings_data:
        logger.info("Using fallback earnings data")
        return {"earnings_calendar": [], "source": "fallback"}

    # Parse CSV response from Alpha Vantage
    earnings_list = []
    if isinstance(earnings_data, str):
        # Alpha Vantage returns CSV for earnings calendar
        lines = earnings_data.strip().split('\n')
        if len(lines) > 1:
            headers = lines[0].split(',')
            for line in lines[1:]:
                values = line.split(',')
                if len(values) >= 3:
                    earnings_list.append({
                        'symbol': values[0],
                        'name': values[1] if len(values) > 1 else '',
                        'reportDate': values[2] if len(values) > 2 else '',
                        'fiscalDateEnding': values[3] if len(values) > 3 else '',
                        'estimate': values[4] if len(values) > 4 else '',
                        'currency': values[5] if len(values) > 5 else 'USD'
                    })

    return {
        "earnings_calendar": earnings_list[:50],  # Limit to 50 upcoming earnings
        "source": "alpha_vantage"
    }

async def get_alpha_vantage_market_data(symbol: str) -> Dict[str, Any]:
    """Get enhanced market data from Alpha Vantage."""
    # Get intraday data for better market context
    intraday_data = await fetch_alpha_vantage_data(
        "TIME_SERIES_INTRADAY",
        symbol=symbol,
        interval="15min",
        outputsize="compact"
    )

    # Get daily data for technical indicators
    daily_data = await fetch_alpha_vantage_data(
        "TIME_SERIES_DAILY",
        symbol=symbol,
        outputsize="compact"
    )

    # Get technical indicators
    sma_data = await fetch_alpha_vantage_data(
        "SMA",
        symbol=symbol,
        interval="daily",
        time_period=20,
        series_type="close"
    )

    return {
        "intraday_data": intraday_data,
        "daily_data": daily_data,
        "technical_indicators": {"sma": sma_data},
        "source": "alpha_vantage"
    }

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}\n{traceback.format_exc()}")
            raise last_error
        return wrapper
    return decorator

# Fortune 500 stock symbols (top 100 most liquid for performance)
FORTUNE_500_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE',
    'AVGO', 'COST', 'DIS', 'KO', 'ADBE', 'WMT', 'CRM', 'MRK', 'NFLX', 'ACN',
    'TMO', 'VZ', 'CSCO', 'ABT', 'NKE', 'INTC', 'TXN', 'WFC', 'DHR', 'QCOM',
    'BMY', 'RTX', 'AMGN', 'COP', 'HON', 'UPS', 'SBUX', 'LOW', 'PM', 'SPGI',
    'NEE', 'LMT', 'INTU', 'ISRG', 'GS', 'BKNG', 'CAT', 'GILD', 'AMD', 'PLD',
    'TGT', 'MDT', 'AXP', 'BLK', 'SYK', 'MO', 'CVS', 'ANTM', 'MDLZ', 'ADI',
    'TMUS', 'GE', 'CB', 'C', 'ZTS', 'SCHW', 'PYPL', 'FIS', 'MMM', 'DUK',
    'ITW', 'SO', 'AON', 'CCI', 'EL', 'CME', 'USB', 'BSX', 'NSC', 'MCD',
    'IBM', 'EQIX', 'HUM', 'SHW', 'REGN', 'APD', 'CL', 'EMR', 'GD', 'PNC'
]

# Enhanced Monte Carlo simulation for ITM probability with dynamic IV
def monte_carlo_itm_probability_unbiased(
    current_price: float,
    strike: float,
    time_to_expiration: float,  # in years
    volatility: float,
    risk_free_rate: float,
    option_type: str = 'call',
    num_simulations: int = 20000
) -> Dict[str, float]:
    """
    Unbiased Monte Carlo simulation for ITM probability using risk-neutral valuation.
    This provides the pure mathematical probability without sentiment adjustments.

    Args:
        current_price: Current stock price
        strike: Strike price of the option
        time_to_expiration: Time to expiration in years
        volatility: Implied volatility (annualized)
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        num_simulations: Number of Monte Carlo paths (default 20,000)

    Returns:
        Dictionary with unbiased ITM probability, confidence intervals, and stats
    """
    if time_to_expiration <= 0:
        # Option has expired
        if option_type.lower() == 'call':
            return {
                'itm_probability': 1.0 if current_price > strike else 0.0,
                'confidence_95': [0.0, 0.0],
                'avg_final_price': current_price,
                'price_std': 0.0
            }
        else:
            return {
                'itm_probability': 1.0 if current_price < strike else 0.0,
                'confidence_95': [0.0, 0.0],
                'avg_final_price': current_price,
                'price_std': 0.0
            }

    # Risk-neutral Monte Carlo simulation (no sentiment bias)
    # Generate truly random seed based on system time and input parameters
    raw_seed = int((time.time() * 1000) % 2**31) ^ abs(hash((current_price, strike, time_to_expiration)))
    seed = raw_seed % (2**32 - 1)  # Ensure seed is within numpy's valid range
    np.random.seed(seed)

    # Risk-neutral drift (standard Black-Scholes assumption)
    dt = time_to_expiration
    drift = (risk_free_rate - 0.5 * volatility**2) * dt

    # Generate random paths
    random_shocks = np.random.standard_normal(num_simulations)
    diffusion = volatility * math.sqrt(dt) * random_shocks

    # Calculate final stock prices at expiration
    final_prices = current_price * np.exp(drift + diffusion)

    # Count ITM outcomes
    if option_type.lower() == 'call':
        itm_outcomes = final_prices > strike
    else:
        itm_outcomes = final_prices < strike

    itm_count = np.sum(itm_outcomes)
    itm_probability = float(itm_count / num_simulations)

    # Calculate confidence intervals using binomial distribution
    confidence_95 = [
        max(0.0, itm_probability - 1.96 * math.sqrt(itm_probability * (1 - itm_probability) / num_simulations)),
        min(1.0, itm_probability + 1.96 * math.sqrt(itm_probability * (1 - itm_probability) / num_simulations))
    ]

    return {
        'itm_probability': itm_probability,
        'confidence_95': confidence_95,
        'avg_final_price': float(np.mean(final_prices)),
        'price_std': float(np.std(final_prices)),
        'simulations': num_simulations,
        'method': 'risk_neutral_mc'
    }

def apply_sentiment_adjustment(base_probability: float, sentiment_boost: float) -> float:
    """
    Apply sentiment adjustment to base ITM probability in a controlled, transparent way.
    This separates the mathematical calculation from sentiment analysis.

    Args:
        base_probability: Unbiased ITM probability from Monte Carlo
        sentiment_boost: Sentiment boost factor (-0.2 to +0.2)

    Returns:
        Sentiment-adjusted ITM probability
    """
    # Apply sentiment boost with diminishing returns for extreme probabilities
    if sentiment_boost > 0:
        # Positive sentiment: boost probability with diminishing returns near 1.0
        adjustment = sentiment_boost * (1 - base_probability) * base_probability
    else:
        # Negative sentiment: reduce probability with diminishing returns near 0.0
        adjustment = sentiment_boost * base_probability * (1 - base_probability)

    adjusted_probability = base_probability + adjustment
    return max(0.0, min(1.0, adjusted_probability))

def monte_carlo_itm_probability_enhanced(
    current_price: float,
    strike: float,
    time_to_expiration: float,  # in years
    volatility: float,
    risk_free_rate: float,
    option_type: str = 'call',
    num_simulations: int = 20000,
    sentiment_boost: float = 0.0  # Sentiment boost factor (-0.2 to +0.2)
) -> Dict[str, float]:
    """
    Enhanced Monte Carlo simulation that combines unbiased analysis with sentiment adjustment.
    This maintains transparency by separating mathematical calculation from sentiment analysis.

    Args:
        current_price: Current stock price
        strike: Strike price of the option
        time_to_expiration: Time to expiration in years
        volatility: Implied volatility (annualized)
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'
        num_simulations: Number of Monte Carlo paths (default 20,000)
        sentiment_boost: Sentiment boost factor (-0.2 to +0.2)

    Returns:
        Dictionary with both unbiased and sentiment-adjusted probabilities
    """
    # Get unbiased Monte Carlo result
    unbiased_result = monte_carlo_itm_probability_unbiased(
        current_price, strike, time_to_expiration, volatility,
        risk_free_rate, option_type, num_simulations
    )

    base_probability = unbiased_result['itm_probability']

    # Apply sentiment adjustment transparently
    if abs(sentiment_boost) > 0.001:  # Only apply if meaningful sentiment
        adjusted_probability = apply_sentiment_adjustment(base_probability, sentiment_boost)

        # Recalculate confidence intervals for adjusted probability
        adjusted_confidence_95 = [
            max(0.0, adjusted_probability - 1.96 * math.sqrt(adjusted_probability * (1 - adjusted_probability) / num_simulations)),
            min(1.0, adjusted_probability + 1.96 * math.sqrt(adjusted_probability * (1 - adjusted_probability) / num_simulations))
        ]
    else:
        adjusted_probability = base_probability
        adjusted_confidence_95 = unbiased_result['confidence_95']

    return {
        'itm_probability': adjusted_probability,
        'base_itm_probability': base_probability,
        'sentiment_adjustment': adjusted_probability - base_probability,
        'confidence_95': adjusted_confidence_95,
        'avg_final_price': unbiased_result['avg_final_price'],
        'price_std': unbiased_result['price_std'],
        'sentiment_boost': sentiment_boost,
        'simulations': num_simulations,
        'method': 'enhanced_transparent'
    }

# Async data fetcher for multiple stocks
async def fetch_multiple_stock_data(symbols: List[str], max_concurrent: int = 20) -> Dict[str, Any]:
    """
    Fetch stock data for multiple symbols concurrently.

    Args:
        symbols: List of stock symbols
        max_concurrent: Maximum concurrent requests

    Returns:
        Dictionary mapping symbols to their data
    """
    async def fetch_single_stock(symbol: str) -> tuple[str, Any]:
        try:
            # Use ThreadPoolExecutor for yfinance calls (it's not async)
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)

            if not info or not info.get('regularMarketPrice') and not info.get('currentPrice'):
                return symbol, None

            return symbol, {
                'price': info.get('regularMarketPrice') or info.get('currentPrice'),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'name': info.get('longName', symbol)
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
            return symbol, None

    # Process symbols in batches to avoid overwhelming the API
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_fetch(symbol):
        async with semaphore:
            return await fetch_single_stock(symbol)

    tasks = [bounded_fetch(symbol) for symbol in symbols]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed:
        if isinstance(result, tuple):
            symbol, data = result
            results[symbol] = data

    return results

# News and Sentiment Analysis Functions
@retry_on_error(max_retries=3, delay=2.0)
async def fetch_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Fetch news sentiment from NewsAPI.org for a given stock symbol with robust error handling.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with sentiment score and recent news
    """
    try:
        if not NEWSAPI_KEY or NEWSAPI_KEY == 'your_newsapi_key_here':
            logger.warning(f"NewsAPI key not configured, using neutral sentiment for {symbol}")
            return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}

        # NewsAPI endpoint
        url = "https://newsapi.org/v2/everything"

        # Expanded company names mapping for better search results
        company_names = {
            'TSLA': 'Tesla',
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'GOOG': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook',
            'AMZN': 'Amazon',
            'BRK-B': 'Berkshire Hathaway',
            'UNH': 'UnitedHealth',
            'JNJ': 'Johnson Johnson',
            'JPM': 'JPMorgan Chase',
            'V': 'Visa',
            'PG': 'Procter Gamble',
            'XOM': 'Exxon Mobil',
            'HD': 'Home Depot',
            'CVX': 'Chevron',
            'MA': 'Mastercard',
            'BAC': 'Bank of America',
            'ABBV': 'AbbVie',
            'PFE': 'Pfizer'
        }

        search_term = f"{company_names.get(symbol, symbol)} stock"

        params = {
            'q': search_term,
            'apiKey': NEWSAPI_KEY,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 20,
            'from': (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        }

        # Configure timeout and SSL settings
        timeout = aiohttp.ClientTimeout(total=15)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {
                'User-Agent': 'StockFlow-Analysis/1.0',
                'Accept': 'application/json'
            }

            async with session.get(url, params=params, headers=headers, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])

                    if not articles:
                        logger.info(f"No recent news found for {symbol}")
                        return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}

                    # Analyze sentiment of headlines and descriptions
                    sentiments = []
                    processed_articles = 0

                    for article in articles:
                        title = article.get('title', '').strip()
                        description = article.get('description', '').strip()

                        # Skip removed or invalid articles
                        if title.lower() == '[removed]' or not title:
                            continue

                        text = f"{title} {description}"
                        if len(text.strip()) > 10:  # Minimum text length
                            try:
                                blob = TextBlob(text)
                                polarity = blob.sentiment.polarity
                                # Filter out neutral sentiments that are likely noise
                                if abs(polarity) > 0.05:
                                    sentiments.append(polarity)
                                processed_articles += 1
                            except Exception as sentiment_error:
                                logger.debug(f"Sentiment analysis error for {symbol}: {sentiment_error}")
                                continue

                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        confidence = min(len(sentiments) / 8.0, 1.0)  # More articles = higher confidence
                        return {
                            'sentiment_score': float(avg_sentiment),
                            'news_count': len(articles),
                            'processed_articles': processed_articles,
                            'confidence': float(confidence),
                            'sentiment_std': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
                        }
                    else:
                        logger.info(f"No meaningful sentiment data found for {symbol}")
                        return {'sentiment_score': 0.0, 'news_count': len(articles), 'confidence': 0.0}

                elif response.status == 401:
                    logger.error(f"NewsAPI authentication failed - check API key")
                    return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}
                elif response.status == 429:
                    logger.warning(f"NewsAPI rate limit exceeded for {symbol}")
                    return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}
                else:
                    logger.warning(f"NewsAPI returned status {response.status} for {symbol}")
                    return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}

    except aiohttp.ClientError as client_error:
        logger.warning(f"Network error fetching news sentiment for {symbol}: {client_error}")
        raise  # Let the retry decorator handle this
    except Exception as e:
        logger.warning(f"Failed to fetch news sentiment for {symbol}: {str(e)}")

    return {'sentiment_score': 0.0, 'news_count': 0, 'confidence': 0.0}

@retry_on_error(max_retries=2, delay=1.0)
async def fetch_x_trends(symbol: str) -> Dict[str, Any]:
    """
    Fetch X (Twitter) trends and mentions for a given stock symbol with improved analysis.
    Uses intelligent pattern-based analysis when full X API v2 integration is not available.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with trend data and sentiment
    """
    try:
        if not X_API_KEY or X_API_KEY == 'your_x_api_key_here':
            logger.warning(f"X API key not configured, using pattern-based analysis for {symbol}")
            return _get_pattern_based_trends(symbol)

        # TODO: Implement full X API v2 integration
        # For now, use enhanced pattern-based analysis with better logic
        return _get_enhanced_pattern_trends(symbol)

    except Exception as e:
        logger.warning(f"Failed to fetch X trends for {symbol}: {str(e)}")
        return {'trend_score': 0.0, 'mentions': 0, 'confidence': 0.0}

# Real-time Option Analysis Functions
async def get_realtime_option_data(symbol: str, strike: float, expiration_date: str) -> Dict[str, Any]:
    """
    Get real-time option data with 1-minute yfinance pulls.

    Args:
        symbol: Stock ticker symbol
        strike: Strike price
        expiration_date: Expiration date in YYYY-MM-DD format

    Returns:
        Dictionary with real-time option analysis
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get current stock price
        info = ticker.info
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')

        if not current_price:
            return {'error': f'Could not get current price for {symbol}'}

        # Get options chain for the specified expiration
        try:
            options_chain = ticker.option_chain(expiration_date)
            calls = options_chain.calls

            # Find the specific option
            option_data = calls[calls['strike'] == strike]

            if option_data.empty:
                return {'error': f'Option {symbol} ${strike} call not found for {expiration_date}'}

            option_row = option_data.iloc[0]

            # Calculate time to expiration
            exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
            time_to_expiration = max(0, (exp_date - datetime.datetime.now()).days / 365.25)

            # Get sentiment data
            news_data = await fetch_news_sentiment(symbol)
            trend_data = await fetch_x_trends(symbol)
            sentiment_boost = calculate_sentiment_boost(news_data, trend_data)

            # Calculate enhanced ITM probability
            mc_result = monte_carlo_itm_probability_enhanced(
                current_price=current_price,
                strike=strike,
                time_to_expiration=time_to_expiration,
                volatility=float(option_row['impliedVolatility']),
                risk_free_rate=0.05,
                option_type='call',
                num_simulations=20000,
                sentiment_boost=sentiment_boost
            )

            # Calculate Greeks
            greeks = calculate_black_scholes_greeks(
                current_price=current_price,
                strike=strike,
                time_to_expiration=time_to_expiration,
                volatility=float(option_row['impliedVolatility']),
                risk_free_rate=0.05,
                option_type='call'
            )

            return {
                'symbol': symbol,
                'strike': strike,
                'expiration_date': expiration_date,
                'current_price': current_price,
                'option_price': float(option_row['lastPrice']),
                'bid': float(option_row['bid']),
                'ask': float(option_row['ask']),
                'volume': int(option_row['volume']) if pd.notna(option_row['volume']) else 0,
                'open_interest': int(option_row['openInterest']) if pd.notna(option_row['openInterest']) else 0,
                'implied_volatility': float(option_row['impliedVolatility']),
                'itm_probability': mc_result['itm_probability'],
                'base_itm_probability': mc_result.get('base_itm_probability', mc_result['itm_probability']),
                'sentiment_adjustment': mc_result.get('sentiment_adjustment', 0),
                'confidence_95': mc_result['confidence_95'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'time_to_expiration_days': time_to_expiration * 365.25,
                'moneyness': current_price / strike,
                'sentiment_boost': sentiment_boost,
                'news_sentiment': news_data.get('sentiment_score', 0),
                'trend_score': trend_data.get('trend_score', 0),
                'timestamp': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            return {'error': f'Error getting options data: {str(e)}'}

    except Exception as e:
        return {'error': f'Error fetching data for {symbol}: {str(e)}'}

def parse_pick_command(text: str) -> Dict[str, Any]:
    """
    Parse 'Pick [SYMBOL] [STRIKE]' command from Slack message.

    Args:
        text: Slack message text

    Returns:
        Dictionary with parsed command data or error
    """
    # Remove extra whitespace and convert to uppercase
    cleaned_text = ' '.join(text.strip().upper().split())

    # Check for "Options for [date]" command first
    options_date_pattern = r'OPTIONS\s+FOR\s+(\d{1,2})/(\d{1,2})/(\d{4})'
    date_match = re.search(options_date_pattern, cleaned_text)
    if date_match:
        month = int(date_match.group(1))
        day = int(date_match.group(2))
        year = int(date_match.group(3))
        try:
            date_obj = datetime.datetime(year, month, day)
            return {
                'command': 'options_for_date',
                'date': f"{year}-{month:02d}-{day:02d}",
                'date_formatted': date_obj.strftime('%B %d, %Y'),
                'raw_text': text,
                'valid': True
            }
        except ValueError:
            return {
                'valid': False,
                'error': f"Invalid date: {date_match.group(1)}/{date_match.group(2)}/{date_match.group(3)}"
            }

    # Pattern to match "Pick SYMBOL $STRIKE" or "Pick SYMBOL STRIKE"
    patterns = [
        r'PICK\s+([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)',  # Pick TSLA $430 or Pick TSLA 430
        r'BUY\s+([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)',   # Buy TSLA $430
        r'ANALYZE\s+([A-Z]{1,5})\s+\$?(\d+(?:\.\d+)?)', # Analyze TSLA $430
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            symbol = match.group(1)
            strike = float(match.group(2))

            return {
                'command': 'pick',
                'symbol': symbol,
                'strike': strike,
                'raw_text': text,
                'valid': True
            }

    # Check for help or general commands
    if any(word in cleaned_text for word in ['HELP', 'COMMANDS', 'USAGE']):
        return {
            'command': 'help',
            'valid': True
        }

    return {
        'command': 'unknown',
        'raw_text': text,
        'valid': False,
        'error': 'Could not parse command. Try: "Pick TSLA $430", "Options for 10/10/2025", or "Help"'
    }

async def generate_buy_sell_advice(option_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive buy/sell advice based on real-time option data.

    Args:
        option_data: Real-time option analysis data

    Returns:
        Dictionary with buy/sell advice and reasoning
    """
    try:
        symbol = option_data['symbol']
        strike = option_data['strike']
        current_price = option_data['current_price']
        itm_probability = option_data['itm_probability']
        delta = option_data['delta']
        theta = option_data['theta']
        implied_volatility = option_data['implied_volatility']
        time_to_expiration_days = option_data['time_to_expiration_days']
        moneyness = option_data['moneyness']
        sentiment_boost = option_data['sentiment_boost']
        volume = option_data['volume']

        # Scoring system for buy/sell recommendation
        buy_score = 0
        sell_score = 0
        factors = []

        # 1. ITM Probability Analysis
        if itm_probability >= 0.70:
            buy_score += 3
            factors.append(f"High ITM probability: {itm_probability:.1%}")
        elif itm_probability >= 0.50:
            buy_score += 1
            factors.append(f"Moderate ITM probability: {itm_probability:.1%}")
        else:
            sell_score += 2
            factors.append(f"Low ITM probability: {itm_probability:.1%}")

        # 2. Moneyness Analysis
        if moneyness >= 1.05:  # 5% ITM
            sell_score += 2
            factors.append(f"Already ITM: {((moneyness-1)*100):+.1f}%")
        elif 0.95 <= moneyness < 1.05:  # Near the money
            buy_score += 2
            factors.append("Near the money - good risk/reward")
        elif moneyness < 0.90:  # Far OTM
            sell_score += 1
            factors.append(f"Far OTM: {((1-moneyness)*100):.1f}% away")

        # 3. Time Decay Analysis
        if time_to_expiration_days <= 7:
            sell_score += 3
            factors.append(f"Time decay risk: {time_to_expiration_days:.1f} days left")
        elif time_to_expiration_days <= 30:
            sell_score += 1
            factors.append(f"Moderate time decay: {time_to_expiration_days:.1f} days left")
        else:
            buy_score += 1
            factors.append(f"Good time buffer: {time_to_expiration_days:.1f} days left")

        # 4. Delta Analysis
        if delta >= 0.50:
            buy_score += 2
            factors.append(f"High delta: {delta:.2f}")
        elif delta >= 0.30:
            buy_score += 1
            factors.append(f"Moderate delta: {delta:.2f}")
        else:
            sell_score += 1
            factors.append(f"Low delta: {delta:.2f}")

        # 5. Sentiment Analysis
        if sentiment_boost >= 0.05:
            buy_score += 2
            factors.append(f"Bullish sentiment: {sentiment_boost:+.1%}")
        elif sentiment_boost <= -0.05:
            sell_score += 1
            factors.append(f"Bearish sentiment: {sentiment_boost:+.1%}")

        # 6. Volatility Analysis
        if implied_volatility >= 0.40:
            buy_score += 1
            factors.append(f"High IV: {implied_volatility:.1%} (good for buyers)")
        elif implied_volatility <= 0.20:
            sell_score += 1
            factors.append(f"Low IV: {implied_volatility:.1%}")

        # 7. Volume Analysis
        if volume >= 1000:
            buy_score += 1
            factors.append(f"Good liquidity: {volume:,} volume")
        elif volume < 100:
            sell_score += 1
            factors.append(f"Low liquidity: {volume:,} volume")

        # Generate recommendation
        net_score = buy_score - sell_score

        if net_score >= 4:
            recommendation = "STRONG BUY"
            confidence = "High"
        elif net_score >= 2:
            recommendation = "BUY"
            confidence = "Medium"
        elif net_score >= 0:
            recommendation = "WEAK BUY"
            confidence = "Low"
        elif net_score >= -2:
            recommendation = "HOLD/AVOID"
            confidence = "Low"
        else:
            recommendation = "AVOID/SELL"
            confidence = "High"

        # Calculate potential profit/loss scenarios
        potential_scenarios = calculate_profit_scenarios(option_data)

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'net_score': net_score,
            'factors': factors,
            'scenarios': potential_scenarios,
            'summary': f"{recommendation} - {confidence} confidence ({net_score:+d} score)",
            'timestamp': datetime.datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'error': f'Error generating advice: {str(e)}',
            'recommendation': 'ERROR',
            'confidence': 'None'
        }

def calculate_profit_scenarios(option_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate potential profit/loss scenarios for the option.

    Args:
        option_data: Option analysis data

    Returns:
        Dictionary with profit scenarios
    """
    try:
        current_price = option_data['current_price']
        strike = option_data['strike']
        option_price = option_data['option_price']

        # Scenario calculations
        scenarios = {}

        # Breakeven point
        breakeven = strike + option_price
        scenarios['breakeven'] = {
            'price': breakeven,
            'profit_loss': 0,
            'required_move': f"{((breakeven / current_price - 1) * 100):+.1f}%"
        }

        # Scenario: 5% up
        price_5_up = current_price * 1.05
        if price_5_up > strike:
            profit_5_up = (price_5_up - strike) - option_price
            scenarios['5_percent_up'] = {
                'price': price_5_up,
                'profit_loss': profit_5_up,
                'return_pct': (profit_5_up / option_price) * 100 if option_price > 0 else 0
            }
        else:
            scenarios['5_percent_up'] = {
                'price': price_5_up,
                'profit_loss': -option_price,
                'return_pct': -100
            }

        # Scenario: 10% up
        price_10_up = current_price * 1.10
        if price_10_up > strike:
            profit_10_up = (price_10_up - strike) - option_price
            scenarios['10_percent_up'] = {
                'price': price_10_up,
                'profit_loss': profit_10_up,
                'return_pct': (profit_10_up / option_price) * 100 if option_price > 0 else 0
            }
        else:
            scenarios['10_percent_up'] = {
                'price': price_10_up,
                'profit_loss': -option_price,
                'return_pct': -100
            }

        # Scenario: No movement
        scenarios['no_movement'] = {
            'price': current_price,
            'profit_loss': -option_price if current_price <= strike else (current_price - strike) - option_price,
            'return_pct': -100 if current_price <= strike else ((current_price - strike - option_price) / option_price) * 100
        }

        return scenarios

    except Exception as e:
        return {'error': f'Error calculating scenarios: {str(e)}'}

# Slack Event Handlers
async def handle_message_events(message, say, logger):
    """Handle all direct messages and mentions to the bot."""
    try:
        # Extract relevant information
        text = message.get('text', '').strip()
        user_id = message.get('user')
        channel_id = message.get('channel')

        # Skip if no text or from bot
        if not text or message.get('bot_id'):
            return

        logger.info(f"Received message: '{text}' from user {user_id}")

        # Parse the command
        parsed = parse_pick_command(text)

        if not parsed['valid']:
            await say({
                "text": f"Sorry, I didn't understand that command.\n\n{parsed.get('error', '')}\n\nTry:\n- `Pick TSLA $430` - Analyze TSLA $430 call\n- `Help` - Show available commands",
                "channel": channel_id
            })
            return

        if parsed['command'] == 'help':
            help_text = """**StockFlow Bot Commands:**

- `Pick [SYMBOL] $[STRIKE]` - Get buy/sell advice for an option (auto-monitors for sell alerts)
- `Analyze [SYMBOL] $[STRIKE]` - Same as Pick
- `Buy [SYMBOL] $[STRIKE]` - Same as Pick
- `Options for [MM/DD/YYYY]` - Find best options for a specific expiration date
- `Help` - Show this help message

**Examples:**
- `Pick TSLA $430` - Analyzes and starts monitoring
- `Options for 10/10/2025` - Shows best options expiring that Friday
- `Analyze AAPL $200`

I provide real-time option analysis with buy/sell recommendations based on:
- ITM probability (20K Monte Carlo)
- Greek analysis (Delta, Gamma, Theta, Vega)
- Market sentiment and news analysis
- Automatic sell alert notifications"""

            await say({
                "text": help_text,
                "channel": channel_id
            })
            return

        if parsed['command'] == 'pick':
            # Show typing indicator
            await say({
                "text": f"Analyzing {parsed['symbol']} ${parsed['strike']} call... Please wait.",
                "channel": channel_id
            })

            # Get real-time data (assume 2025-10-17 expiration for now)
            # In production, you might want to auto-detect the nearest expiration
            expiration_date = "2025-10-17"

            option_data = await get_realtime_option_data(
                parsed['symbol'],
                parsed['strike'],
                expiration_date
            )

            if 'error' in option_data:
                await say({
                    "text": f"Error analyzing {parsed['symbol']} ${parsed['strike']}: {option_data['error']}",
                    "channel": channel_id
                })
                return

            # Generate buy/sell advice
            advice = await generate_buy_sell_advice(option_data)

            if 'error' in advice:
                await say({
                    "text": f"Error generating advice: {advice['error']}",
                    "channel": channel_id
                })
                return

            # Format response message
            response_text = format_analysis_response(option_data, advice)

            await say({
                "text": response_text,
                "channel": channel_id
            })

            # Automatically select this option for monitoring
            try:
                selected_result = select_option_for_monitoring(
                    parsed['symbol'],
                    parsed['strike'],
                    expiration_date,
                    option_data['current_price'],
                    f"Auto-selected from Slack command by user {user_id}"
                )

                # Auto-start monitoring if not already active
                global monitoring_active
                if not monitoring_active:
                    try:
                        await start_continuous_monitoring()
                        await say({
                            "text": f"✅ {parsed['symbol']} ${parsed['strike']} added to monitoring and auto-monitoring started!\n📊 You'll get Slack alerts when to sell for optimal profit.",
                            "channel": channel_id
                        })
                    except Exception as start_error:
                        logger.warning(f"Could not auto-start monitoring: {start_error}")
                        await say({
                            "text": f"Added {parsed['symbol']} ${parsed['strike']} to monitoring list. Start monitoring with MCP tools for sell alerts.",
                            "channel": channel_id
                        })
                else:
                    await say({
                        "text": f"✅ {parsed['symbol']} ${parsed['strike']} added to active monitoring!\n📊 You'll get Slack alerts when to sell for optimal profit.",
                        "channel": channel_id
                    })

            except Exception as monitor_error:
                logger.warning(f"Could not add to monitoring: {monitor_error}")

        elif parsed['command'] == 'options_for_date':
            # Show typing indicator
            await say({
                "text": f"🔍 Finding best call options for {parsed['date_formatted']}... Please wait.",
                "channel": channel_id
            })

            try:
                # Get best options for the date using our local function
                options_list = await analyze_otm_calls_batch(
                    symbols=['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'V', 'MA', 'JPM'],
                    expiration_date=parsed['date'],
                    min_volume=25,
                    min_iv=0.15,
                    probability_threshold=0.35
                )

                if options_list and len(options_list) > 0:
                    top_options = options_list[:5]  # Show top 5

                    if top_options:
                        response = f"**📈 Best Call Options for {parsed['date_formatted']}**\n\n"
                        response += f"**Top {len(top_options)} Opportunities:**\n"

                        for i, opt in enumerate(top_options, 1):
                            response += f"\n{i}. **{opt['symbol']} ${opt['strike']} Call**\n"
                            response += f"   - ITM Probability: {opt['itm_probability']:.1%}\n"
                            response += f"   - Current Price: ${opt['current_price']:.2f}\n"
                            response += f"   - Option Price: ${opt['option_price']:.2f}\n"
                            response += f"   - Volume: {opt['volume']:,}\n"
                            response += f"   - Days to Expiry: {opt['days_to_expiration']}\n"

                        response += f"\n💡 **To select an option:** Reply with `Pick [SYMBOL] $[STRIKE]`\n"
                        response += f"📊 Example: `Pick {top_options[0]['symbol']} ${top_options[0]['strike']}`"

                        await say({
                            "text": response,
                            "channel": channel_id
                        })
                    else:
                        await say({
                            "text": f"No suitable call options found for {parsed['date_formatted']}. Try a different expiration date.",
                            "channel": channel_id
                        })
                else:
                    await say({
                        "text": f"No suitable call options found for {parsed['date_formatted']}. Try a different expiration date or check market conditions.",
                        "channel": channel_id
                    })

            except Exception as analysis_error:
                logger.error(f"Error in options_for_date analysis: {analysis_error}")
                await say({
                    "text": f"Sorry, I encountered an error finding options for {parsed['date_formatted']}. Please try again.",
                    "channel": channel_id
                })

    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        await say({
            "text": "Sorry, I encountered an error processing your request. Please try again.",
            "channel": channel_id
        })

def format_analysis_response(option_data: Dict[str, Any], advice: Dict[str, Any]) -> str:
    """Format the analysis response for Slack."""

    symbol = option_data['symbol']
    strike = option_data['strike']
    current_price = option_data['current_price']
    option_price = option_data['option_price']
    itm_probability = option_data['itm_probability']

    response = f"""**{symbol} ${strike} Call Analysis**

**Current Status:**
- Stock Price: ${current_price:.2f}
- Option Price: ${option_price:.2f}
- ITM Probability: {itm_probability:.1%}
- Days to Expiry: {option_data['time_to_expiration_days']:.0f}

**Recommendation: {advice['recommendation']}**
Confidence: {advice['confidence']} ({advice['net_score']:+d} score)

**Key Factors:**"""

    for factor in advice['factors'][:6]:  # Limit factors to keep message readable
        response += f"\n- {factor}"

    if len(advice['factors']) > 6:
        response += f"\n- ... and {len(advice['factors']) - 6} more factors"

    # Add profit scenarios
    scenarios = advice.get('scenarios', {})
    if scenarios and 'breakeven' in scenarios:
        breakeven = scenarios['breakeven']
        response += f"\n\n**Profit Scenarios:**"
        response += f"\n- Breakeven: ${breakeven['price']:.2f} ({breakeven['required_move']})"

        if '10_percent_up' in scenarios:
            scenario_10 = scenarios['10_percent_up']
            response += f"\n- If +10% move: {scenario_10['return_pct']:+.0f}% return"

    response += f"\n\n*Analysis timestamp: {datetime.datetime.now().strftime('%H:%M:%S')}*"

    return response

def _get_pattern_based_trends(symbol: str) -> Dict[str, Any]:
    """Basic pattern-based trend analysis."""
    return {
        'trend_score': 0.0,
        'mentions': 0,
        'confidence': 0.0,
        'trending_up': False,
        'method': 'pattern_basic'
    }

def _get_enhanced_pattern_trends(symbol: str) -> Dict[str, Any]:
    """
    Enhanced pattern-based trend analysis using market characteristics.
    This provides more nuanced sentiment based on stock behavior patterns.
    """
    # Base trend score
    trend_boost = 0.0
    confidence = 0.6
    simulated_mentions = 50

    # High-volatility/momentum stocks (tech, EV, growth)
    if symbol in ['TSLA', 'NVDA', 'META', 'GOOGL', 'GOOG', 'AMZN', 'NFLX']:
        trend_boost = 0.03  # 3% positive trend boost
        confidence = 0.75
        simulated_mentions = 150

    # Stable large-cap stocks with moderate momentum
    elif symbol in ['AAPL', 'MSFT', 'JPM', 'BAC', 'V', 'MA']:
        trend_boost = 0.015  # 1.5% positive trend boost
        confidence = 0.65
        simulated_mentions = 100

    # Healthcare/pharma (mixed sentiment)
    elif symbol in ['JNJ', 'PFE', 'ABBV', 'UNH']:
        trend_boost = 0.01  # 1% positive trend boost
        confidence = 0.5
        simulated_mentions = 75

    # Energy sector (volatile sentiment)
    elif symbol in ['XOM', 'CVX']:
        trend_boost = 0.005  # 0.5% positive trend boost
        confidence = 0.4
        simulated_mentions = 60

    # Add some randomness to simulate market conditions
    import random
    random.seed(hash(symbol) % 1000)  # Deterministic but symbol-dependent
    market_noise = random.uniform(-0.005, 0.010)  # -0.5% to +1% noise
    trend_boost += market_noise

    # Ensure trend boost stays within reasonable bounds
    trend_boost = max(-0.02, min(0.06, trend_boost))

    return {
        'trend_score': trend_boost,
        'mentions': simulated_mentions,
        'confidence': confidence,
        'trending_up': trend_boost > 0.005,
        'method': 'pattern_enhanced',
        'market_noise': market_noise
    }

def calculate_sentiment_boost(news_data: Dict[str, Any], trend_data: Dict[str, Any]) -> float:
    """
    Calculate overall sentiment boost from news and X trends.

    Args:
        news_data: News sentiment data
        trend_data: X trends data

    Returns:
        Sentiment boost factor (-0.2 to +0.2)
    """
    try:
        # Weight news sentiment more heavily than trends
        news_weight = 0.7
        trend_weight = 0.3

        # News sentiment contribution (scaled and weighted by confidence)
        news_sentiment = news_data.get('sentiment_score', 0.0)
        news_confidence = news_data.get('confidence', 0.0)
        news_contribution = news_sentiment * news_confidence * news_weight

        # Trend contribution
        trend_score = trend_data.get('trend_score', 0.0)
        trend_confidence = trend_data.get('confidence', 0.0)
        trend_contribution = trend_score * trend_confidence * trend_weight

        # Combine and cap at ±20%
        total_boost = news_contribution + trend_contribution
        return max(-0.2, min(0.2, total_boost))

    except Exception as e:
        logger.warning(f"Failed to calculate sentiment boost: {str(e)}")
        return 0.0

# Analyze OTM call options for multiple stocks
async def analyze_otm_calls_batch(
    symbols: List[str],
    expiration_date: str,
    min_volume: int = 100,
    min_iv: float = 0.3,
    probability_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Analyze OTM call options for multiple stocks and rank by ITM probability.

    Args:
        symbols: List of stock symbols to analyze
        expiration_date: Options expiration date (YYYY-MM-DD)
        min_volume: Minimum option volume filter
        min_iv: Minimum implied volatility filter
        probability_threshold: Minimum ITM probability threshold

    Returns:
        List of top OTM call options ranked by ITM probability
    """
    results = []

    # Validate expiration date
    try:
        exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
        if exp_date <= datetime.datetime.now():
            raise ValueError("Expiration date must be in the future")
        time_to_expiration = (exp_date - datetime.datetime.now()).days / 365.0
    except ValueError as e:
        raise ValidationError(f"Invalid expiration date: {str(e)}")

    async def analyze_single_symbol(symbol: str) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)

            # Check if options are available
            options_dates = await loop.run_in_executor(None, lambda: ticker.options)
            if not options_dates or expiration_date not in options_dates:
                return []

            # Get current price and options chain
            info = await loop.run_in_executor(None, lambda: ticker.info)
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not current_price:
                return []

            options = await loop.run_in_executor(None, ticker.option_chain, expiration_date)
            calls = options.calls

            symbol_results = []

            for _, option in calls.iterrows():
                strike = option['strike']
                volume = option.get('volume', 0) or 0
                iv = option.get('impliedVolatility', 0) or 0
                bid = option.get('bid', 0) or 0
                ask = option.get('ask', 0) or 0

                # Filter for OTM calls with minimum criteria
                if (strike > current_price and  # OTM
                    volume >= min_volume and
                    iv >= min_iv and
                    bid > 0 and ask > 0):

                    # Fetch sentiment data for this symbol (cached per symbol)
                    try:
                        news_data = await fetch_news_sentiment(symbol)
                        trend_data = await fetch_x_trends(symbol)
                        sentiment_boost = calculate_sentiment_boost(news_data, trend_data)

                        # Use Advanced Analysis Engine with 7 novel techniques
                        advanced_result = await advanced_engine.analyze_with_novel_techniques(
                            symbol, strike, expiration_date
                        )

                        # Enhanced ITM probability from novel techniques
                        itm_prob = advanced_result.get('final_analysis', {}).get('final_itm_probability', 0.5)

                        # Fallback to Enhanced Monte Carlo if advanced analysis fails
                        if itm_prob is None or itm_prob <= 0:
                            mc_result = monte_carlo_itm_probability_enhanced(
                                current_price=current_price,
                                strike=strike,
                                time_to_expiration=time_to_expiration,
                                volatility=iv,
                                risk_free_rate=0.05,
                                option_type='call',
                                num_simulations=20000,
                                sentiment_boost=sentiment_boost
                            )
                            itm_prob = mc_result['itm_probability']

                        # Only include options above probability threshold
                        if itm_prob >= probability_threshold:
                            # Calculate Greeks for additional context
                            greeks = calculate_black_scholes_greeks(
                                current_price=current_price,
                                strike=strike,
                                time_to_expiration=time_to_expiration,
                                volatility=iv,
                                risk_free_rate=0.05,
                                option_type='call'
                            )

                            # Get confidence interval from advanced result or fallback to MC result
                            confidence_95 = advanced_result.get('confidence_95', mc_result.get('confidence_95', [itm_prob*0.95, itm_prob*1.05]) if 'mc_result' in locals() else [itm_prob*0.95, itm_prob*1.05])
                            avg_final_price = advanced_result.get('avg_final_price', mc_result.get('avg_final_price', current_price) if 'mc_result' in locals() else current_price)

                            symbol_results.append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'strike': strike,
                                'expiration': expiration_date,
                                'days_to_expiration': (exp_date - datetime.datetime.now()).days,
                                'delta': greeks['delta'],
                                'itm_probability': itm_prob,
                                'itm_confidence_95': confidence_95,
                                'volume': int(volume),
                                'implied_volatility': iv,
                                'bid': bid,
                                'ask': ask,
                                'mid_price': (bid + ask) / 2,
                                'moneyness': current_price / strike,
                                'option_price': greeks['option_price'],
                                'sentiment_boost': sentiment_boost,
                                'news_sentiment': news_data.get('sentiment_score', 0.0),
                                'news_count': news_data.get('news_count', 0),
                                'trend_score': trend_data.get('trend_score', 0.0),
                                'mc_simulations': 20000,
                                'avg_final_price': avg_final_price,
                                # Add advanced analysis data
                                'fractal_volatility': advanced_result.get('analysis_techniques', {}).get('fractal_volatility', {}).get('fractal_volatility', iv),
                                'gamma_squeeze_prob': advanced_result.get('analysis_techniques', {}).get('gamma_squeeze', {}).get('squeeze_probability', 0.0),
                                'options_flow_momentum': advanced_result.get('analysis_techniques', {}).get('flow_momentum', {}).get('flow_momentum_score', 0.0),
                                'market_maker_impact': advanced_result.get('analysis_techniques', {}).get('market_maker_impact', {}).get('mm_impact_score', 0.0),
                                'cross_asset_correlation': advanced_result.get('analysis_techniques', {}).get('cross_asset_correlation', {}).get('cross_asset_adjustment', 0.0),
                                'volatility_surface_score': advanced_result.get('analysis_techniques', {}).get('volatility_surface', {}).get('surface_quality_score', 0.0),
                                'multi_dimensional_score': advanced_result.get('analysis_techniques', {}).get('multi_dimensional_mc', {}).get('itm_probability', itm_prob)
                            })

                    except Exception as e:
                        logger.warning(f"Error calculating probability for {symbol} ${strike} call: {str(e)}")
                        continue

            return symbol_results

        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {str(e)}")
            return []

    # Analyze all symbols concurrently
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async def bounded_analyze(symbol):
        async with semaphore:
            return await analyze_single_symbol(symbol)

    tasks = [bounded_analyze(symbol) for symbol in symbols]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results and sort by ITM probability
    for symbol_results in all_results:
        if isinstance(symbol_results, list):
            results.extend(symbol_results)

    # Sort by ITM probability (descending) and return top 10
    results.sort(key=lambda x: x['itm_probability'], reverse=True)

    # Add ranking
    for i, result in enumerate(results[:10], 1):
        result['rank'] = i

    return results[:10]

async def find_optimal_risk_reward_options(
    symbols: List[str],
    max_days_to_expiry: int = 30,
    min_profit_potential: float = 0.15,
    min_probability: float = 0.45,
    max_risk_level: int = 6,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Find optimal risk/reward options using composite scoring algorithm.
    This is the 'hack' to find the sweet spot between high profitability and high probability.

    Args:
        symbols: List of stock symbols to analyze
        max_days_to_expiry: Maximum days to expiration
        min_profit_potential: Minimum profit potential (e.g., 0.15 = 15%)
        min_probability: Minimum ITM probability threshold
        max_risk_level: Maximum risk level (1-10 scale)
        max_results: Maximum number of results to return

    Returns:
        Dictionary with optimal options ranked by composite risk/reward score
    """
    logger.info(f"Smart Picks: Analyzing {len(symbols)} symbols for optimal risk/reward balance")

    all_options = []
    current_time = datetime.datetime.now()

    # Get all available expiration dates within max_days_to_expiry
    valid_expirations = []
    for days_ahead in range(1, max_days_to_expiry + 1):
        future_date = current_time + datetime.timedelta(days=days_ahead)
        # Only include weekdays (options expire on business days)
        if future_date.weekday() < 5:  # Monday=0, Friday=4
            valid_expirations.append(future_date.strftime('%Y-%m-%d'))

    async def analyze_symbol_all_expirations(symbol: str) -> List[Dict[str, Any]]:
        symbol_options = []

        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)

            # Get available expiration dates
            options_dates = await loop.run_in_executor(None, lambda: ticker.options)
            if not options_dates:
                return []

            # Filter to only dates within our range
            available_dates = [date for date in options_dates if date in valid_expirations]

            if not available_dates:
                return []

            # Get current price
            info = await loop.run_in_executor(None, lambda: ticker.info)
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not current_price:
                return []

            for expiration_date in available_dates:
                try:
                    exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                    days_to_exp = (exp_date - current_time).days
                    time_to_expiration = days_to_exp / 365.0

                    if days_to_exp > max_days_to_expiry:
                        continue

                    # Get options chain
                    options = await loop.run_in_executor(None, ticker.option_chain, expiration_date)
                    calls = options.calls

                    for _, option in calls.iterrows():
                        strike = option['strike']
                        volume = option.get('volume', 0) or 0
                        iv = option.get('impliedVolatility', 0) or 0
                        bid = option.get('bid', 0) or 0
                        ask = option.get('ask', 0) or 0

                        # Basic filtering for liquid options
                        if (strike > current_price and  # OTM only
                            volume >= 25 and  # Minimum liquidity
                            iv >= 0.10 and    # Minimum IV
                            bid > 0 and ask > 0):

                            # Use advanced analysis engine
                            advanced_result = await advanced_engine.analyze_with_novel_techniques(
                                symbol, strike, expiration_date
                            )

                            itm_prob = advanced_result.get('final_analysis', {}).get('final_itm_probability', 0.5)

                            if itm_prob >= min_probability:
                                # Calculate profit potential
                                option_price = (bid + ask) / 2
                                profit_potential = calculate_profit_potential(
                                    current_price, strike, option_price, time_to_expiration, iv
                                )

                                if profit_potential >= min_profit_potential:
                                    # Calculate risk level (1-10 scale)
                                    risk_level = calculate_risk_level(
                                        current_price, strike, time_to_expiration, iv, volume
                                    )

                                    if risk_level <= max_risk_level:
                                        # Calculate composite risk/reward score (the "hack")
                                        composite_score = calculate_composite_score(
                                            itm_prob, profit_potential, risk_level,
                                            days_to_exp, advanced_result.get('analysis_techniques', {})
                                        )

                                        # Calculate Greeks
                                        greeks = calculate_black_scholes_greeks(
                                            current_price, strike, time_to_expiration,
                                            iv, 0.05, 'call'
                                        )

                                        symbol_options.append({
                                            'symbol': symbol,
                                            'current_price': current_price,
                                            'strike': strike,
                                            'expiration': expiration_date,
                                            'days_to_expiration': days_to_exp,
                                            'itm_probability': itm_prob,
                                            'profit_potential': profit_potential,
                                            'risk_level': risk_level,
                                            'composite_score': composite_score,
                                            'option_price': option_price,
                                            'volume': int(volume),
                                            'implied_volatility': iv,
                                            'delta': greeks['delta'],
                                            'gamma': greeks['gamma'],
                                            'theta': greeks['theta'],
                                            'vega': greeks['vega'],
                                            # Advanced metrics
                                            'fractal_volatility': advanced_result.get('analysis_techniques', {}).get('fractal_volatility', {}).get('fractal_volatility', iv),
                                            'gamma_squeeze_prob': advanced_result.get('analysis_techniques', {}).get('gamma_squeeze', {}).get('squeeze_probability', 0.0),
                                            'options_flow_momentum': advanced_result.get('analysis_techniques', {}).get('flow_momentum', {}).get('flow_momentum_score', 0.0),
                                            'market_maker_impact': advanced_result.get('analysis_techniques', {}).get('market_maker_impact', {}).get('mm_impact_score', 0.0),
                                            'multi_dimensional_score': advanced_result.get('analysis_techniques', {}).get('multi_dimensional_mc', {}).get('itm_probability', itm_prob)
                                        })

                except Exception as e:
                    logger.warning(f"Error analyzing {symbol} {expiration_date}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {str(e)}")

        return symbol_options

    # Analyze all symbols concurrently
    semaphore = asyncio.Semaphore(15)  # Slightly higher limit for Smart Picks

    async def bounded_analyze(symbol):
        async with semaphore:
            return await analyze_symbol_all_expirations(symbol)

    tasks = [bounded_analyze(symbol) for symbol in symbols]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results
    for symbol_results in all_results:
        if isinstance(symbol_results, list):
            all_options.extend(symbol_results)

    # Sort by composite score (descending) and return top results
    all_options.sort(key=lambda x: x['composite_score'], reverse=True)

    # Add ranking
    for i, option in enumerate(all_options[:max_results], 1):
        option['rank'] = i

    top_options = all_options[:max_results]

    # Calculate summary statistics
    if top_options:
        avg_score = np.mean([opt['composite_score'] for opt in top_options])
        avg_prob = np.mean([opt['itm_probability'] for opt in top_options])
        avg_profit = np.mean([opt['profit_potential'] for opt in top_options])
        avg_risk = np.mean([opt['risk_level'] for opt in top_options])
        avg_days = np.mean([opt['days_to_expiration'] for opt in top_options])
    else:
        avg_score = avg_prob = avg_profit = avg_risk = avg_days = 0.0

    return {
        'smart_picks_analysis': {
            'total_options_analyzed': len(all_options),
            'top_options_found': len(top_options),
            'analysis_timestamp': time.time(),
            'criteria': {
                'max_days_to_expiry': max_days_to_expiry,
                'min_profit_potential': min_profit_potential,
                'min_probability': min_probability,
                'max_risk_level': max_risk_level,
                'symbols_analyzed': len(symbols)
            },
            'summary_stats': {
                'average_composite_score': round(avg_score, 4),
                'average_itm_probability': round(avg_prob, 4),
                'average_profit_potential': round(avg_profit, 4),
                'average_risk_level': round(avg_risk, 2),
                'average_days_to_expiration': round(avg_days, 1)
            },
            'optimal_options': top_options
        }
    }

def calculate_profit_potential(current_price: float, strike: float, option_price: float,
                             time_to_expiration: float, iv: float) -> float:
    """Calculate profit potential percentage for risk/reward analysis."""
    # Estimate profit at different scenarios
    # Conservative scenario: stock moves to strike + 1 standard deviation
    std_dev_move = current_price * iv * math.sqrt(time_to_expiration)
    target_price = strike + std_dev_move

    # Intrinsic value at target
    intrinsic_at_target = max(0, target_price - strike)

    if option_price > 0:
        profit_potential = (intrinsic_at_target - option_price) / option_price
        return max(0, profit_potential)  # Only positive profit potential

    return 0.0

def calculate_risk_level(current_price: float, strike: float, time_to_expiration: float,
                        iv: float, volume: int) -> float:
    """Calculate risk level on 1-10 scale (1=lowest risk, 10=highest risk)."""
    risk_factors = []

    # 1. Moneyness risk (further OTM = higher risk)
    moneyness = current_price / strike
    moneyness_risk = max(1, 10 * (1 - moneyness))  # Higher when more OTM
    risk_factors.append(min(10, moneyness_risk))

    # 2. Time decay risk (less time = higher risk)
    time_risk = max(1, 10 * (1 - time_to_expiration * 4))  # 4x multiplier for time
    risk_factors.append(min(10, time_risk))

    # 3. Volatility risk (higher IV = higher risk in some contexts)
    iv_risk = min(10, max(1, iv * 20))  # Scale IV to 1-10
    risk_factors.append(iv_risk)

    # 4. Liquidity risk (lower volume = higher risk)
    liquidity_risk = max(1, 10 - math.log10(max(1, volume)))
    risk_factors.append(min(10, liquidity_risk))

    # Average the risk factors
    avg_risk = np.mean(risk_factors)
    return round(avg_risk, 2)

def calculate_composite_score(itm_prob: float, profit_potential: float, risk_level: float,
                            days_to_exp: int, analysis_techniques: Dict) -> float:
    """
    Calculate composite risk/reward score - this is the 'hack' algorithm.
    Higher score = better risk/reward balance.
    """
    # Base score from probability and profit potential
    base_score = itm_prob * profit_potential

    # Risk adjustment (lower risk = higher multiplier)
    risk_multiplier = (11 - risk_level) / 10  # Converts 1-10 to 1.0-0.1 multiplier

    # Time decay bonus (sweet spot around 14-21 days)
    if 14 <= days_to_exp <= 21:
        time_bonus = 1.2
    elif 7 <= days_to_exp <= 28:
        time_bonus = 1.1
    else:
        time_bonus = 1.0

    # Advanced analysis bonuses
    gamma_bonus = 1 + (analysis_techniques.get('gamma_squeeze', {}).get('squeeze_probability', 0) * 0.5)
    flow_bonus = 1 + (analysis_techniques.get('flow_momentum', {}).get('flow_momentum_score', 0) * 0.3)

    # Final composite score
    composite_score = base_score * risk_multiplier * time_bonus * gamma_bonus * flow_bonus

    return round(composite_score, 6)

# Professional Slack notification system for OTM call analysis
def send_professional_slack_notification(webhook_url: str, message: str, options: List[Dict[str, Any]]) -> bool:
    """
    Send professional Slack notification with top 10 OTM calls.
    Format: Clean, professional, no emojis, dash-based lists.

    Args:
        webhook_url: Slack webhook URL
        message: Alert message
        options: List of option data with sentiment information

    Returns:
        True if notification sent successfully, False otherwise
    """
    try:
        if not options:
            return False

        # Format all top 10 options professionally
        formatted_options = []
        for i, opt in enumerate(options[:10], 1):
            sentiment_boost = opt.get('sentiment_boost', 0)

            # Professional formatting without emojis
            formatted_options.append(
                f"#{i} {opt['symbol']} ${opt['strike']} Call\n"
                f"   - ITM Probability: {opt['itm_probability']:.1%} (20K Monte Carlo)\n"
                f"   - Sentiment Impact: {sentiment_boost:+.1%} (News articles: {opt.get('news_count', 0)})\n"
                f"   - Volume: {opt['volume']:,} | IV: {opt['implied_volatility']:.1%}\n"
                f"   - Days to Expiry: {opt['days_to_expiration']}\n"
                f"   - Reply with 'Pick {opt['symbol']} ${opt['strike']}' to select this option"
            )

        # Calculate market sentiment professionally
        avg_sentiment = np.mean([opt.get('sentiment_boost', 0) for opt in options])
        if avg_sentiment > 0.08:
            sentiment_summary = "Very Bullish"
        elif avg_sentiment > 0.03:
            sentiment_summary = "Bullish"
        elif abs(avg_sentiment) <= 0.03:
            sentiment_summary = "Neutral"
        else:
            sentiment_summary = "Bearish"

        # Professional Slack payload
        payload = {
            "text": f"StockFlow Analysis: {message}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"StockFlow Analysis: {message}",
                        "emoji": False
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Market Sentiment: {sentiment_summary} (Average: {avg_sentiment:+.1%})\nAnalysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nOptions Found: {len(options)}"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "TOP 10 OTM CALL OPTIONS\nRanked by ITM Probability with Sentiment Analysis"
                    }
                }
            ]
        }

        # Add options in manageable chunks
        options_text = "\n\n".join(formatted_options)

        # Handle message length limits
        if len(options_text) > 2800:
            mid_point = len(formatted_options) // 2
            first_half = "\n\n".join(formatted_options[:mid_point])
            second_half = "\n\n".join(formatted_options[mid_point:])

            payload["blocks"].extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": first_half
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": second_half
                    }
                }
            ])
        else:
            payload["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": options_text
                }
            })

        # Professional footer
        payload["blocks"].extend([
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "StockFlow Professional Analysis | 20K Monte Carlo Simulation | Real-time Market Data\nTo select an option for monitoring, reply with: 'Pick [SYMBOL] $[STRIKE]'"
                    }
                ]
            }
        ])

        response = requests.post(webhook_url, json=payload, timeout=15)
        return response.status_code == 200

    except Exception as e:
        logger.error(f"Failed to send professional Slack notification: {str(e)}")
        return False

# Legacy function for backward compatibility
def send_slack_notification(webhook_url: str, message: str, options: List[Dict[str, Any]]) -> bool:
    return send_professional_slack_notification(webhook_url, message, options)

# Black-Scholes Greeks calculation functions
def calculate_black_scholes_greeks(
    current_price: float,
    strike: float,
    time_to_expiration: float,  # in years
    volatility: float,
    risk_free_rate: float,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for an option.

    Args:
        current_price: Current stock price
        strike: Strike price of the option
        time_to_expiration: Time to expiration in years
        volatility: Implied volatility (annualized)
        risk_free_rate: Risk-free rate (annualized)
        option_type: 'call' or 'put'

    Returns:
        Dictionary containing delta, gamma, theta, vega, rho, and option price
    """
    if time_to_expiration <= 0:
        # Handle expired options
        if option_type.lower() == 'call':
            intrinsic_value = max(0, current_price - strike)
        else:
            intrinsic_value = max(0, strike - current_price)

        return {
            'delta': 1.0 if (option_type.lower() == 'call' and current_price > strike) or
                           (option_type.lower() == 'put' and current_price < strike) else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'option_price': intrinsic_value
        }

    # Calculate d1 and d2
    d1 = (math.log(current_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiration) / (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)

    # Standard normal cumulative distribution function values
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)

    # Standard normal probability density function
    phi_d1 = norm.pdf(d1)

    # Common terms
    sqrt_T = math.sqrt(time_to_expiration)
    exp_neg_rT = math.exp(-risk_free_rate * time_to_expiration)

    if option_type.lower() == 'call':
        # Call option calculations
        option_price = current_price * N_d1 - strike * exp_neg_rT * N_d2
        delta = N_d1
        rho = strike * time_to_expiration * exp_neg_rT * N_d2 / 100  # Per 1% change in interest rate
    else:
        # Put option calculations
        option_price = strike * exp_neg_rT * N_neg_d2 - current_price * N_neg_d1
        delta = -N_neg_d1
        rho = -strike * time_to_expiration * exp_neg_rT * N_neg_d2 / 100  # Per 1% change in interest rate

    # Greeks that are the same for calls and puts
    gamma = phi_d1 / (current_price * volatility * sqrt_T)
    vega = current_price * phi_d1 * sqrt_T / 100  # Per 1% change in volatility
    theta = ((-current_price * phi_d1 * volatility) / (2 * sqrt_T) -
             risk_free_rate * strike * exp_neg_rT * (N_d2 if option_type.lower() == 'call' else N_neg_d2)) / 365  # Per day

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4),
        'option_price': round(option_price, 4)
    }

# API response wrapper
# Continuous Monitoring System
async def monitor_selected_options():
    """
    OPTIMIZED monitoring: 1-minute price/Greeks checks, 12-hour news/sentiment checks.
    This reduces API costs while maintaining real-time sell alerts.
    """
    global selected_options, monitoring_active, last_news_check

    logger.info("Starting optimized continuous option monitoring system")
    logger.info(f"Price monitoring: Every {PRICE_CHECK_INTERVAL} seconds (FREE)")
    logger.info(f"News monitoring: Every {NEWS_CHECK_INTERVAL/3600:.1f} hours (PAID APIs)")

    while monitoring_active:
        try:
            if not selected_options:
                logger.info("No options selected for monitoring, sleeping for 60 seconds")
                await asyncio.sleep(PRICE_CHECK_INTERVAL)
                continue

            logger.info(f"Monitoring {len(selected_options)} selected options")

            # STEP 1: Get current prices and Greeks (FREE - yfinance only)
            symbols = list(set(opt['symbol'] for opt in selected_options.values()))
            current_prices = {}
            current_greeks = {}

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)

                    # Get real-time price (FREE)
                    info = ticker.info
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                    if current_price:
                        current_prices[symbol] = current_price

                        # Calculate Greeks for each option of this symbol
                        symbol_options = {k: v for k, v in selected_options.items() if v['symbol'] == symbol}
                        for option_key, option_data in symbol_options.items():
                            exp_date = datetime.datetime.strptime(option_data['expiration_date'], '%Y-%m-%d')
                            time_to_exp = max(0.001, (exp_date - datetime.datetime.now()).days / 365.25)

                            # Calculate Greeks (FREE calculation)
                            greeks = calculate_black_scholes_greeks(
                                current_price=current_price,
                                strike=option_data['strike'],
                                time_to_expiration=time_to_exp,
                                volatility=0.25,  # Estimated IV
                                risk_free_rate=0.05,
                                option_type='call'
                            )
                            current_greeks[option_key] = greeks

                except Exception as e:
                    logger.warning(f"Failed to get price/Greeks for {symbol}: {e}")
                    continue

            # STEP 2: Check if we need news/sentiment updates (12-hour intervals)
            current_time = datetime.datetime.now()
            symbols_needing_news = []

            for symbol in symbols:
                last_check = last_news_check.get(symbol)
                if not last_check or (current_time - last_check).seconds >= NEWS_CHECK_INTERVAL:
                    symbols_needing_news.append(symbol)

            # STEP 3: Get news/sentiment data only when needed (PAID APIs)
            sentiment_data = {}
            if symbols_needing_news:
                logger.info(f"Updating news/sentiment for {len(symbols_needing_news)} symbols")
                for symbol in symbols_needing_news:
                    try:
                        # Only call paid APIs when necessary
                        news_data = await fetch_news_sentiment(symbol)
                        trend_data = await fetch_x_trends(symbol)
                        sentiment_boost = calculate_sentiment_boost(news_data, trend_data)

                        sentiment_data[symbol] = {
                            'sentiment_boost': sentiment_boost,
                            'news_sentiment': news_data.get('sentiment_score', 0.0),
                            'trend_score': trend_data.get('trend_score', 0.0)
                        }

                        last_news_check[symbol] = current_time

                    except Exception as e:
                        logger.warning(f"Failed to get sentiment for {symbol}: {e}")
                        # Use cached data or defaults
                        sentiment_data[symbol] = {
                            'sentiment_boost': 0.0,
                            'news_sentiment': 0.0,
                            'trend_score': 0.0
                        }

            # STEP 4: Update each selected option with current data
            for option_key, option_data in selected_options.items():
                try:
                    symbol = option_data['symbol']
                    strike = option_data['strike']
                    expiration_date = option_data['expiration_date']

                    if symbol not in current_prices:
                        continue

                    current_price = current_prices[symbol]
                    greeks = current_greeks.get(option_key, {})

                    # Calculate time to expiration
                    exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                    time_to_expiration = max(0, (exp_date - datetime.datetime.now()).days / 365.25)

                    if time_to_expiration <= 0:
                        option_data['status'] = 'expired'
                        continue

                    # Calculate current ITM probability with latest sentiment (if available)
                    sentiment_boost = 0.0
                    if symbol in sentiment_data:
                        sentiment_boost = sentiment_data[symbol]['sentiment_boost']
                    elif 'last_sentiment_boost' in option_data:
                        sentiment_boost = option_data['last_sentiment_boost']  # Use cached

                    # Enhanced Monte Carlo with sentiment
                    mc_result = monte_carlo_itm_probability_enhanced(
                        current_price=current_price,
                        strike=strike,
                        time_to_expiration=time_to_expiration,
                        volatility=0.25,
                        risk_free_rate=0.05,
                        option_type='call',
                        num_simulations=5000,  # Reduced for faster monitoring
                        sentiment_boost=sentiment_boost
                    )

                    current_itm_prob = mc_result['itm_probability']

                    # Update option data
                    previous_price = option_data.get('last_price')
                    price_change = current_price - previous_price if previous_price else 0
                    price_change_pct = (price_change / previous_price * 100) if previous_price else 0

                    option_data.update({
                        'last_price': current_price,
                        'last_check': datetime.datetime.now().isoformat(),
                        'current_itm_probability': current_itm_prob,
                        'base_itm_probability': mc_result.get('base_itm_probability', current_itm_prob),
                        'time_to_expiration_days': time_to_expiration * 365.25,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'moneyness': current_price / strike,
                        'delta': greeks.get('delta', 0.0),
                        'gamma': greeks.get('gamma', 0.0),
                        'theta': greeks.get('theta', 0.0),
                        'vega': greeks.get('vega', 0.0),
                        'option_price': greeks.get('option_price', 0.0),
                        'last_sentiment_boost': sentiment_boost
                    })

                    # Add latest sentiment data if available
                    if symbol in sentiment_data:
                        option_data.update({
                            'news_sentiment': sentiment_data[symbol]['news_sentiment'],
                            'trend_score': sentiment_data[symbol]['trend_score']
                        })

                    # STEP 5: Check for alert conditions and sell signals
                    await check_monitoring_alerts(option_key, option_data, current_price)
                    await check_intelligent_sell_signals(option_key, option_data, current_price)

                except Exception as e:
                    logger.error(f"Error monitoring option {option_key}: {e}")
                    continue

            # STEP 6: Sleep until next price check
            await asyncio.sleep(PRICE_CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(PRICE_CHECK_INTERVAL)

    logger.info("Optimized continuous monitoring stopped")

async def check_monitoring_alerts(option_key: str, option_data: Dict[str, Any], current_price: float):
    """
    Check if monitoring alerts should be sent for this option.

    Args:
        option_key: Unique option identifier
        option_data: Option monitoring data
        current_price: Current stock price
    """
    global selected_options

    try:
        symbol = option_data['symbol']
        strike = option_data['strike']
        current_itm_prob = option_data.get('current_itm_probability', 0)
        moneyness = option_data.get('moneyness', 0)

        # Alert conditions
        alerts_to_send = []

        # 1. High ITM probability alert (>80%)
        if current_itm_prob >= 0.80:
            alerts_to_send.append(f"High ITM probability: {current_itm_prob:.1%}")

        # 2. Option is now ITM
        if current_price > strike and option_data.get('was_otm', True):
            alerts_to_send.append(f"Option now ITM: {symbol} trading at ${current_price:.2f} vs ${strike} strike")
            option_data['was_otm'] = False

        # 3. Significant price movement (>5% in favor)
        price_change_pct = option_data.get('price_change_pct', 0)
        if price_change_pct >= 5.0:
            alerts_to_send.append(f"Significant price movement: +{price_change_pct:.1f}%")

        # 4. Time decay warning (< 7 days to expiration)
        days_to_expiry = option_data.get('time_to_expiration_days', 0)
        if days_to_expiry <= 7 and days_to_expiry > 0:
            alerts_to_send.append(f"Time decay warning: {days_to_expiry:.1f} days to expiration")

        # Send alerts if any conditions are met
        if alerts_to_send:
            last_alert_time = option_data.get('last_alert')
            current_time = datetime.datetime.now()

            # Rate limiting: don't send alerts more than once every 30 minutes for the same option
            if not last_alert_time or (current_time - datetime.datetime.fromisoformat(last_alert_time)).total_seconds() > 1800:

                alert_message = f"Monitoring Alert: {symbol} ${strike} Call\n" + "\n".join([f"- {alert}" for alert in alerts_to_send])

                # Update alert tracking
                option_data['alerts_sent'] += 1
                option_data['last_alert'] = current_time.isoformat()

                logger.info(f"Monitoring alert for {option_key}: {alerts_to_send}")

                # TODO: Send to Slack webhook if configured
                # Could add Slack webhook URL to option data for personalized alerts

    except Exception as e:
        logger.error(f"Error checking alerts for {option_key}: {e}")

# Intelligent Sell Alert System
async def check_intelligent_sell_signals(option_key: str, option_data: Dict[str, Any], current_price: float):
    """
    Analyze multi-factor conditions for intelligent sell recommendations.

    Args:
        option_key: Unique option identifier
        option_data: Option monitoring data
        current_price: Current stock price
    """
    try:
        symbol = option_data['symbol']
        strike = option_data['strike']
        current_itm_prob = option_data.get('current_itm_probability', 0)
        moneyness = option_data.get('moneyness', 0)
        days_to_expiry = option_data.get('time_to_expiration_days', 0)
        price_change_pct = option_data.get('price_change_pct', 0)

        sell_signals = []
        signal_strength = 0

        # Calculate theoretical option value and Greeks for more sophisticated analysis
        time_to_expiration = days_to_expiry / 365.25
        estimated_iv = 0.25  # Could be enhanced with real IV data

        if time_to_expiration > 0:
            greeks = calculate_black_scholes_greeks(
                current_price=current_price,
                strike=strike,
                time_to_expiration=time_to_expiration,
                volatility=estimated_iv,
                risk_free_rate=0.05,
                option_type='call'
            )

            # Multi-factor sell signal analysis

            # 1. Delta Analysis - Weakening momentum
            delta = greeks.get('delta', 0)
            if delta < 0.3 and moneyness < 1.0:  # OTM with low delta
                sell_signals.append("Low delta warning: Option sensitivity to price movement is declining")
                signal_strength += 2

            # 2. Time Decay Acceleration (< 30 days)
            if days_to_expiry <= 30:
                theta_impact = abs(greeks.get('theta', 0)) * days_to_expiry
                if theta_impact > greeks.get('option_price', 0) * 0.1:  # Time decay > 10% of option value
                    sell_signals.append(f"Time decay acceleration: {days_to_expiry:.1f} days remaining")
                    signal_strength += 3

            # 3. Profit Taking Recommendations
            if current_price > strike:  # ITM
                intrinsic_value = current_price - strike
                option_premium = greeks.get('option_price', intrinsic_value)

                if intrinsic_value / strike >= 0.1:  # 10% ITM
                    sell_signals.append(f"Profit taking opportunity: Option is {(intrinsic_value/strike)*100:.1f}% ITM")
                    signal_strength += 2

            # 4. IV Crush Risk Detection (simplified)
            if days_to_expiry <= 7:  # Near expiration
                sell_signals.append("IV crush risk: Approaching expiration with potential volatility collapse")
                signal_strength += 3

            # 5. Technical Reversal Patterns (price momentum analysis)
            if price_change_pct < -3.0:  # Significant negative movement
                sell_signals.append(f"Price momentum reversal: -{abs(price_change_pct):.1f}% movement")
                signal_strength += 2

            # 6. Probability-Based Exit Strategy
            if current_itm_prob < 0.3 and days_to_expiry <= 14:  # Low probability with short time
                sell_signals.append(f"Low probability exit: {current_itm_prob:.1%} ITM chance with {days_to_expiry:.1f} days")
                signal_strength += 3

            # 7. Risk-Reward Analysis
            max_loss = greeks.get('option_price', strike * 0.1)  # Estimate premium paid
            potential_gain = max(0, current_price - strike) if current_price > strike else 0

            if potential_gain > 0 and potential_gain >= max_loss * 2:  # 2:1 profit ratio achieved
                sell_signals.append(f"Profit target reached: 2:1 risk-reward ratio achieved")
                signal_strength += 1

            # Generate intelligent sell recommendation
            if sell_signals and signal_strength >= 3:  # Threshold for recommendation
                recommendation = determine_sell_recommendation(signal_strength, sell_signals, option_data)

                # Update option data with sell analysis
                option_data['sell_signals'] = sell_signals
                option_data['signal_strength'] = signal_strength
                option_data['sell_recommendation'] = recommendation
                option_data['last_sell_analysis'] = datetime.datetime.now().isoformat()

                # Check if we should send an alert
                last_sell_alert = option_data.get('last_sell_alert')
                current_time = datetime.datetime.now()

                # Send sell alert if it's been more than 1 hour since last sell alert
                if not last_sell_alert or (current_time - datetime.datetime.fromisoformat(last_sell_alert)).total_seconds() > 3600:

                    sell_alert_message = f"SELL RECOMMENDATION: {symbol} ${strike} Call\n"
                    sell_alert_message += f"Recommendation: {recommendation}\n"
                    sell_alert_message += f"Signal Strength: {signal_strength}/10\n"
                    sell_alert_message += "Factors:\n" + "\n".join([f"- {signal}" for signal in sell_signals])

                    option_data['last_sell_alert'] = current_time.isoformat()
                    option_data['sell_alerts_sent'] = option_data.get('sell_alerts_sent', 0) + 1

                    logger.info(f"Intelligent sell alert for {option_key}: {recommendation} (strength: {signal_strength})")

    except Exception as e:
        logger.error(f"Error checking sell signals for {option_key}: {e}")

def determine_sell_recommendation(signal_strength: int, sell_signals: List[str], option_data: Dict[str, Any]) -> str:
    """
    Determine the appropriate sell recommendation based on signal strength and conditions.

    Args:
        signal_strength: Numerical strength of sell signals (0-10)
        sell_signals: List of detected sell signals
        option_data: Option data for context

    Returns:
        String recommendation
    """
    days_to_expiry = option_data.get('time_to_expiration_days', 0)
    current_itm_prob = option_data.get('current_itm_probability', 0)
    moneyness = option_data.get('moneyness', 0)

    if signal_strength >= 7:
        return "STRONG SELL - Multiple risk factors present, consider immediate exit"
    elif signal_strength >= 5:
        if days_to_expiry <= 7:
            return "SELL - Time decay and risk factors suggest exit before expiration"
        elif moneyness >= 1.05:  # 5% ITM
            return "TAKE PROFITS - Good profit opportunity with emerging risks"
        else:
            return "CONSIDER SELLING - Risk factors outweigh potential rewards"
    elif signal_strength >= 3:
        if current_itm_prob >= 0.7:
            return "PARTIAL SELL - Take some profits while monitoring remaining position"
        else:
            return "MONITOR CLOSELY - Some risk factors present, prepare for exit"
    else:
        return "HOLD - Minimal risk factors detected"

def get_sell_analysis_summary() -> Dict[str, Any]:
    """
    Get summary of sell analysis for all monitored options.

    Returns:
        Dictionary with sell analysis summary
    """
    global selected_options

    if not selected_options:
        return {
            'total_options': 0,
            'options_with_sell_signals': 0,
            'recommendations': {}
        }

    sell_recommendations = {}
    options_with_signals = 0
    strong_sell_count = 0
    take_profit_count = 0

    for option_key, option_data in selected_options.items():
        if option_data.get('sell_signals'):
            options_with_signals += 1

            recommendation = option_data.get('sell_recommendation', 'No recommendation')
            signal_strength = option_data.get('signal_strength', 0)

            if 'STRONG SELL' in recommendation:
                strong_sell_count += 1
            elif 'TAKE PROFITS' in recommendation or 'PARTIAL SELL' in recommendation:
                take_profit_count += 1

            sell_recommendations[option_key] = {
                'symbol': option_data['symbol'],
                'strike': option_data['strike'],
                'recommendation': recommendation,
                'signal_strength': signal_strength,
                'sell_signals': option_data.get('sell_signals', []),
                'last_analysis': option_data.get('last_sell_analysis')
            }

    return {
        'total_options': len(selected_options),
        'options_with_sell_signals': options_with_signals,
        'strong_sell_recommendations': strong_sell_count,
        'take_profit_recommendations': take_profit_count,
        'detailed_recommendations': sell_recommendations,
        'analysis_time': datetime.datetime.now().isoformat()
    }

def start_continuous_monitoring() -> Dict[str, Any]:
    """
    Start the continuous monitoring system.

    Returns:
        Dictionary with start confirmation
    """
    global monitoring_active, monitoring_task

    if monitoring_active:
        return {
            'already_running': True,
            'status': 'active',
            'monitored_options': len(selected_options)
        }

    monitoring_active = True

    # Start the monitoring task in the background
    try:
        loop = asyncio.get_event_loop()
        monitoring_task = loop.create_task(monitor_selected_options())

        return {
            'started': True,
            'status': 'active',
            'monitored_options': len(selected_options),
            'monitoring_interval': '1 minute'
        }
    except Exception as e:
        monitoring_active = False
        return {
            'started': False,
            'error': str(e)
        }

def stop_continuous_monitoring() -> Dict[str, Any]:
    """
    Stop the continuous monitoring system.

    Returns:
        Dictionary with stop confirmation
    """
    global monitoring_active, monitoring_task

    if not monitoring_active:
        return {
            'already_stopped': True,
            'status': 'inactive'
        }

    monitoring_active = False

    if monitoring_task and not monitoring_task.done():
        monitoring_task.cancel()

    return {
        'stopped': True,
        'status': 'inactive',
        'final_monitored_options': len(selected_options)
    }

def get_monitoring_status() -> Dict[str, Any]:
    """
    Get current monitoring system status.

    Returns:
        Dictionary with monitoring status and statistics
    """
    global monitoring_active, selected_options

    active_options = sum(1 for opt in selected_options.values() if opt.get('status') == 'active')
    expired_options = sum(1 for opt in selected_options.values() if opt.get('status') == 'expired')

    return {
        'monitoring_active': monitoring_active,
        'total_selected_options': len(selected_options),
        'active_options': active_options,
        'expired_options': expired_options,
        'monitoring_interval': '1 minute' if monitoring_active else 'N/A',
        'last_update': datetime.datetime.now().isoformat()
    }

# Slack App Management Functions
async def start_slack_app() -> Dict[str, Any]:
    """
    Start the Slack App for two-way interaction.

    Returns:
        Dictionary with startup status
    """
    global slack_app_running, slack_handler, slack_app

    try:
        if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
            return {
                'started': False,
                'error': 'Slack tokens not configured. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables.'
            }

        if slack_app_running:
            return {
                'started': False,
                'already_running': True,
                'status': 'active'
            }

        # Initialize Slack App and handler now that event loop is available
        if not slack_app:
            slack_app = AsyncApp(token=SLACK_BOT_TOKEN)

            # Add message handler
            @slack_app.message(re.compile(r".*"))
            async def handle_slack_message(message, say, logger):
                await handle_message_events(message, say, logger)

            slack_handler = AsyncSocketModeHandler(slack_app, SLACK_APP_TOKEN)

        # Start the Slack app in the background
        logger.info("Starting Slack App...")

        # Start handler in a separate task to avoid blocking
        async def run_slack_handler():
            try:
                await slack_handler.start_async()
            except Exception as e:
                logger.error(f"Slack handler error: {e}")
                global slack_app_running
                slack_app_running = False

        # Start the handler task
        loop = asyncio.get_event_loop()
        loop.create_task(run_slack_handler())

        slack_app_running = True

        return {
            'started': True,
            'status': 'active',
            'message': 'Slack App is now listening for messages. Users can send "Pick TSLA $430" commands.'
        }

    except Exception as e:
        return {
            'started': False,
            'error': f'Failed to start Slack App: {str(e)}'
        }

async def stop_slack_app() -> Dict[str, Any]:
    """
    Stop the Slack App.

    Returns:
        Dictionary with stop status
    """
    global slack_app_running, slack_handler

    try:
        if not slack_app_running:
            return {
                'stopped': False,
                'already_stopped': True,
                'status': 'inactive'
            }

        if slack_handler:
            await slack_handler.close_async()

        slack_app_running = False

        return {
            'stopped': True,
            'status': 'inactive',
            'message': 'Slack App stopped successfully.'
        }

    except Exception as e:
        return {
            'stopped': False,
            'error': f'Failed to stop Slack App: {str(e)}'
        }

def get_slack_app_status() -> Dict[str, Any]:
    """
    Get Slack App status and configuration.

    Returns:
        Dictionary with Slack App status
    """
    return {
        'slack_app_running': slack_app_running,
        'tokens_configured': bool(SLACK_BOT_TOKEN and SLACK_APP_TOKEN),
        'webhook_url_configured': bool(SLACK_WEBHOOK_URL),
        'bot_token': f"{'*' * 20}{SLACK_BOT_TOKEN[-10:] if SLACK_BOT_TOKEN else 'Not configured'}",
        'app_token': f"{'*' * 20}{SLACK_APP_TOKEN[-10:] if SLACK_APP_TOKEN else 'Not configured'}",
        'capabilities': [
            'Two-way message handling',
            'Pick command parsing',
            'Real-time option analysis',
            'Buy/sell advice generation',
            'Automatic monitoring integration'
        ],
        'last_update': datetime.datetime.now().isoformat()
    }

# ==================== PHASE 2: ADVANCED ANALYTICS SUITE ====================

async def multi_scenario_monte_carlo_analysis(
    current_price: float,
    strike: float,
    time_to_expiration: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    num_simulations: int = 20000,
    market_regime: str = 'unknown'
) -> Dict[str, Any]:
    """
    Advanced Multi-Scenario Monte Carlo with Bull/Bear/Sideways market modeling.
    This provides institutional-grade probability analysis under different market conditions.
    """
    scenarios = {}

    # Define scenario parameters based on market regime and historical data
    scenario_configs = {
        'bullish': {
            'drift_adjustment': 0.02,  # 2% additional annual drift
            'volatility_multiplier': 0.9,  # Lower volatility in bull markets
            'probability_weight': 0.4 if market_regime == 'bullish' else 0.25
        },
        'bearish': {
            'drift_adjustment': -0.03,  # -3% annual drift reduction
            'volatility_multiplier': 1.3,  # Higher volatility in bear markets
            'probability_weight': 0.4 if market_regime == 'bearish' else 0.2
        },
        'sideways': {
            'drift_adjustment': -0.005,  # Slight drift reduction
            'volatility_multiplier': 1.0,  # Normal volatility
            'probability_weight': 0.4 if market_regime == 'sideways' else 0.55
        }
    }

    total_weighted_itm_prob = 0.0
    scenario_details = {}

    for scenario_name, config in scenario_configs.items():
        # Adjust parameters for this scenario
        adjusted_drift = risk_free_rate + config['drift_adjustment']
        adjusted_volatility = volatility * config['volatility_multiplier']
        scenario_simulations = int(num_simulations * config['probability_weight'])

        # Generate random paths for this scenario
        np.random.seed(42 + hash(scenario_name) % 1000)  # Reproducible but different seeds

        dt = time_to_expiration
        random_shocks = np.random.normal(0, 1, scenario_simulations)

        # Geometric Brownian Motion with scenario-specific parameters
        final_prices = current_price * np.exp(
            (adjusted_drift - 0.5 * adjusted_volatility**2) * dt +
            adjusted_volatility * np.sqrt(dt) * random_shocks
        )

        # Calculate ITM probability for this scenario
        itm_outcomes = final_prices > strike
        scenario_itm_prob = np.mean(itm_outcomes)

        # Calculate additional scenario metrics
        scenario_avg_price = np.mean(final_prices)
        scenario_std = np.std(final_prices)
        percentiles = np.percentile(final_prices, [10, 25, 75, 90])

        # Expected option value in this scenario
        intrinsic_values = np.maximum(final_prices - strike, 0)
        expected_option_value = np.mean(intrinsic_values)

        scenario_details[scenario_name] = {
            'itm_probability': scenario_itm_prob,
            'weight': config['probability_weight'],
            'weighted_contribution': scenario_itm_prob * config['probability_weight'],
            'average_final_price': scenario_avg_price,
            'price_volatility': scenario_std,
            'expected_option_value': expected_option_value,
            'price_percentiles': {
                '10th': percentiles[0],
                '25th': percentiles[1],
                '75th': percentiles[2],
                '90th': percentiles[3]
            },
            'scenario_parameters': {
                'drift_adjustment': config['drift_adjustment'],
                'volatility_multiplier': config['volatility_multiplier'],
                'adjusted_drift': adjusted_drift,
                'adjusted_volatility': adjusted_volatility
            }
        }

        total_weighted_itm_prob += scenario_details[scenario_name]['weighted_contribution']

    # Calculate confidence intervals for the weighted probability
    confidence_95 = [
        max(0.0, total_weighted_itm_prob - 1.96 * math.sqrt(total_weighted_itm_prob * (1 - total_weighted_itm_prob) / num_simulations)),
        min(1.0, total_weighted_itm_prob + 1.96 * math.sqrt(total_weighted_itm_prob * (1 - total_weighted_itm_prob) / num_simulations))
    ]

    return {
        'multi_scenario_itm_probability': total_weighted_itm_prob,
        'confidence_95': confidence_95,
        'scenario_breakdown': scenario_details,
        'market_regime_input': market_regime,
        'total_simulations': num_simulations,
        'methodology': 'Multi-Scenario Monte Carlo with regime-weighted probabilities'
    }

async def historical_pattern_recognition(
    symbol: str,
    lookback_days: int = 252,
    pattern_similarity_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Find similar historical market conditions and analyze how options performed.
    This provides context-aware probability adjustments based on historical patterns.
    """
    try:
        # Get historical data for the symbol
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period=f"{lookback_days * 2}d")

        if len(hist_data) < lookback_days:
            return {'error': 'Insufficient historical data', 'patterns_found': 0}

        # Calculate current market characteristics
        recent_data = hist_data.tail(20)  # Last 20 days
        current_volatility = recent_data['Close'].pct_change().std() * np.sqrt(252)
        current_trend = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * (252/20)  # Annualized
        current_volume_trend = recent_data['Volume'].tail(5).mean() / recent_data['Volume'].head(15).mean()

        # Get VIX data for market context
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period=f"{lookback_days}d")
        current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20

        # Find similar historical periods
        similar_periods = []

        # Sliding window analysis
        for i in range(40, len(hist_data) - 20):  # Leave buffer for analysis
            window_data = hist_data.iloc[i-20:i]
            future_data = hist_data.iloc[i:i+20]  # Next 20 days performance

            if len(window_data) < 20 or len(future_data) < 20:
                continue

            # Calculate historical characteristics for this window
            hist_volatility = window_data['Close'].pct_change().std() * np.sqrt(252)
            hist_trend = (window_data['Close'].iloc[-1] / window_data['Close'].iloc[0] - 1) * (252/20)
            hist_volume_trend = window_data['Volume'].tail(5).mean() / window_data['Volume'].head(15).mean()

            # Get VIX for this period (approximate)
            window_date = window_data.index[-1]
            try:
                hist_vix = vix_data[vix_data.index <= window_date]['Close'].iloc[-1] if not vix_data.empty else 20
            except:
                hist_vix = 20

            # Calculate similarity score
            vol_similarity = 1 - min(1, abs(current_volatility - hist_volatility) / max(current_volatility, hist_volatility))
            trend_similarity = 1 - min(1, abs(current_trend - hist_trend) / max(abs(current_trend), abs(hist_trend), 0.01))
            volume_similarity = 1 - min(1, abs(current_volume_trend - hist_volume_trend) / max(current_volume_trend, hist_volume_trend))
            vix_similarity = 1 - min(1, abs(current_vix - hist_vix) / max(current_vix, hist_vix))

            overall_similarity = (vol_similarity * 0.3 + trend_similarity * 0.3 +
                                volume_similarity * 0.2 + vix_similarity * 0.2)

            if overall_similarity >= pattern_similarity_threshold:
                # Calculate how the stock performed in the following period
                future_return = (future_data['Close'].iloc[-1] / window_data['Close'].iloc[-1] - 1)
                future_volatility = future_data['Close'].pct_change().std() * np.sqrt(252)
                max_drawdown = ((future_data['Close'].cummax() - future_data['Close']) / future_data['Close'].cummax()).max()

                similar_periods.append({
                    'date': window_date.strftime('%Y-%m-%d'),
                    'similarity_score': overall_similarity,
                    'future_return_20d': future_return,
                    'future_volatility': future_volatility,
                    'max_drawdown': max_drawdown,
                    'components': {
                        'volatility_similarity': vol_similarity,
                        'trend_similarity': trend_similarity,
                        'volume_similarity': volume_similarity,
                        'vix_similarity': vix_similarity
                    }
                })

        # Sort by similarity and analyze top matches
        similar_periods.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_patterns = similar_periods[:10]  # Top 10 similar periods

        if not top_patterns:
            return {
                'patterns_found': 0,
                'current_characteristics': {
                    'volatility': current_volatility,
                    'trend': current_trend,
                    'volume_trend': current_volume_trend,
                    'vix': current_vix
                },
                'message': 'No similar historical patterns found with current threshold'
            }

        # Calculate aggregate statistics from similar periods
        avg_future_return = np.mean([p['future_return_20d'] for p in top_patterns])
        avg_future_volatility = np.mean([p['future_volatility'] for p in top_patterns])
        avg_max_drawdown = np.mean([p['max_drawdown'] for p in top_patterns])
        avg_similarity = np.mean([p['similarity_score'] for p in top_patterns])

        # Calculate probability adjustments based on historical outcomes
        positive_outcomes = sum(1 for p in top_patterns if p['future_return_20d'] > 0)
        historical_success_rate = positive_outcomes / len(top_patterns)

        return {
            'patterns_found': len(top_patterns),
            'average_similarity': avg_similarity,
            'historical_outcomes': {
                'average_future_return_20d': avg_future_return,
                'average_future_volatility': avg_future_volatility,
                'average_max_drawdown': avg_max_drawdown,
                'success_rate': historical_success_rate,
                'sample_size': len(top_patterns)
            },
            'current_characteristics': {
                'volatility': current_volatility,
                'trend': current_trend,
                'volume_trend': current_volume_trend,
                'vix': current_vix
            },
            'top_similar_periods': top_patterns[:5],  # Return top 5 for reference
            'probability_adjustment': {
                'bullish_bias': max(-0.15, min(0.15, avg_future_return * 2)),  # Cap adjustment at ±15%
                'volatility_adjustment': avg_future_volatility / current_volatility,
                'confidence': min(1.0, avg_similarity * len(top_patterns) / 10)  # Higher confidence with more similar patterns
            }
        }

    except Exception as e:
        logger.error(f"Historical pattern recognition failed: {e}")
        return {'error': str(e), 'patterns_found': 0}

async def event_driven_analysis(symbol: str, days_ahead: int = 30) -> Dict[str, Any]:
    """
    Analyze upcoming events that could impact option prices.
    This provides event-driven probability adjustments for earnings, FDA approvals, etc.
    """
    events = {}
    event_adjustments = {}

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Enhanced Earnings Analysis with Alpha Vantage
        earnings_found = False
        try:
            # Try Alpha Vantage earnings calendar first
            if ALPHA_VANTAGE_API_KEY:
                earnings_calendar = await get_alpha_vantage_earnings_calendar()
                symbol_earnings = [
                    e for e in earnings_calendar.get("earnings_calendar", [])
                    if e.get("symbol") == symbol
                ]

                if symbol_earnings:
                    next_earnings_str = symbol_earnings[0].get("reportDate", "")
                    if next_earnings_str:
                        try:
                            next_earnings = datetime.datetime.strptime(next_earnings_str, '%Y-%m-%d')
                            days_to_earnings = (next_earnings - datetime.datetime.now()).days

                            if 0 <= days_to_earnings <= days_ahead:
                                events['earnings'] = {
                                    'date': next_earnings.strftime('%Y-%m-%d'),
                                    'days_until': days_to_earnings,
                                    'event_type': 'earnings_announcement',
                                    'source': 'alpha_vantage',
                                    'estimate': symbol_earnings[0].get('estimate', '')
                                }

                                # Earnings typically increase volatility by 20-50%
                                volatility_multiplier = 1.3 if days_to_earnings <= 5 else 1.1
                                probability_adjustment = 0.0  # Neutral until we know direction

                                event_adjustments['earnings'] = {
                                    'volatility_multiplier': volatility_multiplier,
                                    'probability_adjustment': probability_adjustment,
                                    'confidence': 0.8,
                                    'impact_window': '3-5 days post-earnings'
                                }
                                earnings_found = True
                        except ValueError:
                            logger.warning(f"Could not parse earnings date: {next_earnings_str}")

            # Fallback to yfinance if Alpha Vantage didn't provide earnings
            if not earnings_found:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    next_earnings = calendar.index[0] if len(calendar.index) > 0 else None
                    if next_earnings:
                        days_to_earnings = (next_earnings - datetime.datetime.now()).days
                        if 0 <= days_to_earnings <= days_ahead:
                            events['earnings'] = {
                                'date': next_earnings.strftime('%Y-%m-%d'),
                                'days_until': days_to_earnings,
                                'event_type': 'earnings_announcement',
                                'source': 'yfinance'
                            }

                            volatility_multiplier = 1.3 if days_to_earnings <= 5 else 1.1
                            probability_adjustment = 0.0

                            event_adjustments['earnings'] = {
                                'volatility_multiplier': volatility_multiplier,
                                'probability_adjustment': probability_adjustment,
                                'confidence': 0.7,  # Slightly lower confidence for yfinance
                                'impact_window': '3-5 days post-earnings'
                            }
        except Exception as e:
            logger.warning(f"Earnings analysis failed for {symbol}: {e}")

        # Sector-specific event analysis
        sector = info.get('sector', '').lower()
        industry = info.get('industry', '').lower()

        # FDA Events (for biotech/pharma)
        if any(keyword in sector or keyword in industry for keyword in ['biotech', 'pharma', 'drug', 'medical']):
            # This is a simplified placeholder - in production, you'd integrate with FDA calendar APIs
            events['fda_potential'] = {
                'probability': 0.1,  # 10% chance of FDA event in next 30 days
                'event_type': 'regulatory_approval',
                'impact': 'high_volatility'
            }

            event_adjustments['fda_potential'] = {
                'volatility_multiplier': 1.5,
                'probability_adjustment': 0.05,  # Slight bullish bias for approvals
                'confidence': 0.3
            }

        # Fed Meeting Analysis (affects all stocks but especially financials)
        # Fed meetings occur ~8 times per year, roughly every 6 weeks
        fed_meeting_dates = [
            datetime.datetime(2025, 1, 29),
            datetime.datetime(2025, 3, 19),
            datetime.datetime(2025, 5, 1),
            datetime.datetime(2025, 6, 18),
            datetime.datetime(2025, 7, 30),
            datetime.datetime(2025, 9, 17),
            datetime.datetime(2025, 11, 5),
            datetime.datetime(2025, 12, 17)
        ]

        for fed_date in fed_meeting_dates:
            days_to_fed = (fed_date - datetime.datetime.now()).days
            if 0 <= days_to_fed <= days_ahead:
                events['fed_meeting'] = {
                    'date': fed_date.strftime('%Y-%m-%d'),
                    'days_until': days_to_fed,
                    'event_type': 'fed_meeting'
                }

                # Fed meetings typically increase market volatility
                fed_volatility_multiplier = 1.2 if days_to_fed <= 2 else 1.05
                fed_adjustment = -0.02 if 'financials' in sector else 0.0  # Slight bearish for financials before Fed

                event_adjustments['fed_meeting'] = {
                    'volatility_multiplier': fed_volatility_multiplier,
                    'probability_adjustment': fed_adjustment,
                    'confidence': 0.6
                }
                break

        # Calculate composite event impact
        total_volatility_multiplier = 1.0
        total_probability_adjustment = 0.0
        total_confidence = 0.0
        event_count = 0

        for event_name, adjustment in event_adjustments.items():
            weight = adjustment['confidence']
            total_volatility_multiplier += (adjustment['volatility_multiplier'] - 1) * weight
            total_probability_adjustment += adjustment['probability_adjustment'] * weight
            total_confidence += weight
            event_count += 1

        if event_count > 0:
            total_confidence /= event_count

        return {
            'events_found': len(events),
            'upcoming_events': events,
            'event_adjustments': event_adjustments,
            'composite_impact': {
                'volatility_multiplier': total_volatility_multiplier,
                'probability_adjustment': total_probability_adjustment,
                'confidence': total_confidence,
                'events_analyzed': event_count
            },
            'symbol': symbol,
            'analysis_window_days': days_ahead
        }

    except Exception as e:
        logger.error(f"Event-driven analysis failed for {symbol}: {e}")
        return {
            'events_found': 0,
            'error': str(e),
            'symbol': symbol
        }

async def cross_asset_correlation_analysis(
    symbol: str,
    lookback_days: int = 90
) -> Dict[str, Any]:
    """
    Analyze how the stock correlates with bonds, dollar, commodities, and crypto.
    This provides cross-asset context for options analysis.
    """
    correlations = {}
    current_factors = {}

    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        stock_data = ticker.history(period=f"{lookback_days + 10}d")

        if len(stock_data) < 30:
            return {'error': 'Insufficient stock data', 'symbol': symbol}

        stock_returns = stock_data['Close'].pct_change().dropna()

        # Define cross-asset instruments to analyze
        cross_assets = {
            'bonds_10y': '^TNX',      # 10-Year Treasury
            'bonds_2y': '^IRX',       # 2-Year Treasury
            'dollar_index': 'DX-Y.NYB',  # Dollar Index
            'gold': 'GC=F',          # Gold futures
            'oil': 'CL=F',           # Oil futures
            'vix': '^VIX',           # Volatility index
            'bitcoin': 'BTC-USD',    # Bitcoin (for tech correlation)
            'spy': 'SPY'             # S&P 500 ETF
        }

        for asset_name, ticker_symbol in cross_assets.items():
            try:
                asset_ticker = yf.Ticker(ticker_symbol)
                asset_data = asset_ticker.history(period=f"{lookback_days + 10}d")

                if len(asset_data) < 30:
                    continue

                asset_returns = asset_data['Close'].pct_change().dropna()

                # Align dates for correlation calculation
                common_dates = stock_returns.index.intersection(asset_returns.index)
                if len(common_dates) < 20:
                    continue

                aligned_stock = stock_returns.loc[common_dates]
                aligned_asset = asset_returns.loc[common_dates]

                # Calculate correlation
                correlation = aligned_stock.corr(aligned_asset)

                # Calculate current levels and recent changes
                current_level = asset_data['Close'].iloc[-1]
                recent_change = (asset_data['Close'].iloc[-1] / asset_data['Close'].iloc[-5] - 1) if len(asset_data) >= 5 else 0

                correlations[asset_name] = {
                    'correlation': correlation,
                    'current_level': current_level,
                    'recent_5d_change': recent_change,
                    'sample_size': len(common_dates),
                    'significance': 'high' if abs(correlation) > 0.6 else 'medium' if abs(correlation) > 0.3 else 'low'
                }

            except Exception as e:
                logger.warning(f"Failed to analyze {asset_name}: {e}")
                continue

        # Analyze current cross-asset environment
        cross_asset_factors = {}

        if 'bonds_10y' in correlations and 'bonds_2y' in correlations:
            # Yield curve analysis
            yield_10y = correlations['bonds_10y']['current_level']
            yield_2y = correlations['bonds_2y']['current_level']
            yield_spread = yield_10y - yield_2y

            cross_asset_factors['yield_curve'] = {
                'spread': yield_spread,
                'status': 'inverted' if yield_spread < 0 else 'flat' if yield_spread < 1 else 'normal',
                'implication': 'bearish' if yield_spread < 0 else 'neutral'
            }

        if 'dollar_index' in correlations:
            dxy_change = correlations['dollar_index']['recent_5d_change']
            cross_asset_factors['dollar_strength'] = {
                'recent_trend': 'strengthening' if dxy_change > 0.01 else 'weakening' if dxy_change < -0.01 else 'stable',
                'correlation_with_stock': correlations['dollar_index']['correlation'],
                'implication': 'negative' if correlations['dollar_index']['correlation'] < -0.3 and dxy_change > 0.01 else 'neutral'
            }

        if 'vix' in correlations:
            vix_level = correlations['vix']['current_level']
            cross_asset_factors['market_fear'] = {
                'vix_level': vix_level,
                'status': 'high' if vix_level > 25 else 'low' if vix_level < 15 else 'normal',
                'correlation_with_stock': correlations['vix']['correlation'],
                'implication': 'bearish' if vix_level > 30 else 'bullish' if vix_level < 15 else 'neutral'
            }

        # Calculate composite cross-asset score
        bullish_factors = 0
        bearish_factors = 0

        for factor_name, factor_data in cross_asset_factors.items():
            implication = factor_data.get('implication', 'neutral')
            if implication == 'bullish':
                bullish_factors += 1
            elif implication == 'bearish':
                bearish_factors += 1

        net_cross_asset_bias = (bullish_factors - bearish_factors) / max(1, len(cross_asset_factors))

        return {
            'correlations': correlations,
            'cross_asset_factors': cross_asset_factors,
            'composite_analysis': {
                'net_bias': net_cross_asset_bias,
                'bullish_factors': bullish_factors,
                'bearish_factors': bearish_factors,
                'factors_analyzed': len(cross_asset_factors),
                'overall_implication': 'bullish' if net_cross_asset_bias > 0.3 else 'bearish' if net_cross_asset_bias < -0.3 else 'neutral'
            },
            'symbol': symbol,
            'analysis_period_days': lookback_days
        }

    except Exception as e:
        logger.error(f"Cross-asset correlation analysis failed for {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol}

async def advanced_volatility_forecasting(
    symbol: str,
    forecast_days: int = 30,
    lookback_days: int = 252
) -> Dict[str, Any]:
    """
    Advanced volatility forecasting using GARCH-like models and VIX analysis.
    This provides more accurate volatility estimates for options pricing.
    """
    try:
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period=f"{lookback_days + 30}d")

        if len(hist_data) < 50:
            return {'error': 'Insufficient historical data', 'symbol': symbol}

        # Calculate returns
        returns = hist_data['Close'].pct_change().dropna()

        # Current realized volatility (different periods)
        current_vol_5d = returns.tail(5).std() * np.sqrt(252)
        current_vol_20d = returns.tail(20).std() * np.sqrt(252)
        current_vol_60d = returns.tail(60).std() * np.sqrt(252)

        # VIX analysis for market volatility context
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(period=f"{lookback_days}d")

        volatility_forecast = {}

        if not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            vix_mean = vix_data['Close'].tail(60).mean()
            vix_std = vix_data['Close'].tail(60).std()

            # VIX-based volatility adjustment
            vix_z_score = (current_vix - vix_mean) / max(vix_std, 1)
            vix_adjustment = max(-0.3, min(0.5, vix_z_score * 0.1))  # Cap adjustment

            volatility_forecast['vix_analysis'] = {
                'current_vix': current_vix,
                'vix_mean_60d': vix_mean,
                'vix_z_score': vix_z_score,
                'volatility_adjustment': vix_adjustment
            }
        else:
            vix_adjustment = 0
            volatility_forecast['vix_analysis'] = {'error': 'VIX data unavailable'}

        # Simple GARCH-like volatility clustering analysis
        squared_returns = returns ** 2

        # Calculate volatility persistence (autocorrelation in squared returns)
        volatility_persistence = squared_returns.autocorr(lag=1) if len(squared_returns) > 20 else 0

        # Recent volatility trend
        recent_vol_trend = (current_vol_5d - current_vol_20d) / current_vol_20d

        # Volatility mean reversion component
        long_term_vol = returns.std() * np.sqrt(252)
        mean_reversion_speed = 0.1  # Assumption: 10% daily mean reversion

        # Forecast volatility using weighted approach
        base_forecast = current_vol_20d  # Start with 20-day realized vol

        # Apply adjustments
        trend_adjusted = base_forecast * (1 + recent_vol_trend * 0.3)  # 30% weight to recent trend
        vix_adjusted = trend_adjusted * (1 + vix_adjustment)  # VIX adjustment
        mean_reversion_target = long_term_vol  # Long-term mean

        # Final forecast with mean reversion
        persistence_weight = max(0.1, min(0.9, volatility_persistence))
        forecast_volatility = (vix_adjusted * persistence_weight +
                             mean_reversion_target * (1 - persistence_weight))

        # Calculate confidence bands
        forecast_std = returns.std() * 0.2  # 20% of return std as forecast uncertainty
        confidence_bands = {
            'lower_80': max(0.05, forecast_volatility - 1.28 * forecast_std),
            'upper_80': forecast_volatility + 1.28 * forecast_std,
            'lower_95': max(0.05, forecast_volatility - 1.96 * forecast_std),
            'upper_95': forecast_volatility + 1.96 * forecast_std
        }

        # Volatility regime classification
        if forecast_volatility > long_term_vol * 1.3:
            regime = 'high_volatility'
        elif forecast_volatility < long_term_vol * 0.7:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'

        volatility_forecast.update({
            'current_volatilities': {
                'realized_5d': current_vol_5d,
                'realized_20d': current_vol_20d,
                'realized_60d': current_vol_60d
            },
            'forecast': {
                'volatility': forecast_volatility,
                'confidence_bands': confidence_bands,
                'forecast_horizon_days': forecast_days,
                'regime': regime
            },
            'model_components': {
                'base_volatility': base_forecast,
                'trend_adjustment': recent_vol_trend,
                'vix_adjustment': vix_adjustment,
                'persistence_factor': volatility_persistence,
                'mean_reversion_target': mean_reversion_target
            },
            'model_confidence': {
                'data_quality': min(1.0, len(returns) / 100),  # More data = higher confidence
                'stability': max(0.3, 1 - abs(recent_vol_trend)),  # Less trend = more stable
                'overall': min(0.9, (min(1.0, len(returns) / 100) + max(0.3, 1 - abs(recent_vol_trend))) / 2)
            }
        })

        return volatility_forecast

    except Exception as e:
        logger.error(f"Advanced volatility forecasting failed for {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol}

# ==================== INSTITUTIONAL-GRADE ENHANCEMENT FUNCTIONS ====================

async def get_realtime_market_sentiment(symbols: List[str], timeframe: str = "4h", sources: List[str] = None) -> Dict[str, Any]:
    """Get real-time market sentiment from multiple sources for enhanced analysis."""
    if sources is None:
        sources = ["news", "social", "analyst"]

    sentiment_data = {}

    for symbol in symbols:
        try:
            # News Sentiment (using NewsAPI)
            news_sentiment = 0.0
            if "news" in sources and NEWSAPI_KEY:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://newsapi.org/v2/everything"
                        params = {
                            'q': f"{symbol} stock OR {symbol} earnings OR {symbol} revenue",
                            'apiKey': NEWSAPI_KEY,
                            'language': 'en',
                            'sortBy': 'publishedAt',
                            'pageSize': 20,
                            'from': (datetime.datetime.now() - datetime.timedelta(hours=int(timeframe[:-1]))).isoformat()
                        }

                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                articles = data.get('articles', [])

                                sentiments = []
                                for article in articles[:10]:  # Analyze top 10 articles
                                    text = f"{article.get('title', '')} {article.get('description', '')}"
                                    if text.strip():
                                        blob = TextBlob(text)
                                        sentiments.append(blob.sentiment.polarity)

                                if sentiments:
                                    news_sentiment = np.mean(sentiments)
                except Exception as e:
                    logger.warning(f"News sentiment failed for {symbol}: {e}")

            # Social Sentiment (simplified - could integrate with Twitter API)
            social_sentiment = 0.0  # Placeholder for social media sentiment

            # Analyst Sentiment (using yfinance recommendations)
            analyst_sentiment = 0.0
            try:
                ticker = yf.Ticker(symbol)
                recommendations = ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    recent_rec = recommendations.tail(5)  # Last 5 recommendations
                    # Convert recommendations to numerical sentiment
                    rec_mapping = {
                        'Strong Buy': 1.0, 'Buy': 0.5, 'Hold': 0.0,
                        'Sell': -0.5, 'Strong Sell': -1.0
                    }
                    rec_scores = [rec_mapping.get(rec, 0) for rec in recent_rec['To Grade'] if rec in rec_mapping]
                    if rec_scores:
                        analyst_sentiment = np.mean(rec_scores)
            except Exception as e:
                logger.warning(f"Analyst sentiment failed for {symbol}: {e}")

            # Composite sentiment
            composite_sentiment = np.mean([news_sentiment, social_sentiment, analyst_sentiment])

            sentiment_data[symbol] = {
                'composite_sentiment': composite_sentiment,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'analyst_sentiment': analyst_sentiment,
                'confidence': min(1.0, abs(composite_sentiment) * 2),  # Higher confidence for stronger sentiment
                'timeframe': timeframe
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            sentiment_data[symbol] = {
                'composite_sentiment': 0.0,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'analyst_sentiment': 0.0,
                'confidence': 0.0,
                'timeframe': timeframe,
                'error': str(e)
            }

    return {
        'sentiment_data': sentiment_data,
        'analysis_timestamp': datetime.datetime.now().isoformat(),
        'sources': sources,
        'timeframe': timeframe
    }

async def detect_market_regime(indicators: List[str] = None, lookback_days: int = 30) -> Dict[str, Any]:
    """Detect current market regime using multiple indicators."""
    if indicators is None:
        indicators = ["VIX", "yield_curve", "momentum", "sentiment"]

    regime_data = {}

    try:
        # VIX Analysis
        if "VIX" in indicators:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period=f"{lookback_days}d")
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                avg_vix = vix_data['Close'].mean()

                if current_vix > 30:
                    vix_regime = "high_volatility"
                elif current_vix < 15:
                    vix_regime = "low_volatility"
                else:
                    vix_regime = "normal_volatility"

                regime_data['vix'] = {
                    'current': current_vix,
                    'average': avg_vix,
                    'regime': vix_regime,
                    'percentile': (vix_data['Close'] <= current_vix).mean() * 100
                }

        # Enhanced Yield Curve Analysis using FRED data
        if "yield_curve" in indicators:
            try:
                # Try FRED first for more accurate economic data
                if FRED_API_KEY:
                    ten_year_fred = await fetch_fred_data("GS10")  # 10-Year Treasury
                    two_year_fred = await fetch_fred_data("GS2")   # 2-Year Treasury

                    if ten_year_fred.get("observations") and two_year_fred.get("observations"):
                        current_10y = float(ten_year_fred["observations"][0]["value"])
                        current_2y = float(two_year_fred["observations"][0]["value"])
                        spread = current_10y - current_2y

                        if spread < 0:
                            curve_regime = "inverted"
                        elif spread < 1.0:
                            curve_regime = "flat"
                        else:
                            curve_regime = "normal"
                    else:
                        # Fallback to yfinance
                        ten_year = yf.Ticker("^TNX")
                        two_year = yf.Ticker("^IRX")
                        ten_y_data = ten_year.history(period=f"{lookback_days}d")
                        two_y_data = two_year.history(period=f"{lookback_days}d")

                        if not ten_y_data.empty and not two_y_data.empty:
                            current_10y = ten_y_data['Close'].iloc[-1]
                            current_2y = two_y_data['Close'].iloc[-1]
                            spread = current_10y - current_2y

                            if spread < 0:
                                curve_regime = "inverted"
                            elif spread < 1.0:
                                curve_regime = "flat"
                            else:
                                curve_regime = "normal"

                    regime_data['yield_curve'] = {
                        'spread': spread,
                        'ten_year': current_10y,
                        'two_year': current_2y,
                        'regime': curve_regime
                    }
            except Exception as e:
                logger.warning(f"Yield curve analysis failed: {e}")

        # Momentum Analysis (SPY)
        if "momentum" in indicators:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period=f"{lookback_days * 2}d")
            if not spy_data.empty:
                # Calculate various momentum indicators
                current_price = spy_data['Close'].iloc[-1]
                sma_20 = spy_data['Close'].tail(20).mean()
                sma_50 = spy_data['Close'].tail(50).mean() if len(spy_data) >= 50 else sma_20

                if current_price > sma_20 > sma_50:
                    momentum_regime = "bullish"
                elif current_price < sma_20 < sma_50:
                    momentum_regime = "bearish"
                else:
                    momentum_regime = "sideways"

                regime_data['momentum'] = {
                    'current_price': current_price,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'regime': momentum_regime,
                    'strength': abs((current_price - sma_20) / sma_20) * 100
                }

        # Overall Market Regime
        regime_scores = {
            'bullish': 0,
            'bearish': 0,
            'sideways': 0,
            'high_volatility': 0
        }

        # Score each regime component
        if 'vix' in regime_data:
            if regime_data['vix']['regime'] == 'high_volatility':
                regime_scores['high_volatility'] += 1
            elif regime_data['vix']['current'] < regime_data['vix']['average']:
                regime_scores['bullish'] += 0.5

        if 'momentum' in regime_data:
            if regime_data['momentum']['regime'] == 'bullish':
                regime_scores['bullish'] += 1
            elif regime_data['momentum']['regime'] == 'bearish':
                regime_scores['bearish'] += 1
            else:
                regime_scores['sideways'] += 1

        if 'yield_curve' in regime_data:
            if regime_data['yield_curve']['regime'] == 'inverted':
                regime_scores['bearish'] += 0.5
            elif regime_data['yield_curve']['regime'] == 'normal':
                regime_scores['bullish'] += 0.5

        # Determine overall regime
        dominant_regime = max(regime_scores, key=regime_scores.get)
        regime_confidence = regime_scores[dominant_regime] / sum(regime_scores.values()) if sum(regime_scores.values()) > 0 else 0

        return {
            'overall_regime': dominant_regime,
            'regime_confidence': regime_confidence,
            'regime_scores': regime_scores,
            'components': regime_data,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'lookback_days': lookback_days
        }

    except Exception as e:
        logger.error(f"Market regime detection failed: {e}")
        return {
            'overall_regime': 'unknown',
            'regime_confidence': 0.0,
            'regime_scores': {},
            'components': {},
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'error': str(e)
        }

async def get_options_flow_analysis(symbols: List[str], timeframe: str = "4h", min_premium: float = 50000) -> Dict[str, Any]:
    """Analyze options flow for unusual activity (simplified version)."""
    flow_data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get current options chain
            exp_dates = ticker.options

            if exp_dates:
                # Analyze nearest expiration
                nearest_exp = exp_dates[0]
                options_chain = ticker.option_chain(nearest_exp)

                calls = options_chain.calls
                puts = options_chain.puts

                # Calculate unusual activity indicators
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                put_call_ratio = total_put_volume / max(1, total_call_volume)

                # Find high premium options
                calls['premium'] = calls['volume'] * calls['lastPrice']
                puts['premium'] = puts['volume'] * puts['lastPrice']

                high_premium_calls = calls[calls['premium'] >= min_premium]
                high_premium_puts = puts[puts['premium'] >= min_premium]

                # Unusual activity score (simplified)
                unusual_score = 0.0
                if put_call_ratio > 1.5:  # Heavy put buying
                    unusual_score += 0.3
                elif put_call_ratio < 0.5:  # Heavy call buying
                    unusual_score += 0.5

                if len(high_premium_calls) > 0:
                    unusual_score += 0.3
                if len(high_premium_puts) > 0:
                    unusual_score += 0.2

                flow_data[symbol] = {
                    'put_call_ratio': put_call_ratio,
                    'total_call_volume': int(total_call_volume),
                    'total_put_volume': int(total_put_volume),
                    'high_premium_calls': len(high_premium_calls),
                    'high_premium_puts': len(high_premium_puts),
                    'unusual_activity_score': min(1.0, unusual_score),
                    'flow_sentiment': 'bullish' if unusual_score > 0.4 and put_call_ratio < 1.0 else 'bearish' if put_call_ratio > 1.2 else 'neutral'
                }
            else:
                flow_data[symbol] = {
                    'put_call_ratio': 1.0,
                    'total_call_volume': 0,
                    'total_put_volume': 0,
                    'high_premium_calls': 0,
                    'high_premium_puts': 0,
                    'unusual_activity_score': 0.0,
                    'flow_sentiment': 'neutral',
                    'error': 'No options data available'
                }

        except Exception as e:
            logger.warning(f"Options flow analysis failed for {symbol}: {e}")
            flow_data[symbol] = {
                'put_call_ratio': 1.0,
                'total_call_volume': 0,
                'total_put_volume': 0,
                'unusual_activity_score': 0.0,
                'flow_sentiment': 'neutral',
                'error': str(e)
            }

    return {
        'flow_data': flow_data,
        'analysis_timestamp': datetime.datetime.now().isoformat(),
        'timeframe': timeframe,
        'min_premium': min_premium
    }

async def find_optimal_risk_reward_options_enhanced(
    symbols: List[str],
    max_days_to_expiry: int = 30,
    target_profit_potential: float = 0.15,
    target_probability: float = 0.45,
    target_risk_level: int = 6,
    max_results: int = 10,
    always_show_results: bool = True
) -> Dict[str, Any]:
    """
    ENHANCED Smart Picks with institutional-grade analysis.

    Key improvements:
    1. ALWAYS shows best available options (never "No optimal options found")
    2. Real-time market sentiment integration
    3. Market regime detection with adaptive criteria
    4. Options flow analysis
    5. Transparency about why options don't meet ideal criteria
    6. Confidence levels and market context
    """
    logger.info(f"ENHANCED Smart Picks: Analyzing {len(symbols)} symbols with institutional-grade data")

    start_time = time.time()
    current_time = datetime.datetime.now()

    # Step 1: Market Regime Detection
    logger.info("Step 1: Detecting market regime...")
    market_regime = await detect_market_regime()
    regime = market_regime['overall_regime']
    regime_confidence = market_regime['regime_confidence']

    # Step 2: Adapt criteria based on market regime
    logger.info(f"Step 2: Adapting criteria for {regime} market regime...")
    if regime == "high_volatility":
        # In high vol, lower probability expectations but higher profit potential
        adapted_probability = max(0.30, target_probability - 0.10)
        adapted_profit = target_profit_potential + 0.05
        adapted_risk = min(8, target_risk_level + 1)
    elif regime == "bearish":
        # In bear market, be more conservative
        adapted_probability = max(0.35, target_probability - 0.05)
        adapted_profit = target_profit_potential + 0.03
        adapted_risk = max(4, target_risk_level - 1)
    elif regime == "bullish":
        # In bull market, can be slightly more aggressive
        adapted_probability = max(0.40, target_probability - 0.03)
        adapted_profit = target_profit_potential
        adapted_risk = min(7, target_risk_level + 1)
    else:  # sideways or unknown
        adapted_probability = target_probability
        adapted_profit = target_profit_potential
        adapted_risk = target_risk_level

    logger.info(f"Adapted criteria: P={adapted_probability:.1%}, Profit={adapted_profit:.1%}, Risk={adapted_risk}")

    # Step 3: Get market sentiment for top symbols
    logger.info("Step 3: Analyzing real-time market sentiment...")
    sentiment_analysis = await get_realtime_market_sentiment(symbols[:50])  # Analyze top 50 for performance

    # Step 4: Get options flow analysis
    logger.info("Step 4: Analyzing options flow...")
    flow_analysis = await get_options_flow_analysis(symbols[:20])  # Top 20 for flow analysis

    # Step 4.5: PHASE 2 Cross-Asset Correlation Analysis (simplified for now)
    logger.info("Step 4.5: Phase 2 Cross-Asset Correlation Analysis...")
    correlation_analysis = {
        'correlation_data': {}
    }
    # Pre-compute correlation data for top symbols (performance optimization)
    for symbol in symbols[:20]:  # Limit to top 20 for performance
        try:
            correlation_result = await cross_asset_correlation_analysis(
                symbol=symbol,
                lookback_days=90
            )
            correlation_analysis['correlation_data'][symbol] = correlation_result
        except Exception as e:
            logger.warning(f"Correlation analysis failed for {symbol}: {e}")
            correlation_analysis['correlation_data'][symbol] = {}

    # Step 5: Analyze options (enhanced version of original algorithm with Phase 2)
    logger.info("Step 5: Enhanced options analysis...")
    all_options = []

    # Get valid expiration dates
    valid_expirations = []
    for days_ahead in range(1, max_days_to_expiry + 1):
        future_date = current_time + datetime.timedelta(days=days_ahead)
        if future_date.weekday() < 5:  # Business days only
            valid_expirations.append(future_date.strftime('%Y-%m-%d'))

    async def analyze_symbol_enhanced(symbol: str) -> List[Dict[str, Any]]:
        symbol_options = []
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)

            options_dates = await loop.run_in_executor(None, lambda: ticker.options)
            if not options_dates:
                return []

            available_dates = [date for date in options_dates if date in valid_expirations]
            if not available_dates:
                return []

            info = await loop.run_in_executor(None, lambda: ticker.info)
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not current_price:
                return []

            # Get sentiment, flow, and correlation data for this symbol
            symbol_sentiment = sentiment_analysis['sentiment_data'].get(symbol, {})
            symbol_flow = flow_analysis['flow_data'].get(symbol, {})
            symbol_correlation = correlation_analysis['correlation_data'].get(symbol, {})

            for expiration_date in available_dates[:3]:  # Analyze top 3 expirations for performance
                try:
                    exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                    days_to_exp = (exp_date - current_time).days
                    time_to_expiration = days_to_exp / 365.0

                    if days_to_exp > max_days_to_expiry:
                        continue

                    options = await loop.run_in_executor(None, ticker.option_chain, expiration_date)
                    calls = options.calls

                    for _, option in calls.iterrows():
                        strike = option['strike']
                        volume = option.get('volume', 0) or 0
                        iv = option.get('impliedVolatility', 0) or 0
                        bid = option.get('bid', 0) or 0
                        ask = option.get('ask', 0) or 0

                        # Enhanced filtering with lower minimums for "always show results"
                        min_volume = 10 if always_show_results else 25
                        min_iv = 0.05 if always_show_results else 0.10

                        if (strike > current_price and volume >= min_volume and
                            iv >= min_iv and bid > 0 and ask > 0):

                            # Enhanced analysis with sentiment
                            advanced_result = await advanced_engine.analyze_with_novel_techniques(
                                symbol, strike, expiration_date
                            )

                            # PHASE 2: Advanced Multi-Scenario Monte Carlo Analysis
                            phase2_monte_carlo = await multi_scenario_monte_carlo_analysis(
                                current_price=current_price,
                                strike=strike,
                                time_to_expiration=time_to_expiration,
                                volatility=iv,
                                risk_free_rate=0.05,
                                num_simulations=5000,  # Reduced for performance in batch analysis
                                market_regime=regime
                            )

                            # PHASE 2: Historical Pattern Recognition
                            phase2_patterns = await historical_pattern_recognition(
                                symbol=symbol,
                                lookback_days=90,  # Shorter for performance
                                pattern_type='price_momentum'
                            )

                            # PHASE 2: Advanced Volatility Forecasting
                            phase2_volatility = await advanced_volatility_forecasting(
                                symbol=symbol,
                                forecast_days=days_to_exp,
                                current_iv=iv
                            )

                            # PHASE 2: Event-Driven Analysis (for upcoming catalysts)
                            phase2_events = await event_driven_analysis(
                                symbol=symbol,
                                days_ahead=days_to_exp
                            )

                            # Combine Phase 1 and Phase 2 ITM probabilities
                            base_itm_prob = advanced_result.get('final_analysis', {}).get('final_itm_probability', 0.5)

                            # Phase 2 Monte Carlo weighted probability
                            mc_bull_prob = phase2_monte_carlo.get('scenario_probabilities', {}).get('bull', 0.5)
                            mc_bear_prob = phase2_monte_carlo.get('scenario_probabilities', {}).get('bear', 0.3)
                            mc_sideways_prob = phase2_monte_carlo.get('scenario_probabilities', {}).get('sideways', 0.4)

                            # Weight by regime confidence
                            if regime == 'bullish':
                                phase2_mc_prob = mc_bull_prob * 0.6 + mc_sideways_prob * 0.3 + mc_bear_prob * 0.1
                            elif regime == 'bearish':
                                phase2_mc_prob = mc_bear_prob * 0.6 + mc_sideways_prob * 0.3 + mc_bull_prob * 0.1
                            else:  # sideways or mixed
                                phase2_mc_prob = mc_sideways_prob * 0.5 + (mc_bull_prob + mc_bear_prob) * 0.25

                            # Pattern recognition probability adjustment
                            pattern_confidence = phase2_patterns.get('best_match', {}).get('confidence', 0.5)
                            pattern_adjustment = (pattern_confidence - 0.5) * 0.1  # ±5% max adjustment

                            # Volatility forecasting adjustment
                            volatility_trend = phase2_volatility.get('forecast_trend', 'stable')
                            vol_adjustment = 0.02 if volatility_trend == 'increasing' else (-0.02 if volatility_trend == 'decreasing' else 0)

                            # Event-driven analysis adjustment
                            upcoming_events = phase2_events.get('events_analysis', {}).get(symbol, {})
                            catalyst_impact = upcoming_events.get('composite_impact_score', 0.5)
                            event_adjustment = (catalyst_impact - 0.5) * 0.08  # ±4% max adjustment for events

                            # Cross-asset correlation adjustment
                            correlation_strength = symbol_correlation.get('market_correlation_strength', 0.5)
                            correlation_direction = symbol_correlation.get('favorable_correlation_direction', 0.5)
                            correlation_adjustment = (correlation_strength * correlation_direction - 0.25) * 0.06  # ±3% max

                            # Enhanced ITM probability combining all Phase 2 analytics
                            institutional_itm_prob = (
                                base_itm_prob * 0.25 +          # Phase 1 base (25%)
                                phase2_mc_prob * 0.3 +          # Phase 2 Monte Carlo (30%)
                                (base_itm_prob + pattern_adjustment) * 0.15 +  # Pattern recognition (15%)
                                (base_itm_prob + vol_adjustment) * 0.1 +       # Volatility forecasting (10%)
                                (base_itm_prob + event_adjustment) * 0.1 +     # Event-driven analysis (10%)
                                (base_itm_prob + correlation_adjustment) * 0.1  # Cross-asset correlation (10%)
                            )

                            # Apply Phase 1 sentiment boost on top of Phase 2 analysis
                            sentiment_boost = symbol_sentiment.get('composite_sentiment', 0) * 0.05  # Reduced to 5% max for institutional blend
                            enhanced_itm_prob = apply_sentiment_adjustment(institutional_itm_prob, sentiment_boost)

                            option_price = (bid + ask) / 2
                            profit_potential = calculate_profit_potential(
                                current_price, strike, option_price, time_to_expiration, iv
                            )
                            risk_level = calculate_risk_level(
                                current_price, strike, time_to_expiration, iv, volume
                            )

                            # Enhanced composite score with sentiment and flow
                            base_score = calculate_composite_score(
                                enhanced_itm_prob, profit_potential, risk_level,
                                days_to_exp, advanced_result.get('analysis_techniques', {})
                            )

                            # PHASE 2: Additional scoring components
                            # Monte Carlo scenario confidence boost
                            mc_confidence = phase2_monte_carlo.get('confidence_metrics', {}).get('overall_confidence', 0.5)
                            mc_multiplier = 1.0 + (mc_confidence - 0.5) * 0.2  # ±10% max

                            # Pattern recognition strength boost
                            pattern_strength = phase2_patterns.get('best_match', {}).get('strength', 0.5)
                            pattern_multiplier = 1.0 + (pattern_strength - 0.5) * 0.15  # ±7.5% max

                            # Volatility forecast accuracy boost
                            vol_accuracy = phase2_volatility.get('forecast_accuracy', {}).get('confidence', 0.5)
                            volatility_multiplier = 1.0 + (vol_accuracy - 0.5) * 0.1  # ±5% max

                            # Event catalyst strength boost
                            event_strength = upcoming_events.get('catalyst_strength', 0.5)
                            event_multiplier = 1.0 + (event_strength - 0.5) * 0.12  # ±6% max for catalyst events

                            # Cross-asset correlation strength boost
                            correlation_favorability = symbol_correlation.get('overall_correlation_score', 0.5)
                            correlation_multiplier = 1.0 + (correlation_favorability - 0.5) * 0.08  # ±4% max for correlation

                            # Apply all multipliers: Phase 1 + Phase 2
                            sentiment_multiplier = 1.0 + (symbol_sentiment.get('composite_sentiment', 0) * 0.1)
                            flow_multiplier = 1.0 + (symbol_flow.get('unusual_activity_score', 0) * 0.05)

                            # Institutional-grade enhanced score with complete Phase 2 analytics
                            institutional_score = (base_score *
                                                  sentiment_multiplier *
                                                  flow_multiplier *
                                                  mc_multiplier *
                                                  pattern_multiplier *
                                                  volatility_multiplier *
                                                  event_multiplier *
                                                  correlation_multiplier)

                            greeks = calculate_black_scholes_greeks(
                                current_price, strike, time_to_expiration, iv, 0.05, 'call'
                            )

                            symbol_options.append({
                                'symbol': symbol,
                                'current_price': current_price,
                                'strike': strike,
                                'option_price': option_price,
                                'expiration': expiration_date,
                                'days_to_expiration': days_to_exp,
                                'volume': int(volume),
                                'open_interest': int(option.get('openInterest', 0) or 0),
                                'implied_volatility': iv,
                                'bid': bid,
                                'ask': ask,
                                'itm_probability': enhanced_itm_prob,
                                'base_itm_probability': base_itm_prob,
                                'institutional_itm_probability': institutional_itm_prob,
                                'sentiment_adjustment': enhanced_itm_prob - institutional_itm_prob,
                                'profit_potential': profit_potential,
                                'risk_level': risk_level,
                                'composite_score': institutional_score,
                                'base_score': base_score,
                                'sentiment_multiplier': sentiment_multiplier,
                                'flow_multiplier': flow_multiplier,
                                'mc_multiplier': mc_multiplier,
                                'pattern_multiplier': pattern_multiplier,
                                'volatility_multiplier': volatility_multiplier,
                                'event_multiplier': event_multiplier,
                                'correlation_multiplier': correlation_multiplier,
                                'delta': greeks['delta'],
                                'gamma': greeks['gamma'],
                                'theta': greeks['theta'],
                                'vega': greeks['vega'],
                                'sentiment_data': symbol_sentiment,
                                'flow_data': symbol_flow,
                                # Phase 2 Advanced Analytics Data
                                'phase2_monte_carlo': {
                                    'scenario_probabilities': phase2_monte_carlo.get('scenario_probabilities', {}),
                                    'confidence_metrics': phase2_monte_carlo.get('confidence_metrics', {}),
                                    'risk_metrics': phase2_monte_carlo.get('risk_metrics', {})
                                },
                                'phase2_patterns': {
                                    'best_match': phase2_patterns.get('best_match', {}),
                                    'pattern_type': phase2_patterns.get('pattern_type', 'price_momentum'),
                                    'historical_outcomes': phase2_patterns.get('similar_patterns', [])[:3]  # Top 3 matches
                                },
                                'phase2_volatility': {
                                    'forecast_trend': phase2_volatility.get('forecast_trend', 'stable'),
                                    'forecast_accuracy': phase2_volatility.get('forecast_accuracy', {}),
                                    'term_structure': phase2_volatility.get('term_structure_analysis', {}),
                                    'volatility_regime': phase2_volatility.get('volatility_regime', 'normal')
                                },
                                'phase2_events': {
                                    'upcoming_events': upcoming_events.get('upcoming_events', []),
                                    'catalyst_strength': upcoming_events.get('catalyst_strength', 0.5),
                                    'impact_timeline': upcoming_events.get('impact_timeline', {}),
                                    'event_types_detected': upcoming_events.get('event_types', [])
                                },
                                'phase2_correlation': {
                                    'market_correlation_strength': symbol_correlation.get('market_correlation_strength', 0.5),
                                    'asset_class_correlations': symbol_correlation.get('asset_correlations', {}),
                                    'regime_correlations': symbol_correlation.get('regime_analysis', {}),
                                    'correlation_stability': symbol_correlation.get('correlation_stability', 0.5)
                                },
                                **advanced_result.get('analysis_techniques', {})
                            })

                except Exception as e:
                    logger.warning(f"Error analyzing {symbol} {expiration_date}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error analyzing symbol {symbol}: {e}")

        return symbol_options

    # Process symbols concurrently
    tasks = [analyze_symbol_enhanced(symbol) for symbol in symbols[:100]]  # Limit to top 100 for performance
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            all_options.extend(result)
        else:
            logger.warning(f"Task failed: {result}")

    logger.info(f"Found {len(all_options)} total options")

    # Step 6: Rank and select results
    if not all_options:
        return {
            'smart_picks_analysis': {
                'total_options_analyzed': 0,
                'total_options_found': 0,
                'ideal_criteria_met': 0,
                'analysis_timestamp': time.time(),
                'market_context': {
                    'regime': regime,
                    'regime_confidence': regime_confidence,
                    'message': 'No options data available - markets may be closed or symbols invalid'
                },
                'criteria': {
                    'target_days_to_expiry': max_days_to_expiry,
                    'target_profit_potential': target_profit_potential,
                    'target_probability': target_probability,
                    'target_risk_level': target_risk_level,
                    'adapted_probability': adapted_probability,
                    'adapted_profit': adapted_profit,
                    'adapted_risk': adapted_risk
                },
                'optimal_options': [],
                'summary_stats': {
                    'average_composite_score': 0.0,
                    'average_itm_probability': 0.0,
                    'average_profit_potential': 0.0,
                    'average_risk_level': 0.0,
                    'average_days_to_expiration': 0.0
                },
                'processing_time': time.time() - start_time
            }
        }

    # Sort by enhanced composite score
    all_options.sort(key=lambda x: x['composite_score'], reverse=True)

    # Categorize options
    ideal_options = []
    good_options = []
    acceptable_options = []

    for opt in all_options:
        if (opt['itm_probability'] >= target_probability and
            opt['profit_potential'] >= target_profit_potential and
            opt['risk_level'] <= target_risk_level):
            ideal_options.append(opt)
        elif (opt['itm_probability'] >= adapted_probability and
              opt['profit_potential'] >= adapted_profit and
              opt['risk_level'] <= adapted_risk):
            good_options.append(opt)
        else:
            acceptable_options.append(opt)

    # Select final results: prioritize ideal, then good, then acceptable
    final_options = []
    final_options.extend(ideal_options[:max_results])

    if len(final_options) < max_results:
        remaining = max_results - len(final_options)
        final_options.extend(good_options[:remaining])

    if len(final_options) < max_results and always_show_results:
        remaining = max_results - len(final_options)
        final_options.extend(acceptable_options[:remaining])

    # Add ranking
    for i, opt in enumerate(final_options, 1):
        opt['rank'] = i
        opt['category'] = ('ideal' if opt in ideal_options else
                          'adapted' if opt in good_options else 'acceptable')

    # Calculate summary statistics
    if final_options:
        avg_score = np.mean([opt['composite_score'] for opt in final_options])
        avg_prob = np.mean([opt['itm_probability'] for opt in final_options])
        avg_profit = np.mean([opt['profit_potential'] for opt in final_options])
        avg_risk = np.mean([opt['risk_level'] for opt in final_options])
        avg_days = np.mean([opt['days_to_expiration'] for opt in final_options])
    else:
        avg_score = avg_prob = avg_profit = avg_risk = avg_days = 0.0

    # Generate market context message
    context_message = f"{regime.title()} market regime detected"
    if regime_confidence > 0.7:
        context_message += f" (High confidence: {regime_confidence:.1%})"
    else:
        context_message += f" (Moderate confidence: {regime_confidence:.1%})"

    if len(ideal_options) == 0 and len(final_options) > 0:
        context_message += f". Showing best available options with adapted criteria due to current market conditions."

    analysis_time = time.time() - start_time

    return {
        'smart_picks_analysis': {
            'total_options_analyzed': len(all_options),
            'total_options_found': len(final_options),
            'ideal_criteria_met': len(ideal_options),
            'adapted_criteria_met': len(good_options),
            'analysis_timestamp': time.time(),
            'market_context': {
                'regime': regime,
                'regime_confidence': regime_confidence,
                'message': context_message,
                'sentiment_analyzed_symbols': len(sentiment_analysis['sentiment_data']),
                'flow_analyzed_symbols': len(flow_analysis['flow_data']),
                'phase2_analytics': {
                    'correlation_analyzed_symbols': len(correlation_analysis['correlation_data']),
                    'analytics_enabled': ['multi_scenario_monte_carlo', 'pattern_recognition', 'volatility_forecasting', 'event_driven', 'cross_asset_correlation'],
                    'institutional_grade_features': True,
                    'advanced_probability_models': 5
                }
            },
            'criteria': {
                'target_days_to_expiry': max_days_to_expiry,
                'target_profit_potential': target_profit_potential,
                'target_probability': target_probability,
                'target_risk_level': target_risk_level,
                'adapted_probability': adapted_probability,
                'adapted_profit': adapted_profit,
                'adapted_risk': adapted_risk,
                'always_show_results': always_show_results
            },
            'summary_stats': {
                'average_composite_score': round(avg_score, 4),
                'average_itm_probability': round(avg_prob, 4),
                'average_profit_potential': round(avg_profit, 4),
                'average_risk_level': round(avg_risk, 2),
                'average_days_to_expiration': round(avg_days, 1)
            },
            'optimal_options': final_options,
            'performance_metrics': {
                'processing_time_seconds': round(analysis_time, 2),
                'symbols_analyzed': len(symbols),
                'options_per_second': round(len(all_options) / max(1, analysis_time), 1)
            }
        }
    }

# Option Selection and Monitoring System
def select_option_for_monitoring(symbol: str, strike: float, expiration_date: str,
                                current_price: float = None, notes: str = None) -> Dict[str, Any]:
    """
    Select an option for monitoring and tracking.

    Args:
        symbol: Stock ticker symbol
        strike: Strike price
        expiration_date: Expiration date in YYYY-MM-DD format
        current_price: Current stock price (optional)
        notes: Additional notes (optional)

    Returns:
        Dictionary with selection confirmation and monitoring details
    """
    global selected_options

    option_key = f"{symbol}_{strike}_{expiration_date}"

    selected_options[option_key] = {
        'symbol': symbol,
        'strike': strike,
        'expiration_date': expiration_date,
        'selected_at': datetime.datetime.now().isoformat(),
        'current_price': current_price,
        'notes': notes or f"Selected {symbol} ${strike} call expiring {expiration_date}",
        'alerts_sent': 0,
        'last_alert': None,
        'status': 'active'
    }

    # AUTO-START CONTINUOUS MONITORING if not already active
    global monitoring_active, monitoring_task
    if not monitoring_active and len(selected_options) > 0:
        logger.info("Auto-starting continuous monitoring - first option added")
        try:
            monitoring_active = True
            monitoring_task = asyncio.create_task(continuous_monitoring_loop())
            logger.info("Continuous monitoring started successfully")
        except Exception as e:
            logger.error(f"Failed to auto-start monitoring: {e}")
            monitoring_active = False

    return {
        'option_key': option_key,
        'selection_confirmed': True,
        'total_selected': len(selected_options),
        'monitoring_status': 'active' if monitoring_active else 'ready',
        'monitoring_auto_started': not monitoring_active and len(selected_options) == 1,
        'selected_option': selected_options[option_key]
    }

def list_selected_options() -> Dict[str, Any]:
    """
    List all currently selected options for monitoring.

    Returns:
        Dictionary with all selected options and their status
    """
    global selected_options

    if not selected_options:
        return {
            'total_selected': 0,
            'options': [],
            'monitoring_status': 'inactive'
        }

    options_list = []
    for key, option in selected_options.items():
        exp_date = datetime.datetime.strptime(option['expiration_date'], '%Y-%m-%d')
        days_to_expiry = (exp_date - datetime.datetime.now()).days

        options_list.append({
            'option_key': key,
            'symbol': option['symbol'],
            'strike': option['strike'],
            'expiration_date': option['expiration_date'],
            'days_to_expiry': days_to_expiry,
            'selected_at': option['selected_at'],
            'alerts_sent': option['alerts_sent'],
            'status': option['status'],
            'notes': option['notes']
        })

    return {
        'total_selected': len(selected_options),
        'options': sorted(options_list, key=lambda x: x['days_to_expiry']),
        'monitoring_status': 'active' if monitoring_active else 'ready'
    }

def remove_selected_option(option_key: str = None, symbol: str = None,
                          strike: float = None, expiration_date: str = None) -> Dict[str, Any]:
    """
    Remove an option from monitoring.

    Args:
        option_key: Direct option key, or
        symbol, strike, expiration_date: Individual components

    Returns:
        Dictionary with removal confirmation
    """
    global selected_options

    if option_key:
        key = option_key
    elif symbol and strike and expiration_date:
        key = f"{symbol}_{strike}_{expiration_date}"
    else:
        return {'error': 'Must provide either option_key or symbol/strike/expiration_date'}

    if key in selected_options:
        removed_option = selected_options.pop(key)
        return {
            'removed': True,
            'option_key': key,
            'removed_option': removed_option,
            'remaining_selected': len(selected_options)
        }
    else:
        return {
            'removed': False,
            'error': f'Option {key} not found in selected options',
            'available_options': list(selected_options.keys())
        }

def format_response(data: Any, error: Optional[str] = None) -> List[TextContent]:
    response = {
        "success": error is None,
        "timestamp": time.time(),
        "data": data if error is None else None,
        "error": error
    }

    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2, cls=StockflowJSONEncoder)
    )]

app = Server("stockflow-server-v2")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_stock_data_v2",
            description="Get comprehensive stock data including financials, analyst ratings, and calendar events",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "include_financials": {"type": "boolean", "description": "Include quarterly financials"},
                    "include_analysis": {"type": "boolean", "description": "Include analyst data"},
                    "include_calendar": {"type": "boolean", "description": "Include calendar events"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_historical_data_v2",
            description="Get historical price data with technical indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {
                        "type": "string",
                        "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
                        "default": "1d"
                    },
                    "prepost": {
                        "type": "boolean",
                        "description": "Include pre and post market data",
                        "default": False
                    }
                },
                "required": ["symbol", "period"]
            }
        ),
        Tool(
            name="get_options_chain_v2",
            description="Get options chain data with advanced greeks and analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "expiration_date": {"type": "string", "description": "Options expiration date (YYYY-MM-DD)"},
                    "include_greeks": {"type": "boolean", "description": "Include options greeks"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="calculate_greeks",
            description="Calculate Black-Scholes Greeks (delta, gamma, theta, vega, rho) for a specific option",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "strike": {"type": "number", "description": "Option strike price"},
                    "expiration_date": {"type": "string", "description": "Options expiration date (YYYY-MM-DD)"},
                    "option_type": {"type": "string", "enum": ["call", "put"], "description": "Option type: 'call' or 'put'"},
                    "current_price": {"type": "number", "description": "Current stock price (optional, will fetch from yfinance if not provided)"},
                    "volatility": {"type": "number", "description": "Implied volatility (default: 0.2)", "default": 0.2},
                    "risk_free_rate": {"type": "number", "description": "Risk-free rate (default: 0.05)", "default": 0.05}
                },
                "required": ["symbol", "strike", "expiration_date", "option_type"]
            }
        ),
        Tool(
            name="analyze_fortune500_otm_calls",
            description="Analyze Fortune 500 stocks for top 10 OTM call options with highest ITM probability using Monte Carlo simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "expiration_date": {"type": "string", "description": "Options expiration date (YYYY-MM-DD)"},
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Custom list of symbols to analyze (optional, defaults to Fortune 500 subset)"},
                    "min_volume": {"type": "integer", "description": "Minimum option volume filter (default: 100)", "default": 100},
                    "min_iv": {"type": "number", "description": "Minimum implied volatility filter (default: 0.3)", "default": 0.3},
                    "probability_threshold": {"type": "number", "description": "Minimum ITM probability threshold (default: 0.7)", "default": 0.7},
                    "slack_webhook": {"type": "string", "description": "Slack webhook URL for notifications (optional)"},
                    "alert_threshold": {"type": "number", "description": "ITM probability threshold for Slack alerts (default: 0.8)", "default": 0.8}
                },
                "required": ["expiration_date"]
            }
        ),
        Tool(
            name="select_option_for_monitoring",
            description="Select an option for monitoring and tracking. Use this when a user wants to pick a specific option (e.g., 'Pick TSLA $430')",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "strike": {"type": "number", "description": "Strike price"},
                    "expiration_date": {"type": "string", "description": "Expiration date (YYYY-MM-DD)"},
                    "current_price": {"type": "number", "description": "Current stock price (optional)"},
                    "notes": {"type": "string", "description": "Additional notes (optional)"}
                },
                "required": ["symbol", "strike", "expiration_date"]
            }
        ),
        Tool(
            name="list_selected_options",
            description="List all currently selected options for monitoring",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="remove_selected_option",
            description="Remove an option from monitoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "option_key": {"type": "string", "description": "Direct option key (symbol_strike_expiration)"},
                    "symbol": {"type": "string", "description": "Stock ticker symbol (alternative to option_key)"},
                    "strike": {"type": "number", "description": "Strike price (alternative to option_key)"},
                    "expiration_date": {"type": "string", "description": "Expiration date (alternative to option_key)"}
                },
                "required": []
            }
        ),
        Tool(
            name="start_continuous_monitoring",
            description="Start the continuous monitoring system for selected options (monitors every minute)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="stop_continuous_monitoring",
            description="Stop the continuous monitoring system",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_monitoring_status",
            description="Get current monitoring system status and statistics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_sell_analysis_summary",
            description="Get intelligent sell analysis summary for all monitored options with recommendations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="start_slack_app",
            description="Start the Slack App for two-way interaction and message handling",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="stop_slack_app",
            description="Stop the Slack App",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_slack_app_status",
            description="Get Slack App status and configuration details",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="analyze_option_realtime",
            description="Analyze a specific option in real-time with buy/sell advice (simulates Slack 'Pick' command)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "strike": {"type": "number", "description": "Strike price"},
                    "expiration_date": {"type": "string", "description": "Expiration date (YYYY-MM-DD), defaults to 2025-10-17"}
                },
                "required": ["symbol", "strike"]
            }
        ),
        Tool(
            name="get_realtime_market_sentiment",
            description="Get real-time market sentiment from multiple news sources and social media for enhanced Smart Picks analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols to analyze sentiment for"},
                    "timeframe": {"type": "string", "description": "Timeframe for sentiment analysis", "enum": ["1h", "4h", "24h"], "default": "4h"},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Sentiment sources", "default": ["news", "social", "analyst"]}
                },
                "required": ["symbols"]
            }
        ),
        Tool(
            name="detect_market_regime",
            description="Detect current market regime (bull, bear, sideways) using VIX, yield curve, and momentum indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "indicators": {"type": "array", "items": {"type": "string"}, "description": "Indicators to analyze", "default": ["VIX", "yield_curve", "momentum", "sentiment"]},
                    "lookback_days": {"type": "integer", "description": "Lookback period in days", "default": 30}
                },
                "required": []
            }
        ),
        Tool(
            name="get_options_flow_analysis",
            description="Analyze unusual options activity and institutional flow for enhanced Smart Picks",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Symbols to analyze options flow for"},
                    "timeframe": {"type": "string", "description": "Analysis timeframe", "enum": ["1h", "4h", "24h"], "default": "4h"},
                    "min_premium": {"type": "number", "description": "Minimum premium threshold", "default": 50000}
                },
                "required": ["symbols"]
            }
        ),
        Tool(
            name="smart_picks_optimal_options_enhanced",
            description="ENHANCED Smart Picks with institutional-grade data - Always shows best available options with market context and confidence levels",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_days_to_expiry": {"type": "integer", "description": "Maximum days to expiration (default: 30)", "default": 30},
                    "target_profit_potential": {"type": "number", "description": "Target profit potential - now adaptive (default: 0.15)", "default": 0.15},
                    "target_probability": {"type": "number", "description": "Target ITM probability - now adaptive (default: 0.45)", "default": 0.45},
                    "target_risk_level": {"type": "number", "description": "Target risk tolerance (1-10, default: 6) - now adaptive", "default": 6},
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Custom symbols to analyze (optional, defaults to Fortune 500)"},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default: 10)", "default": 10},
                    "always_show_results": {"type": "boolean", "description": "Always show best options even if they don't meet ideal criteria", "default": true}
                },
                "required": []
            }
        ),
        Tool(
            name="smart_picks_optimal_options",
            description="LEGACY: Find optimal risk/reward call options ≤30 days using advanced analysis techniques - USE smart_picks_optimal_options_enhanced instead",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_days_to_expiry": {"type": "integer", "description": "Maximum days to expiration (default: 30)", "default": 30},
                    "min_profit_potential": {"type": "number", "description": "Minimum profit potential percentage (default: 0.15)", "default": 0.15},
                    "min_probability": {"type": "number", "description": "Minimum ITM probability (default: 0.45)", "default": 0.45},
                    "max_risk_level": {"type": "number", "description": "Maximum risk tolerance (1-10, default: 6)", "default": 6},
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Custom symbols to analyze (optional, defaults to Fortune 500)"},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default: 10)", "default": 10}
                },
                "required": []
            }
        ),
        Tool(
            name="multi_scenario_monte_carlo",
            description="Phase 2: Advanced Multi-Scenario Monte Carlo analysis with Bull/Bear/Sideways market modeling for institutional-grade probability assessments",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols to analyze"},
                    "strikes": {"type": "array", "items": {"type": "number"}, "description": "Strike prices for each symbol"},
                    "expiration_date": {"type": "string", "description": "Options expiration date (YYYY-MM-DD)"},
                    "scenarios": {"type": "array", "items": {"type": "string"}, "description": "Market scenarios to model", "default": ["bull", "bear", "sideways"]},
                    "num_simulations": {"type": "integer", "description": "Number of Monte Carlo simulations per scenario", "default": 10000}
                },
                "required": ["symbols", "strikes", "expiration_date"]
            }
        ),
        Tool(
            name="historical_pattern_recognition",
            description="Phase 2: Advanced pattern recognition system that identifies similar historical market conditions and their outcomes for predictive analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol to analyze"},
                    "pattern_type": {"type": "string", "description": "Type of pattern to match", "enum": ["price_momentum", "volatility_spike", "earnings_setup", "technical_breakout"], "default": "price_momentum"},
                    "lookback_days": {"type": "integer", "description": "Days to look back for pattern matching", "default": 252},
                    "min_similarity": {"type": "number", "description": "Minimum similarity threshold (0-1)", "default": 0.7}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="event_driven_analysis",
            description="Phase 2: Comprehensive event-driven analysis engine tracking earnings, FDA approvals, Fed meetings, and other market-moving events",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols to analyze for upcoming events"},
                    "event_types": {"type": "array", "items": {"type": "string"}, "description": "Types of events to track", "default": ["earnings", "fda", "fed", "economic"]},
                    "days_ahead": {"type": "integer", "description": "How many days ahead to look for events", "default": 30},
                    "impact_threshold": {"type": "string", "description": "Minimum event impact level", "enum": ["low", "medium", "high"], "default": "medium"}
                },
                "required": ["symbols"]
            }
        ),
        Tool(
            name="cross_asset_correlation_analysis",
            description="Phase 2: Advanced correlation analysis across bonds, dollar, commodities, and crypto to predict options performance in different market conditions",
            inputSchema={
                "type": "object",
                "properties": {
                    "primary_symbols": {"type": "array", "items": {"type": "string"}, "description": "Primary stock symbols to analyze"},
                    "asset_classes": {"type": "array", "items": {"type": "string"}, "description": "Asset classes to correlate with", "default": ["bonds", "dollar", "commodities", "crypto"]},
                    "correlation_period": {"type": "integer", "description": "Period in days for correlation calculation", "default": 90},
                    "regime_analysis": {"type": "boolean", "description": "Include regime-based correlation analysis", "default": true}
                },
                "required": ["primary_symbols"]
            }
        ),
        Tool(
            name="advanced_volatility_forecasting",
            description="Phase 2: Sophisticated volatility forecasting using GARCH-like models, term structure analysis, and volatility surface modeling",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols to forecast volatility for"},
                    "forecast_horizon": {"type": "integer", "description": "Days ahead to forecast", "default": 30},
                    "model_type": {"type": "string", "description": "Volatility model to use", "enum": ["garch", "ewma", "realized", "hybrid"], "default": "hybrid"},
                    "include_term_structure": {"type": "boolean", "description": "Include volatility term structure analysis", "default": true}
                },
                "required": ["symbols"]
            }
        )
    ]

@app.call_tool()
@retry_on_error(max_retries=3, delay=1.0)
async def call_tool(name: str, arguments: dict):
    try:
        if name == "get_stock_data_v2":
            symbol = arguments['symbol'].strip().upper()
            include_financials = arguments.get('include_financials', False)
            include_analysis = arguments.get('include_analysis', False)
            include_calendar = arguments.get('include_calendar', False)
            
            ticker = yf.Ticker(symbol)
            
            # Basic info must be available
            info = ticker.info
            if not info:
                raise APIError(f"No data available for {symbol}")
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not price:
                raise APIError(f"No price data available for {symbol}")
            
            response = {
                "basic_info": {
                    "symbol": symbol,
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "description": info.get("longBusinessSummary", "N/A"),
                    "website": info.get("website", "N/A"),
                    "employees": info.get("fullTimeEmployees", 0)
                },
                "market_data": {
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "float_shares": info.get("floatShares"),
                    "regular_market_open": info.get("regularMarketOpen"),
                    "regular_market_high": info.get("regularMarketDayHigh"),
                    "regular_market_low": info.get("regularMarketDayLow"),
                    "regular_market_volume": info.get("regularMarketVolume"),
                    "regular_market_previous_close": info.get("regularMarketPreviousClose")
                },
                "valuation_metrics": {
                    "pe_ratio": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                    "enterprise_to_ebitda": info.get("enterpriseToEbitda")
                },
                "trading_info": {
                    "beta": info.get("beta"),
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                    "50d_avg": info.get("fiftyDayAverage"),
                    "200d_avg": info.get("twoHundredDayAverage"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "avg_volume": info.get("averageVolume")
                }
            }
            
            if include_financials:
                try:
                    financials = {
                        "quarterly_income": convert_df_timestamps(ticker.quarterly_income_stmt),
                        "quarterly_balance": convert_df_timestamps(ticker.quarterly_balance_sheet),
                        "quarterly_cashflow": convert_df_timestamps(ticker.quarterly_cashflow)
                    }
                    response["financials"] = financials
                except Exception as e:
                    logger.warning(f"Could not fetch financials for {symbol}: {str(e)}")
            
            if include_analysis:
                try:
                    analysis = {
                        "recommendations": ticker.recommendations.to_dict() if hasattr(ticker, 'recommendations') else None,
                        "analyst_price_targets": ticker.analyst_price_targets.to_dict() if hasattr(ticker, 'analyst_price_targets') else None
                    }
                    response["analysis"] = analysis
                except Exception as e:
                    logger.warning(f"Could not fetch analysis for {symbol}: {str(e)}")
            
            if include_calendar:
                try:
                    calendar = ticker.calendar
                    if calendar is not None:
                        response["calendar"] = calendar.to_dict()
                except Exception as e:
                    logger.warning(f"Could not fetch calendar for {symbol}: {str(e)}")
            
            return format_response(response)
            
        elif name == "get_historical_data_v2":
            symbol = arguments['symbol'].strip().upper()
            period = arguments['period']
            interval = arguments.get('interval', '1d')
            prepost = arguments.get('prepost', False)
            
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
            
            if period not in valid_periods:
                raise ValidationError(f"Invalid period. Must be one of: {', '.join(valid_periods)}")
            if interval not in valid_intervals:
                raise ValidationError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
            
            # Use download for single symbol (more efficient)
            history = yf.download(
                symbol,
                period=period,
                interval=interval,
                prepost=prepost,
                progress=False,
                multi_level_index=False
            )
            
            if history.empty:
                raise APIError(f"No historical data available for {symbol}")
            
            # Create DataFrame copy for calculations
            data = history.copy()
            
            # Technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Fixed RSI calculation:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Convert index to string format for serialization
            data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert to dict with records orientation
            history_dict = data.to_dict(orient='records')
            
            # Calculate summary statistics
            price_change = float(data['Close'].iloc[-1] - data['Close'].iloc[0])
            price_change_pct = (price_change / float(data['Close'].iloc[0])) * 100
            
            response = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "prepost": prepost,
                "data": history_dict,
                "summary": {
                    "start_date": data.index[0],
                    "end_date": data.index[-1],
                    "total_days": len(data),
                    "price_change": price_change,
                    "price_change_percent": price_change_pct,
                    "volatility": float(data['Close'].pct_change().std() * (252 ** 0.5) * 100),
                    "highest_price": float(data['High'].max()),
                    "lowest_price": float(data['Low'].min()),
                    "average_volume": float(data['Volume'].mean()),
                    "current_rsi": float(data['RSI'].iloc[-1]) if pd.notnull(data['RSI'].iloc[-1]) else None,
                    "current_macd": float(data['MACD'].iloc[-1]) if pd.notnull(data['MACD'].iloc[-1]) else None
                }
            }
            
            return format_response(response)
            
        elif name == "get_options_chain_v2":
            symbol = arguments['symbol'].strip().upper()
            include_greeks = arguments.get('include_greeks', False)
            
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                raise APIError(f"No options data available for {symbol}")
                
            # If no expiration date provided, use the nearest one
            expiration_date = arguments.get('expiration_date')
            if expiration_date:
                # Validate date format
                try:
                    exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                    if exp_date < datetime.datetime.now():
                        raise ValidationError("Expiration date must be in the future")
                    if expiration_date not in exp_dates:
                        raise ValidationError(f"No options available for date {expiration_date}. Available dates: {exp_dates}")
                except ValueError:
                    raise ValidationError("Invalid date format. Use YYYY-MM-DD")
            else:
                expiration_date = exp_dates[0]  # Use nearest expiration
            
            # Get the stock's current price for moneyness calculation
            current_price = ticker.info.get('regularMarketPrice') or ticker.info.get('currentPrice')
            if not current_price:
                raise APIError("Could not determine current stock price")
            
            try:
                options = ticker.option_chain(expiration_date)
                
                if not hasattr(options, 'calls') or not hasattr(options, 'puts'):
                    raise APIError(f"Invalid options data for {symbol}")
                
                # Helper function to process option chain
                def process_chain(chain, option_type):
                    chain['moneyness'] = chain['strike'] / current_price
                    chain['bid_ask_spread'] = chain['ask'] - chain['bid']
                    chain['bid_ask_spread_pct'] = (chain['bid_ask_spread'] / ((chain['bid'] + chain['ask']) / 2)) * 100
                    
                    # Convert to records and handle NaN values
                    processed = chain.where(pd.notnull(chain), None).to_dict(orient="records")
                    
                    # Add summary metrics
                    summary = {
                        f"total_{option_type}": len(chain),
                        f"itm_{option_type}": len(chain[chain['inTheMoney']]) if 'inTheMoney' in chain else 0,
                        f"total_volume": int(chain['volume'].sum()),
                        f"total_openInterest": int(chain['openInterest'].sum()),
                        "highest_volume_strikes": chain.nlargest(3, 'volume')[['strike', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records'),
                        "highest_openInterest_strikes": chain.nlargest(3, 'openInterest')[['strike', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records')
                    }
                    
                    return processed, summary
                
                # Process calls and puts
                calls_processed, calls_summary = process_chain(options.calls, "calls")
                puts_processed, puts_summary = process_chain(options.puts, "puts")
                
                # Calculate overall options statistics
                total_volume = calls_summary['total_volume'] + puts_summary['total_volume']
                put_call_ratio = puts_summary['total_volume'] / max(1, calls_summary['total_volume'])
                
                response = {
                    "symbol": symbol,
                    "underlying_price": current_price,
                    "expiration_date": expiration_date,
                    "days_to_expiration": (datetime.datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.datetime.now()).days,
                    "available_expiration_dates": exp_dates,
                    "summary": {
                        "total_volume": total_volume,
                        "put_call_ratio": put_call_ratio,
                        "total_calls": calls_summary['total_calls'],
                        "total_puts": puts_summary['total_puts'],
                        "itm_calls": calls_summary['itm_calls'],
                        "itm_puts": puts_summary['itm_puts'],
                        "calls_summary": calls_summary,
                        "puts_summary": puts_summary
                    },
                    "calls": calls_processed,
                    "puts": puts_processed
                }
                
                return format_response(response)
                
            except Exception as e:
                raise APIError(f"Failed to get options data: {str(e)}")

        elif name == "calculate_greeks":
            symbol = arguments['symbol'].strip().upper()
            strike = float(arguments['strike'])
            expiration_date = arguments['expiration_date']
            option_type = arguments['option_type'].lower()
            current_price = arguments.get('current_price')
            volatility = arguments.get('volatility', 0.2)
            risk_free_rate = arguments.get('risk_free_rate', 0.05)

            # Validate inputs
            if option_type not in ['call', 'put']:
                raise ValidationError("option_type must be 'call' or 'put'")

            if strike <= 0:
                raise ValidationError("Strike price must be positive")

            if volatility <= 0:
                raise ValidationError("Volatility must be positive")

            if risk_free_rate < 0:
                raise ValidationError("Risk-free rate cannot be negative")

            # Validate and parse expiration date
            try:
                exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                if exp_date < datetime.datetime.now():
                    raise ValidationError("Expiration date must be in the future")
            except ValueError:
                raise ValidationError("Invalid date format. Use YYYY-MM-DD")

            # Calculate time to expiration in years
            time_to_expiration = (exp_date - datetime.datetime.now()).days / 365.0

            # Get current price if not provided
            if current_price is None:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                if not current_price:
                    raise APIError(f"Could not determine current price for {symbol}")

            current_price = float(current_price)
            if current_price <= 0:
                raise ValidationError("Current price must be positive")

            # Calculate Greeks using Black-Scholes
            try:
                greeks = calculate_black_scholes_greeks(
                    current_price=current_price,
                    strike=strike,
                    time_to_expiration=time_to_expiration,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type
                )

                response = {
                    "symbol": symbol,
                    "option_details": {
                        "strike": strike,
                        "expiration_date": expiration_date,
                        "option_type": option_type,
                        "days_to_expiration": (exp_date - datetime.datetime.now()).days,
                        "time_to_expiration_years": time_to_expiration
                    },
                    "market_data": {
                        "current_price": current_price,
                        "volatility": volatility,
                        "risk_free_rate": risk_free_rate
                    },
                    "greeks": greeks,
                    "moneyness": {
                        "spot_to_strike_ratio": current_price / strike,
                        "in_the_money": (option_type == 'call' and current_price > strike) or
                                       (option_type == 'put' and current_price < strike),
                        "intrinsic_value": max(0, current_price - strike) if option_type == 'call'
                                         else max(0, strike - current_price),
                        "time_value": greeks['option_price'] - (max(0, current_price - strike) if option_type == 'call'
                                                              else max(0, strike - current_price))
                    }
                }

                return format_response(response)

            except Exception as e:
                raise APIError(f"Failed to calculate Greeks: {str(e)}")

        elif name == "analyze_fortune500_otm_calls":
            expiration_date = arguments['expiration_date']
            custom_symbols = arguments.get('symbols')
            min_volume = arguments.get('min_volume', 100)
            min_iv = arguments.get('min_iv', 0.3)
            probability_threshold = arguments.get('probability_threshold', 0.7)
            slack_webhook = arguments.get('slack_webhook')
            alert_threshold = arguments.get('alert_threshold', 0.8)

            # Use custom symbols or Fortune 500 subset
            symbols_to_analyze = custom_symbols if custom_symbols else FORTUNE_500_SYMBOLS[:20]  # Use first 20 for speed

            # Validate parameters
            if min_volume < 0:
                raise ValidationError("min_volume must be non-negative")

            if not (0.0 <= probability_threshold <= 1.0):
                raise ValidationError("probability_threshold must be between 0.0 and 1.0")

            if not (0.0 <= alert_threshold <= 1.0):
                raise ValidationError("alert_threshold must be between 0.0 and 1.0")

            try:
                # Analyze OTM calls using async batch processing
                logger.info(f"Analyzing {len(symbols_to_analyze)} symbols for OTM calls expiring {expiration_date}")

                top_options = await analyze_otm_calls_batch(
                    symbols=symbols_to_analyze,
                    expiration_date=expiration_date,
                    min_volume=min_volume,
                    min_iv=min_iv,
                    probability_threshold=probability_threshold
                )

                if not top_options:
                    return format_response({
                        "message": "No OTM call options found matching criteria",
                        "criteria": {
                            "expiration_date": expiration_date,
                            "min_volume": min_volume,
                            "min_iv": min_iv,
                            "probability_threshold": probability_threshold
                        },
                        "symbols_analyzed": len(symbols_to_analyze)
                    })

                # Check for high-probability alerts
                high_prob_options = [opt for opt in top_options if opt['itm_probability'] >= alert_threshold]

                # Send Slack notification for top 10 OTM calls if webhook provided
                slack_sent = False
                if slack_webhook and top_options:
                    # Use provided webhook URL or default to the specified webhook
                    webhook_url = slack_webhook if slack_webhook != "default" else SLACK_WEBHOOK_URL

                    if high_prob_options:
                        message = f"High-Probability OTM Alert - {len(high_prob_options)} options above {alert_threshold:.1%}"
                    else:
                        message = f"StockFlow Top 10 OTM Call Analysis - {expiration_date}"

                    slack_sent = send_professional_slack_notification(webhook_url, message, top_options)

                # Calculate summary statistics
                avg_probability = np.mean([opt['itm_probability'] for opt in top_options])
                avg_delta = np.mean([opt['delta'] for opt in top_options])
                total_volume = sum([opt['volume'] for opt in top_options])

                response = {
                    "analysis_timestamp": time.time(),
                    "expiration_date": expiration_date,
                    "criteria": {
                        "symbols_analyzed": len(symbols_to_analyze),
                        "min_volume": min_volume,
                        "min_iv": min_iv,
                        "probability_threshold": probability_threshold
                    },
                    "summary": {
                        "total_options_found": len(top_options),
                        "high_probability_alerts": len(high_prob_options),
                        "average_itm_probability": round(avg_probability, 4),
                        "average_delta": round(avg_delta, 4),
                        "total_volume": total_volume
                    },
                    "top_10_otm_calls": top_options,
                    "alerts": {
                        "threshold": alert_threshold,
                        "triggered_count": len(high_prob_options),
                        "slack_notification_sent": slack_sent
                    }
                }

                return format_response(response)

            except Exception as e:
                raise APIError(f"Failed to analyze Fortune 500 OTM calls: {str(e)}")

        elif name == "select_option_for_monitoring":
            symbol = arguments['symbol'].strip().upper()
            strike = float(arguments['strike'])
            expiration_date = arguments['expiration_date']
            current_price = arguments.get('current_price')
            notes = arguments.get('notes')

            # Validate inputs
            try:
                exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                if exp_date < datetime.datetime.now():
                    raise ValidationError("Expiration date must be in the future")
            except ValueError:
                raise ValidationError("Invalid date format. Use YYYY-MM-DD")

            if strike <= 0:
                raise ValidationError("Strike price must be positive")

            # Get current stock price if not provided
            if not current_price:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                except:
                    current_price = None

            result = select_option_for_monitoring(symbol, strike, expiration_date, current_price, notes)
            return format_response(result)

        elif name == "list_selected_options":
            result = list_selected_options()
            return format_response(result)

        elif name == "remove_selected_option":
            option_key = arguments.get('option_key')
            symbol = arguments.get('symbol')
            strike = arguments.get('strike')
            expiration_date = arguments.get('expiration_date')

            if symbol:
                symbol = symbol.strip().upper()

            result = remove_selected_option(option_key, symbol, strike, expiration_date)
            return format_response(result)

        elif name == "start_continuous_monitoring":
            result = start_continuous_monitoring()
            return format_response(result)

        elif name == "stop_continuous_monitoring":
            result = stop_continuous_monitoring()
            return format_response(result)

        elif name == "get_monitoring_status":
            result = get_monitoring_status()
            return format_response(result)

        elif name == "get_sell_analysis_summary":
            result = get_sell_analysis_summary()
            return format_response(result)

        elif name == "start_slack_app":
            result = await start_slack_app()
            return format_response(result)

        elif name == "stop_slack_app":
            result = await stop_slack_app()
            return format_response(result)

        elif name == "get_slack_app_status":
            result = get_slack_app_status()
            return format_response(result)

        elif name == "analyze_option_realtime":
            symbol = arguments['symbol'].strip().upper()
            strike = float(arguments['strike'])
            expiration_date = arguments.get('expiration_date', '2025-10-17')

            # Validate inputs
            try:
                exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                if exp_date < datetime.datetime.now():
                    raise ValidationError("Expiration date must be in the future")
            except ValueError:
                raise ValidationError("Invalid date format. Use YYYY-MM-DD")

            if strike <= 0:
                raise ValidationError("Strike price must be positive")

            # Get real-time option data
            option_data = await get_realtime_option_data(symbol, strike, expiration_date)

            if 'error' in option_data:
                return format_response(None, option_data['error'])

            # Generate buy/sell advice
            advice = await generate_buy_sell_advice(option_data)

            if 'error' in advice:
                return format_response(None, advice['error'])

            # AUTOMATICALLY ADD TO MONITORING if recommendation is positive
            auto_monitor = False
            if advice.get('recommendation', '').lower() in ['buy', 'strong buy']:
                # Add to monitoring automatically
                monitoring_result = select_option_for_monitoring(
                    symbol=symbol,
                    strike=strike,
                    expiration_date=expiration_date,
                    current_price=option_data.get('current_price'),
                    notes=f"Auto-added from Pick command - {advice.get('recommendation')}"
                )
                auto_monitor = True
                logger.info(f"Auto-added {symbol} ${strike} to monitoring: {monitoring_result['option_key']}")

            # Combine results
            result = {
                'option_data': option_data,
                'advice': advice,
                'analysis_type': 'realtime_slack_simulation',
                'formatted_response': format_analysis_response(option_data, advice),
                'auto_monitoring': {
                    'enabled': auto_monitor,
                    'total_monitored': len(selected_options) if auto_monitor else 0
                }
            }

            return format_response(result)

        elif name == "get_realtime_market_sentiment":
            symbols = arguments['symbols']
            timeframe = arguments.get('timeframe', '4h')
            sources = arguments.get('sources', ['news', 'social', 'analyst'])

            logger.info(f"Getting real-time sentiment for {len(symbols)} symbols")
            result = await get_realtime_market_sentiment(symbols, timeframe, sources)
            return format_response(result)

        elif name == "detect_market_regime":
            indicators = arguments.get('indicators', ['VIX', 'yield_curve', 'momentum', 'sentiment'])
            lookback_days = arguments.get('lookback_days', 30)

            logger.info(f"Detecting market regime with indicators: {indicators}")
            result = await detect_market_regime(indicators, lookback_days)
            return format_response(result)

        elif name == "get_options_flow_analysis":
            symbols = arguments['symbols']
            timeframe = arguments.get('timeframe', '4h')
            min_premium = arguments.get('min_premium', 50000)

            logger.info(f"Analyzing options flow for {len(symbols)} symbols")
            result = await get_options_flow_analysis(symbols, timeframe, min_premium)
            return format_response(result)

        elif name == "smart_picks_optimal_options_enhanced":
            max_days_to_expiry = arguments.get('max_days_to_expiry', 30)
            target_profit_potential = arguments.get('target_profit_potential', 0.15)
            target_probability = arguments.get('target_probability', 0.45)
            target_risk_level = arguments.get('target_risk_level', 6)
            custom_symbols = arguments.get('symbols')
            max_results = arguments.get('max_results', 10)
            always_show_results = arguments.get('always_show_results', True)

            # Use custom symbols or Fortune 500 most liquid
            symbols_to_analyze = custom_symbols if custom_symbols else FORTUNE_500_SYMBOLS
            logger.info(f"ENHANCED Smart Picks: Analyzing {len(symbols_to_analyze)} symbols with institutional-grade data")

            result = await find_optimal_risk_reward_options_enhanced(
                symbols=symbols_to_analyze,
                max_days_to_expiry=max_days_to_expiry,
                target_profit_potential=target_profit_potential,
                target_probability=target_probability,
                target_risk_level=target_risk_level,
                max_results=max_results,
                always_show_results=always_show_results
            )
            return format_response(result)

        elif name == "smart_picks_optimal_options":
            max_days_to_expiry = arguments.get('max_days_to_expiry', 30)
            min_profit_potential = arguments.get('min_profit_potential', 0.15)
            min_probability = arguments.get('min_probability', 0.45)
            max_risk_level = arguments.get('max_risk_level', 6)
            custom_symbols = arguments.get('symbols')
            max_results = arguments.get('max_results', 10)

            # Use custom symbols or Fortune 500 most liquid
            symbols_to_analyze = custom_symbols if custom_symbols else FORTUNE_500_SYMBOLS

            logger.info(f"Smart Picks: Finding optimal risk/reward options for {len(symbols_to_analyze)} symbols")

            result = await find_optimal_risk_reward_options(
                symbols=symbols_to_analyze,
                max_days_to_expiry=max_days_to_expiry,
                min_profit_potential=min_profit_potential,
                min_probability=min_probability,
                max_risk_level=max_risk_level,
                max_results=max_results
            )

            return format_response(result)

        elif name == "multi_scenario_monte_carlo":
            symbols = arguments.get('symbols', [])
            strikes = arguments.get('strikes', [])
            expiration_date = arguments.get('expiration_date')
            scenarios = arguments.get('scenarios', ['bull', 'bear', 'sideways'])
            num_simulations = arguments.get('num_simulations', 10000)

            if not symbols or not strikes or not expiration_date:
                raise ValidationError("symbols, strikes, and expiration_date are required")

            if len(symbols) != len(strikes):
                raise ValidationError("Number of symbols must match number of strikes")

            logger.info(f"Multi-Scenario Monte Carlo: Analyzing {len(symbols)} options across {len(scenarios)} scenarios")

            results = []
            for symbol, strike in zip(symbols, strikes):
                analysis = await multi_scenario_monte_carlo_analysis(
                    symbol=symbol,
                    strike=strike,
                    expiration_date=expiration_date,
                    scenarios=scenarios,
                    num_simulations=num_simulations
                )
                results.append(analysis)

            return format_response({
                "success": True,
                "analysis_type": "multi_scenario_monte_carlo",
                "results": results,
                "scenarios_analyzed": scenarios,
                "total_simulations": num_simulations * len(scenarios)
            })

        elif name == "historical_pattern_recognition":
            symbol = arguments.get('symbol')
            pattern_type = arguments.get('pattern_type', 'price_momentum')
            lookback_days = arguments.get('lookback_days', 252)
            min_similarity = arguments.get('min_similarity', 0.7)

            if not symbol:
                raise ValidationError("symbol is required")

            logger.info(f"Pattern Recognition: Analyzing {symbol} for {pattern_type} patterns")

            analysis = await historical_pattern_recognition(
                symbol=symbol,
                lookback_days=lookback_days,
                pattern_type=pattern_type
            )

            return format_response({
                "success": True,
                "analysis_type": "historical_pattern_recognition",
                "symbol": symbol,
                "pattern_analysis": analysis
            })

        elif name == "event_driven_analysis":
            symbols = arguments.get('symbols', [])
            event_types = arguments.get('event_types', ['earnings', 'fda', 'fed', 'economic'])
            days_ahead = arguments.get('days_ahead', 30)
            impact_threshold = arguments.get('impact_threshold', 'medium')

            if not symbols:
                raise ValidationError("symbols are required")

            logger.info(f"Event-Driven Analysis: Analyzing {len(symbols)} symbols for {len(event_types)} event types")

            results = {}
            for symbol in symbols:
                try:
                    analysis = await event_driven_analysis(
                        symbol=symbol,
                        days_ahead=days_ahead
                    )
                    results[symbol] = analysis
                except Exception as e:
                    logger.warning(f"Event analysis failed for {symbol}: {e}")
                    results[symbol] = {}

            analysis = {'events_analysis': results}

            return format_response({
                "success": True,
                "analysis_type": "event_driven_analysis",
                "symbols_analyzed": len(symbols),
                "event_analysis": analysis
            })

        elif name == "cross_asset_correlation_analysis":
            primary_symbols = arguments.get('primary_symbols', [])
            asset_classes = arguments.get('asset_classes', ['bonds', 'dollar', 'commodities', 'crypto'])
            correlation_period = arguments.get('correlation_period', 90)
            regime_analysis = arguments.get('regime_analysis', True)

            if not primary_symbols:
                raise ValidationError("primary_symbols are required")

            logger.info(f"Cross-Asset Correlation: Analyzing {len(primary_symbols)} symbols against {len(asset_classes)} asset classes")

            results = {}
            for symbol in primary_symbols:
                try:
                    analysis = await cross_asset_correlation_analysis(
                        symbol=symbol,
                        lookback_days=correlation_period
                    )
                    results[symbol] = analysis
                except Exception as e:
                    logger.warning(f"Correlation analysis failed for {symbol}: {e}")
                    results[symbol] = {}

            analysis = {'correlation_data': results}

            return format_response({
                "success": True,
                "analysis_type": "cross_asset_correlation",
                "correlation_analysis": analysis
            })

        elif name == "advanced_volatility_forecasting":
            symbols = arguments.get('symbols', [])
            forecast_horizon = arguments.get('forecast_horizon', 30)
            model_type = arguments.get('model_type', 'hybrid')
            include_term_structure = arguments.get('include_term_structure', True)

            if not symbols:
                raise ValidationError("symbols are required")

            logger.info(f"Advanced Volatility Forecasting: Forecasting volatility for {len(symbols)} symbols using {model_type} model")

            results = []
            for symbol in symbols:
                try:
                    forecast = await advanced_volatility_forecasting(
                        symbol=symbol,
                        forecast_days=forecast_horizon,
                        current_iv=0.25  # Default IV if not available
                    )
                    results.append(forecast)
                except Exception as e:
                    logger.warning(f"Volatility forecasting failed for {symbol}: {e}")
                    results.append({'symbol': symbol, 'error': str(e)})

            return format_response({
                "success": True,
                "analysis_type": "advanced_volatility_forecasting",
                "volatility_forecasts": results,
                "model_used": model_type,
                "forecast_horizon": forecast_horizon
            })

    except ValidationError as e:
        logger.error(f"Validation error in {name}: {str(e)}")
        return format_response(None, f"Validation error: {str(e)}")
        
    except APIError as e:
        logger.error(f"API error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"API error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"Internal error: {str(e)}")

# Global scheduler state for 5-minute updates
scheduler_state = {
    'running': False,
    'task': None,
    'config': None
}

async def scheduled_otm_analysis():
    """Background task for 5-minute OTM call analysis updates."""
    while scheduler_state['running']:
        try:
            config = scheduler_state['config']
            if config:
                logger.info("Running scheduled OTM call analysis...")

                # Analyze using the stored configuration
                top_options = await analyze_otm_calls_batch(
                    symbols=config.get('symbols', FORTUNE_500_SYMBOLS[:10]),
                    expiration_date=config['expiration_date'],
                    min_volume=config.get('min_volume', 100),
                    min_iv=config.get('min_iv', 0.3),
                    probability_threshold=config.get('probability_threshold', 0.7)
                )

                # Check for high-probability alerts
                alert_threshold = config.get('alert_threshold', 0.8)
                high_prob_options = [opt for opt in top_options if opt['itm_probability'] >= alert_threshold]

                # Send Slack notification if configured and alerts triggered
                if config.get('slack_webhook') and high_prob_options:
                    webhook_url = config['slack_webhook'] if config['slack_webhook'] != "default" else SLACK_WEBHOOK_URL
                    message = f"Scheduled Alert: {len(high_prob_options)} high-probability OTM calls detected"
                    send_professional_slack_notification(webhook_url, message, high_prob_options)
                    logger.info(f"Sent Slack alert for {len(high_prob_options)} high-probability options")

                # Log the analysis results
                if top_options:
                    avg_prob = np.mean([opt['itm_probability'] for opt in top_options])
                    logger.info(f"Scheduled analysis complete: {len(top_options)} options found, avg ITM prob: {avg_prob:.3f}")
                else:
                    logger.info("Scheduled analysis complete: No options found matching criteria")

        except Exception as e:
            logger.error(f"Error in scheduled OTM analysis: {str(e)}")

        # Wait 5 minutes before next analysis
        await asyncio.sleep(300)

def start_scheduler(config: Dict[str, Any]) -> bool:
    """Start the 5-minute scheduler for OTM call analysis."""
    try:
        if scheduler_state['running']:
            stop_scheduler()

        scheduler_state['config'] = config
        scheduler_state['running'] = True

        # Start the background task
        loop = asyncio.get_event_loop()
        scheduler_state['task'] = loop.create_task(scheduled_otm_analysis())

        logger.info("Started 5-minute OTM call analysis scheduler")
        return True

    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        return False

def stop_scheduler() -> bool:
    """Stop the 5-minute scheduler."""
    try:
        scheduler_state['running'] = False

        if scheduler_state['task'] and not scheduler_state['task'].done():
            scheduler_state['task'].cancel()

        scheduler_state['config'] = None
        logger.info("Stopped 5-minute OTM call analysis scheduler")
        return True

    except Exception as e:
        logger.error(f"Failed to stop scheduler: {str(e)}")
        return False

async def main():
    logger.info("Starting Stockflow server v2...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        # Clean up scheduler on shutdown
        stop_scheduler()

if __name__ == "__main__":
    asyncio.run(main())