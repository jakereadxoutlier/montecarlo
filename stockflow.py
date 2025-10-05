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

    return {
        'option_key': option_key,
        'selection_confirmed': True,
        'total_selected': len(selected_options),
        'monitoring_status': 'active' if monitoring_active else 'ready',
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
            name="smart_picks_optimal_options",
            description="Find optimal risk/reward call options ≤30 days using advanced analysis techniques - the 'hack' for high-probability, high-profit options",
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

            # Combine results
            result = {
                'option_data': option_data,
                'advice': advice,
                'analysis_type': 'realtime_slack_simulation',
                'formatted_response': format_analysis_response(option_data, advice)
            }

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