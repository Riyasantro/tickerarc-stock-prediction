"""NSE stock universe used by TickerArc Day 1.

Keep the universe in one place so the data pipeline can be updated without
changing downloader code.
"""

NIFTY50_SYMBOLS = [
    "RELIANCE",
    "BHARTIARTL",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "TCS",
    "BAJFINANCE",
    "LT",
    "HINDUNILVR",
    "SUNPHARMA",
    "INFY",
    "TITAN",
    "ADANIPORTS",
    "KOTAKBANK",
    "ADANIENT",
    "AXISBANK",
    "MARUTI",
    "M&M",
    "HCLTECH",
    "ITC",
    "ULTRACEMCO",
    "BAJAJ-AUTO",
    "ETERNAL",
    "NTPC",
    "JSWSTEEL",
    "BAJAJFINSV",
    "ONGC",
    "BEL",
    "NESTLEIND",
    "COALINDIA",
    "POWERGRID",
    "SHRIRAMFIN",
    "ASIANPAINT",
    "TATASTEEL",
    "HINDALCO",
    "GRASIM",
    "EICHERMOT",
    "INDIGO",
    "SBILIFE",
    "WIPRO",
    "JIOFIN",
    "TECHM",
    "TRENT",
    "APOLLOHOSP",
    "HDFCLIFE",
    "CIPLA",
    "TMPV",
    "MAXHEALTH",
    "DRREDDY",
    "TATACONSUM",
]


def yahoo_symbol(symbol: str) -> str:
    """Convert an NSE symbol to its Yahoo Finance symbol."""
    return f"{symbol}.NS"
