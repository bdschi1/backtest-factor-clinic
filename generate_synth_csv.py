"""
Synthetic Data Generator
========================
Generates a realistic 'dummy' CSV file for testing backtest strategies.
Output: among_synth.csv (approx 20 years of daily data for 50 stocks)
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def generate_csv(filename='among_synth.csv', n_stocks=50, start_year=2004, end_year=2024):
    logger.info(f"Generating synthetic data from {start_year} to {end_year}...")
    
    # 1. Setup Dates
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    dates = pd.date_range(start_date, end_date, freq='B') # Business days
    
    tickers = [f"STK_{i:03d}" for i in range(n_stocks)]
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # 2. Assign Static Metadata (Sector)
    ticker_sectors = {t: np.random.choice(sectors) for t in tickers}
    
    # 3. Generate Price Walks (Geometric Brownian Motion)
    logger.info("Generating price paths...")
    price_data = []
    
    # Pre-generate random returns for speed
    # Mean daily ret ~0.03% (approx 7-8% annual), Vol ~2% daily
    returns = np.random.normal(0.0003, 0.02, (len(dates), n_stocks))
    
    # Calculate price paths
    prices = 100 * np.cumprod(1 + returns, axis=0)
    
    # 4. Generate Fundamental Data
    logger.info("Generating fundamentals (Market Cap & Earnings)...")
    
    # Initialize lists to build DataFrame
    all_dates = []
    all_tickers = []
    all_prices = []
    all_caps = []
    all_sectors = []
    all_earnings = []
    
    for i, ticker in enumerate(tickers):
        # Extract price series for this ticker
        p_series = prices[:, i]
        
        # Market Cap = Price * Random Shares (10M to 100M)
        shares = np.random.uniform(10_000_000, 100_000_000)
        caps = p_series * shares
        
        # Earnings (TTM)
        # P/E ratio fluctuates between 10x and 30x
        pe_ratios = np.random.uniform(10, 30, len(dates))
        earnings = p_series / pe_ratios
        
        # Add a "shock" to earnings occasionally to test robustness
        # (Randomly make some earnings negative)
        shock_mask = np.random.random(len(dates)) < 0.01 # 1% chance
        earnings[shock_mask] = earnings[shock_mask] * -1.0
        
        # Append to master lists
        all_dates.extend(dates)
        all_tickers.extend([ticker] * len(dates))
        all_prices.extend(p_series)
        all_caps.extend(caps)
        all_sectors.extend([ticker_sectors[ticker]] * len(dates))
        all_earnings.extend(earnings)
        
    # 5. Create DataFrame
    logger.info("Assembling DataFrame...")
    df = pd.DataFrame({
        'date': all_dates,
        'ticker': all_tickers,
        'price': all_prices,
        'market_cap': all_caps,
        'sector': all_sectors,
        'earnings': all_earnings
    })
    
    # 6. Formatting
    # Round prices to 2 decimals, caps to integers
    df['price'] = df['price'].round(2)
    df['market_cap'] = df['market_cap'].astype(int)
    df['earnings'] = df['earnings'].round(2)
    
    # Sort by date then ticker
    df = df.sort_values(['date', 'ticker'])
    
    # 7. Save
    logger.info(f"Saving to {filename}...")
    df.to_csv(filename, index=False)
    logger.info(f"✅ Done! File saved: {filename} ({len(df):,} rows)")

if __name__ == "__main__":
    generate_csv(filename="among_synth.csv")