"""
Data Adapter Module
===================

Provides a unified interface for loading market data.
Supports both synthetic generation (for testing) and CSV loading (for production).
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class DataAdapter:
    def __init__(self, source_type: str = 'synthetic', csv_path: Optional[str] = None):
        """
        Args:
            source_type: 'synthetic' or 'csv'
            csv_path: Path to CSV file (required if source_type='csv')
                      CSV must have columns: [date, ticker, price] 
                      and optionally [market_cap, sector, earnings]
        """
        self.source_type = source_type
        self.csv_path = csv_path

    def get_data(self, start_date: str, end_date: str, n_stocks: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """
        Returns:
            prices: DataFrame (index=date, columns=tickers)
            metadata: Dict (e.g., sectors, caps, earnings)
        """
        if self.source_type == 'csv':
            return self._load_from_csv(start_date, end_date)
        else:
            return self._generate_synthetic(start_date, end_date, n_stocks)

    def _load_from_csv(self, start, end) -> Tuple[pd.DataFrame, Dict]:
        logger.info(f"Loading real data from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path, parse_dates=['date'])
            
            # Filter dates
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            
            # Pivot to wide format (Index=Date, Columns=Tickers)
            prices = df.pivot(index='date', columns='ticker', values='price')
            
            # Extract Metadata if available
            metadata = {}
            
            # 1. Market Caps
            if 'market_cap' in df.columns:
                metadata['market_caps'] = df.pivot(index='date', columns='ticker', values='market_cap')
            
            # 2. Sectors (Static assumption for simplicity)
            if 'sector' in df.columns:
                # Take the last known sector for each ticker
                sector_df = df.drop_duplicates(subset=['ticker'], keep='last')
                metadata['sectors'] = sector_df.set_index('ticker')['sector'].to_dict()
                
            # 3. Earnings
            if 'earnings' in df.columns:
                metadata['earnings'] = df.pivot(index='date', columns='ticker', values='earnings')
                
            return prices, metadata

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def _generate_synthetic(self, start, end, n_stocks) -> Tuple[pd.DataFrame, Dict]:
        logger.info("Generating synthetic data...")
        dates = pd.date_range(start, end, freq='B')
        tickers = [f"STK_{i:03d}" for i in range(n_stocks)]
        
        # Prices
        prices = pd.DataFrame(
            100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, (len(dates), n_stocks)), axis=0),
            index=dates, columns=tickers
        )
        
        # Metadata
        metadata = {
            'market_caps': prices * np.random.uniform(1e6, 1e9, n_stocks),
            'sectors': {t: np.random.choice(['Tech', 'Finance', 'Health']) for t in tickers},
            'earnings': prices * 0.05  # Mock earnings yield
        }
        
        return prices, metadata