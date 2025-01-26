import yfinance as yf
import pandas as pd
import streamlit as st

def safe_yf_download(tickers, start, end, interval=None):
    """
    Safely download data from Yahoo Finance with enhanced error handling and data validation.
    
    Args:
        tickers: String or list of ticker symbols
        start: Start date for data download
        end: End date for data download
        interval: Optional interval parameter ('1d', '1mo', etc.)
        
    Returns:
        DataFrame or Series with adjusted close prices
    """
    try:
        # Prepare download parameters
        params = {
            'start': start,
            'end': end
        }
        if interval:
            params['interval'] = interval

        # Download the data
        data = yf.download(tickers, **params)
        
        # Check if data is empty
        if data.empty:
            st.error(f"No data received for tickers: {tickers}")
            return None
        
        # Normalize column structure
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.values]
        
        # Find the adjusted close column
        adj_close_cols = [col for col in data.columns if 'Adj Close' in col or 'Adj_Close' in col]
        
        if not adj_close_cols:
            st.error(f"No 'Adj Close' column found in data. Available columns: {list(data.columns)}")
            return None
        
        # Handle single or multiple tickers
        if isinstance(tickers, str) or len(tickers) == 1:
            # For single ticker, return full DataFrame
            return data
        else:
            # For multiple tickers, return Adjusted Close columns
            return data[adj_close_cols]
            
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None