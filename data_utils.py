import yfinance as yf
import pandas as pd
import streamlit as st

def safe_yf_download(tickers, start, end, interval=None):
    """
    Safely download data from Yahoo Finance with error handling and data validation.
    
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
            
        # Handle different data structures based on single vs multiple tickers
        if isinstance(tickers, str) or len(tickers) == 1:
            # Single ticker - data will be a simple DataFrame
            if 'Adj Close' not in data.columns:
                st.error("Expected 'Adj Close' column not found in data")
                return None
            # Return full DataFrame for single ticker case
            return data
        else:
            # Multiple tickers - data will have MultiIndex columns
            if ('Adj Close',) not in data.columns and 'Adj Close' not in data.columns:
                st.error("Expected 'Adj Close' column not found in data")
                return None
                
            # Handle both possible column structures
            try:
                return data['Adj Close']
            except:
                # If 'Adj Close' is part of MultiIndex
                adj_close_cols = [col for col in data.columns if 'Adj Close' in col]
                if adj_close_cols:
                    return data[adj_close_cols]
                return None
                
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None