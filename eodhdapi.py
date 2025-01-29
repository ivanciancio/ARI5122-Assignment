import pandas as pd
import requests
from datetime import datetime, timedelta  # Added timedelta import
import streamlit as st

class EODHDClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"

    def download(self, symbol, start=None, end=None, interval='d'):
        """
        Download historical price data for a given symbol.
        
        Parameters:
        - symbol: Stock symbol (e.g., 'AAPL.US' for US stocks)
        - start: Start date (datetime or string 'YYYY-MM-DD')
        - end: End date (datetime or string 'YYYY-MM-DD')
        - interval: Data interval ('d' for daily, 'm' for monthly)
        
        Returns:
        - pandas DataFrame with OHLCV data
        """
        # Format dates
        if isinstance(start, datetime):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, datetime):
            end = end.strftime('%Y-%m-%d')
        
        # Handle special cases for indices
        if symbol == '^GSPC':
            symbol = 'SPY.US'  # Use SPY ETF as a proxy for S&P 500
        elif symbol == '^FTSE' or symbol == 'FTSE.INDX' or symbol == 'UKX.INDX':
            symbol = 'ISF.LSE'  # Use iShares FTSE 100 ETF as proxy for FTSE 100
        elif '.' not in symbol:
            symbol = f"{symbol}.US"  # Add .US extension for US stocks
            
        # Construct API URL
        url = f"{self.base_url}/eod/{symbol}"
        
        # Set up parameters
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'period': interval,
        }
        if start:
            params['from'] = start
        if end:
            params['to'] = end
            
        try:
            # Make API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Convert response to DataFrame
            df = pd.DataFrame(response.json())
            
            # Check if we got any data
            if df.empty:
                raise Exception(f"No data returned for symbol {symbol}")
            
            # Rename columns
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'adjusted_close': 'Adj Close',
                'volume': 'Volume'
            })
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching data from EODHD: {str(e)}")

def get_client():
    """
    Create an EODHD client instance with API key from Streamlit secrets.
    """
    try:
        # Try to get API key from Streamlit secrets
        api_key = st.secrets["EODHD_API_KEY"]
        
        # Test the API key with a simple request
        client = EODHDClient(api_key)
        try:
            # Test with a simple request for Apple stock
            test_data = client.download('AAPL.US', 
                                      start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                      end=datetime.now().strftime('%Y-%m-%d'))
            if test_data is not None and not test_data.empty:
                st.session_state['eodhd_configured'] = True
                return client
        except Exception as e:
            st.error(f"API Key found but test request failed: {str(e)}")
            st.error("Please check if your API key is valid and has access to the EOD Historical Data API.")
            raise ValueError("Invalid API key or API request failed")
            
    except KeyError:
        st.error("""
        EODHD API key not found. Please configure it.""")
        raise ValueError("EODHD_API_KEY not found in Streamlit secrets")