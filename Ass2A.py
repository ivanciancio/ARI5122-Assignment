import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from eodhdHelper import get_client

def main():
    # Title
    st.write("### Beta Calculation and Interpretation")

    # Defining the stock symbols to be used in the analysis
    symbols = {
        "S&P 500": "^GSPC",  # S&P500 stock
        "Amazon (AMZN)": "AMZN.US",  # Amazon stock
        "Boeing (BA)": "BA.US",  # Boeing stock
        "General Electric (GE)": "GE.US",  # General Electric stock
    }

    # Defining the time periods for analysis
    start_date_part1 = "2009-01-01"  # Start of the period for Part 1 analysis
    end_date_part1 = "2018-12-31"  # End of the period for Part 1 analysis
    start_date_part2 = "2014-01-01"  # Start of the period for Part 2 analysis
    end_date_part2 = "2018-11-30"  # End of the period for Part 2 analysis

    # Sidebar options to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section with better visibility
    sections = {
        "1. Data Collection": st.sidebar.checkbox(
            "📊 Data Collection and Returns",
            help="View data collection and daily returns calculation"
        ),
        "2. Beta Calculation": st.sidebar.checkbox(
            "📈 Beta Calculation",
            help="View beta calculations and interpretations"
        ),
        "3. Stock Simulation": st.sidebar.checkbox(
            "🔄 Apple Stock Simulation",
            help="View stock price simulation and risk assessment"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    # Initialise EODHD client
    client = get_client()

    # Function to fetch historical data using EODHD, cached for efficiency
    @st.cache_data
    def fetch_data(ticker, start, end):
        # Fetch historical stock data for a given ticker and date range
        df = client.download(ticker, start=start, end=end)
        return df

    # Function to calculate daily returns based on adjusted closing prices
    @st.cache_data
    def calculate_daily_returns(data):
        # Compute percentage changes in adjusted closing prices to get daily returns
        return data['Adj Close'].pct_change()

    # Function to calculate beta, p-value, and confidence intervals using regression
    @st.cache_data
    def calculate_beta(asset_returns, market_returns):
        # Perform linear regression to determine beta and related statistics
        X = sm.add_constant(market_returns)  # Add a constant to the regression model
        model = sm.OLS(asset_returns, X).fit()  # Fit the ordinary least squares model
        beta = model.params.iloc[1]  # Extract the beta coefficient (slope)
        p_value = round(model.pvalues.iloc[1], 3)  # Extract and round the p-value
        conf_interval = model.conf_int().iloc[1]  # Extract the 95% confidence interval
        return beta, p_value, conf_interval

    # Function to simulate stock prices using the Geometric Brownian Motion (GBM) model
    @st.cache_data
    def simulate_gbm(S0, mu, sigma, days, n_simulations):
        # Initialise array to store simulated prices
        dt = 1  # Time step (1 day)
        price_paths = np.zeros((days, n_simulations))  # Create a matrix for price paths
        price_paths[0] = S0  # Set initial price for all simulations

        # Generate price paths for each day
        for t in range(1, days):
            random_shock = np.random.normal(0, 1, n_simulations)  # Random shocks
            price_paths[t] = price_paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * random_shock * np.sqrt(dt)
            )
        return pd.DataFrame(price_paths)

    try:
        # Fetch and process all data once at the start if needed
        if any([sections["1. Data Collection"], sections["2. Beta Calculation"]]):
            if 'processed_data' not in st.session_state:
                st.session_state.processed_data = {}
                for name, ticker in symbols.items():
                    st.write(f"Fetching data for {name}...")
                    data = fetch_data(ticker, start_date_part1, end_date_part1)
                    data['Daily Returns'] = calculate_daily_returns(data)
                    st.session_state.processed_data[name] = data

        # Part 1A: Data Collection and Returns Calculation
        if sections["1. Data Collection"]:
            st.markdown("---")
            st.subheader("Part 1A: Data Collection and Returns Calculation")

            # Display a preview of the collected data
            for name, data in st.session_state.processed_data.items():
                st.subheader(f"{name} Data Preview")
                preview_data = data[['Adj Close', 'Daily Returns']].dropna().reset_index()
                preview_data['Date'] = preview_data['Date'].dt.strftime('%d-%m-%Y')
                preview_data.columns = ['Date', 'Adj Close', 'Daily Returns']
                st.dataframe(preview_data.head())

        # Part 1B: Beta Calculation
        if sections["2. Beta Calculation"]:
            st.markdown("---")
            st.subheader("Part 1B: Beta Calculation")

            # Get market returns (S&P 500)
            sp500_returns = st.session_state.processed_data["S&P 500"]['Daily Returns'].dropna()

            # Calculate beta for each stock relative to the S&P 500
            beta_results = []
            for name, data in st.session_state.processed_data.items():
                if name != "S&P 500":  # Skip the market benchmark
                    asset_returns = data['Daily Returns'].dropna()

                    # Align asset returns with market returns
                    common_index = sp500_returns.index.intersection(asset_returns.index)
                    aligned_sp500 = sp500_returns.loc[common_index]
                    aligned_asset = asset_returns.loc[common_index]

                    # Calculating beta and related statistics
                    beta, p_value, conf_interval = calculate_beta(aligned_asset, aligned_sp500)

                    # Store results in a list
                    beta_results.append({
                        "Asset": name,
                        "Beta": round(beta, 3),
                        "P-Value": round(p_value, 3),
                        "95% CI Lower": round(conf_interval[0], 3),
                        "95% CI Upper": round(conf_interval[1], 3)
                    })

            # Converting results into a DataFrame for better display
            beta_df = pd.DataFrame(beta_results)

            # Displaying beta results as a table
            st.dataframe(
                beta_df.style.format(
                    {"Beta": "{:.3f}", "P-Value": "{:.3f}", "95% CI Lower": "{:.3f}", "95% CI Upper": "{:.3f}"}
                ).set_table_styles(
                    [
                        {"selector": "th", "props": [("text-align", "center"), ("font-size", "14px")]},
                        {"selector": "td", "props": [("text-align", "center"), ("font-size", "12px")]},
                    ]
                ),
                use_container_width=True
            )

            # Interpretation of beta values
            st.subheader("Interpretation")
            st.write("""
            **Beta values and their implications**
            - **Beta > 1**: Indicates higher volatility compared to the market.
            - **Beta < 1**: Indicates lower volatility compared to the market.
            - **Beta ~ 1**: Indicates the asset moves in line with the market.
            
            **Examples:**
            - Amazon (AMZN): Beta of 1.103 suggests it is slightly more volatile than the market.
            - Boeing (BA): Beta of 1.119 indicates moderate sensitivity to market movements.
            - General Electric (GE): Beta of 1.171 indicates higher sensitivity to market trends.
            """)

        # Part 2: Apple Stock Simulation
        if sections["3. Stock Simulation"]:
            st.markdown("---")
            st.subheader("Part 2: Apple Stock Simulation and Risk Assessment")

            # Fetching Apple stock data for the specified time range
            st.write("Fetching Apple's data...")
            apple_data = fetch_data("AAPL.US", start_date_part2, end_date_part2)
            apple_data['Daily Returns'] = calculate_daily_returns(apple_data)
            daily_returns = apple_data['Daily Returns'].dropna()

            # Calculating parameters for the GBM model
            mean_return = daily_returns.mean()
            std_dev = daily_returns.std()
            last_price = apple_data['Adj Close'].iloc[-1]

            # Displaying calculated parameters
            st.write(f"**Mean Daily Return:** {mean_return:.5f}")
            st.write(f"**Volatility (Standard Deviation):** {std_dev:.5f}")
            st.write(f"**Last Observed Price:** ${last_price:.2f}")

            # Simulating future price paths
            st.write("Simulating future price paths using GBM...")
            days = 20
            n_simulations = 1000
            simulated_prices = simulate_gbm(last_price, mean_return, std_dev, days, n_simulations)

            # Plotting simulated price paths
            plt.figure(figsize=(10, 6))
            plt.plot(simulated_prices, alpha=0.1, color='blue')
            plt.title("Simulated Price Paths for Apple's Stock")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.grid(True)
            st.pyplot(plt)

            # Calculating Value at Risk (VaR) at 95% confidence level
            st.subheader("Value at Risk (VaR) Calculation")
            simulated_returns = (simulated_prices.iloc[-1] - last_price) / last_price
            var_95 = np.percentile(simulated_returns, 5)

            # Displaying VaR result
            st.write(f"**20-Day Value at Risk (VaR) at 95% Confidence Level:** {var_95:.2%}")
            st.write("""
            **Interpretation of VaR:**
            The 20-day VaR at 95% confidence level indicates the potential maximum loss that is unlikely to be exceeded within the given period.
            For example, a VaR of -5% suggests that there is only a 5% chance of losing more than 5% of the current stock value over the next 20 days.
            """)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure your EODHD API key is configured correctly and you have access to the required data.")
        return

if __name__ == "__main__":
    main()