import yfinance as yf  # Library to download financial data
import matplotlib.pyplot as plt  # Library for plotting graphs
import streamlit as st  # Library for creating interactive web apps
import pandas as pd  # Library for data manipulation
from statsmodels.tsa.stattools import adfuller  # For the Augmented Dickey-Fuller test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For ACF and PACF plots
import numpy as np  # Library for numerical operations
from scipy.stats import skew, kurtosis  # For statistical calculations


def main():
    # Set up the description of the Streamlit app
    st.write("### Financial Data Analysis with Random Walk Investigation")
    st.write("Analyzing daily prices for S&P 500, FTSE 100, and Gold (SPDR) from 2015 to 2018.")

    # Sidebar options for user to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section with better visibility
    sections = {
        "1. Stationarity Analysis": st.sidebar.checkbox(
            "📊 Stationarity Analysis",
            help="View stationarity tests and price trends"
        ),
        "2. Random Walk Analysis": st.sidebar.checkbox(
            "🎲 Random Walk Analysis",
            help="Examine autocorrelation and random walk properties"
        ),
        "3. Log Returns Vs Arithmetic Returns": st.sidebar.checkbox(
            "📈 Returns Comparison",
            help="Compare log returns with arithmetic returns"
        ),
        "4. Distribution Moments Analysis": st.sidebar.checkbox(
            "📉 Distribution Analysis",
            help="Analyse statistical moments of returns"
        ),
        "5. Annualisation of Return and Volatility": st.sidebar.checkbox(
            "📅 Annual Metrics",
            help="View annualised returns and volatility"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    # Define the financial instruments (tickers) and the date range for analysis
    tickers = {
        "S&P 500": "^GSPC",       # S&P 500 Index
        "FTSE 100": "^FTSE",      # FTSE 100 Index
        "Gold (SPDR)": "GLD"      # SPDR Gold Shares ETF
    }
    start_date = "2015-01-01"
    end_date = "2018-12-31"

    # Download the closing price data for each ticker using yfinance
    @st.cache_data  # Cache the data to prevent repeated downloads
    def get_data():
        return yf.download(
            list(tickers.values()),  # List of ticker symbols
            start=start_date,
            end=end_date
        )["Close"]  # Get the 'Close' price

    data = get_data()
    # Rename the columns to use the keys from the tickers dictionary (instrument names)
    data.columns = tickers.keys()

    # Calculate daily log returns
    # Formula used: log_return = ln(P_t) - ln(P_{t-1})
    log_returns = np.log(data / data.shift(1)).dropna()

    # Compute statistical moments for each instrument and store them in a dictionary
    moments_summary = {}
    for column in log_returns.columns:
        mean = log_returns[column].mean()  # Calculate the mean (average return)
        variance = log_returns[column].var()  # Calculate the variance (risk)
        skewness_value = skew(log_returns[column])  # Calculate skewness
        kurtosis_value = kurtosis(log_returns[column])  # Calculate kurtosis
        # Store the calculated moments in the dictionary
        moments_summary[column] = {
            "Mean": mean,
            "Variance": variance,
            "Skewness": skewness_value,
            "Kurtosis": kurtosis_value
        }

    # 1. Stationarity Analysis
    if sections["1. Stationarity Analysis"]:
        st.markdown("---")  
        st.header("1. Stationarity Analysis")

        # Plot daily closing prices for each instrument
        st.subheader("Daily Closing Prices (2015 - 2018)")
        fig, ax = plt.subplots(figsize=(14, 8))  # Create a figure for plotting
        for column in data.columns:
            ax.plot(data.index, data[column], label=column)  # Plot each instrument's prices
        ax.set_title("Daily Closing Prices (2015 - 2018)")
        ax.legend()
        st.pyplot(fig)  # Display the plot in the Streamlit app

        # Augmented Dickey-Fuller (ADF) test to check for stationarity
        st.subheader("Augmented Dickey-Fuller Test Results")

        # Create three columns for side-by-side display
        col1, col2, col3 = st.columns(3)

        # Function to display ADF results in a consistent format
        def display_adf_results(column, result, col):
            with col:
                st.markdown(f"**{column}**")
                st.markdown(f"""
                🔹 **ADF Statistic:** {result[0]:.4f}  
                🔹 **p-value:** {result[1]:.4f}  
                
                **Critical Values:**  
                • 1%: {result[4]['1%']:.4f}  
                • 5%: {result[4]['5%']:.4f}  
                • 10%: {result[4]['10%']:.4f}  
                
                **Conclusion:**  
                {'✅ Stationary' if result[1] < 0.05 else '❌ Not Stationary'}
                """)
                st.markdown("---")

        # Display results for each instrument in separate columns
        columns = list(data.columns)
        for i, column in enumerate(columns):
            result = adfuller(data[column].dropna())
            if i == 0:
                display_adf_results(column, result, col1)
            elif i == 1:
                display_adf_results(column, result, col2)
            else:
                display_adf_results(column, result, col3)

        # Add a brief interpretation guide below
        st.markdown("""
        #### Interpretation Guide:
        - **ADF Statistic:** A larger negative number means the data is more likely to have a pattern we can analyse
        - **p-value < 0.05:** When this number is smaller than 0.05, the data behaves in a stable, predictable fashion
        - **Critical Values:** These are our benchmark figures - if our ADF Statistic is more negative than these, we can trust there's a pattern in our data
        """)

    # 2. Random Walk Analysis
    if sections["2. Random Walk Analysis"]:
        st.markdown("---")  
        st.header("2. Random Walk Analysis")

        # Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
        st.subheader("Autocorrelation and Partial Autocorrelation Plots")
        for column in data.columns:
            st.write(f"### {column}")

            # Plot the ACF
            fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
            plot_acf(data[column].dropna(), ax=ax_acf, lags=50)
            st.pyplot(fig_acf)

            # Plot the PACF
            fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
            plot_pacf(data[column].dropna(), ax=ax_pacf, lags=50, method='ywm')
            st.pyplot(fig_pacf)
        
        # Observations and Explanations
        st.subheader("Observations and Conclusions")
        st.write("""
        Based on the ACF and PACF plots, we can make the following observations for each time series:

        - **S&P 500 (^GSPC):**
            - The ACF plot shows a slow decay, indicating strong autocorrelation across many lags.
            - The PACF plot has a significant spike at lag 1 and gradually decreases.
            - **Conclusion:** This suggests that the S&P 500 series exhibits characteristics of a random walk.

        - **FTSE 100 (^FTSE):**
            - The ACF plot also displays a slow decay similar to the S&P 500.
            - The PACF plot shows a significant spike at lag 1, followed by smaller spikes.
            - **Conclusion:** The FTSE 100 series appears to follow a random walk behavior.

        - **Gold (SPDR) (GLD):**
            - The ACF plot shows less pronounced autocorrelation compared to the stock indices.
            - The PACF plot may have a significant spike at lag 1 but less so than the stock indices.
            - **Conclusion:** The Gold series may exhibit random walk characteristics but with weaker autocorrelation, suggesting some mean-reverting tendencies.

        **Overall Interpretation:**

        - The slow decay in the ACF plots and significant spike at lag 1 in the PACF plots are characteristic of non-stationary series that may follow a random walk.
        - This indicates that past values have a strong influence on future values, and the series are not easily predictable using simple linear models.
        """)

    # 3. Log Returns Vs Arithmetic Returns
    if sections["3. Log Returns Vs Arithmetic Returns"]:
        st.markdown("---") 
        st.header("3. Log Returns Vs Arithmetic Returns")

        # Calculate arithmetic returns (percentage change) First fill NA values, then calculate percentage change
        data_filled = data.ffill()  
        arithmetic_returns = data_filled.pct_change().dropna()
    
        # Plot comparison of log returns vs arithmetic returns for each instrument
        st.subheader("Comparison of Log Returns and Arithmetic Returns")
        for column in data.columns:
            st.write(f"### {column}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(log_returns.index, log_returns[column], label='Log Returns')
            ax.plot(arithmetic_returns.index, arithmetic_returns[column], label='Arithmetic Returns', alpha=0.7)
            ax.set_title(f"Log Returns vs Arithmetic Returns for {column}")
            ax.legend()
            st.pyplot(fig)
        
        # Explanation of why log returns are preferred
        st.subheader("Why Are Log Returns Preferred Over Arithmetic Returns?")
        st.write("""
        - **Additivity Over Time:**
            - Log returns are time-additive. The sum of log returns over multiple periods equals the log return over the total period (e.g., log returns of 0.02 and 0.03 sum to a total return of 0.05).
        - **Normal Distribution Assumption:**
            - Log returns are often more normally distributed than arithmetic returns. This makes them more suitable for statistical models that assume normality.
        - **Continuous Compounding Properties:**
            - Log returns assume continuous compounding, which aligns with many financial models and theoretical frameworks. This property, combined with the natural properties of logarithms, makes them particularly useful in financial mathematics.
        - **Risk Management Benefits:**
            - For assets with large price changes, log returns prevent negative prices, which can occur when using arithmetic returns.
        """)

    # 4. Distribution Moments Analysis
    if sections["4. Distribution Moments Analysis"]:
        st.markdown("---") 
        st.header("4. Distribution Moments Analysis")

        # Description of calculations and steps
        st.subheader("Calculations and Steps Performed")
        st.write("""
        - **Step 1: Calculate Daily Log Returns**
            - Computed as the natural logarithm of the ratio of consecutive closing prices.
        - **Step 2: Compute the First Four Moments**
            - **Mean (First Moment):** Average of the daily log returns.
            - **Variance (Second Moment):** Measures the dispersion of returns around the mean.
            - **Skewness (Third Moment):** Assesses the asymmetry of the return distribution.
            - **Kurtosis (Fourth Moment):** Evaluates the 'tailedness' of the distribution.
        - **Step 3: Summarise the Results**
            - Stored the computed statistics in a summary table for comparison.
        """)

        # Display the first four moments for each instrument and compare them based on risk and return
        st.subheader("First Four Moments of Daily Log Returns")
        st.write("""
        Comparison of Instruments Based on Risk and Return
        """)

        # Create a table to display the comparison
        comparison_table = {
            "Instrument": [],
            "Mean Return": [],
            "Variance (Risk)": [],
            "Skewness": [],
            "Kurtosis": []
        }

        for instrument, stats in moments_summary.items():
            comparison_table["Instrument"].append(instrument)
            comparison_table["Mean Return"].append(stats['Mean'])
            comparison_table["Variance (Risk)"].append(stats['Variance'])
            comparison_table["Skewness"].append(stats['Skewness'])
            comparison_table["Kurtosis"].append(stats['Kurtosis'])

        # Convert the dictionary to a DataFrame for display
        comparison_df = pd.DataFrame(comparison_table)
        st.table(comparison_df.style.format({
            "Mean Return": "{:.6f}",
            "Variance (Risk)": "{:.8f}",
            "Skewness": "{:.6f}",
            "Kurtosis": "{:.6f}"
        }))

        # Provide analysis of significant differences
        st.subheader("Analysis of Significant Differences")
        st.write("""
        - **Mean Return:**
            - **S&P 500** has the highest mean daily return, suggesting better average performance over the period.
            - **Gold (SPDR)** shows a lower mean return, indicating more modest growth.
        - **Variance (Risk):**
            - **S&P 500** and **FTSE 100** have higher variances, implying greater risk and volatility.
            - **Gold (SPDR)** exhibits lower variance, suggesting it is less volatile compared to the stock indices.
        - **Skewness:**
            - All instruments have skewness values close to zero, indicating relatively symmetric return distributions.
            - Slight negative skewness may suggest a tendency for occasional large negative returns.
        - **Kurtosis:**
            - All instruments show positive kurtosis greater than 0 (since excess kurtosis is calculated), indicating heavier tails than a normal distribution.
            - This implies a higher probability of extreme returns (both positive and negative), which is important for risk management.
        
        **Conclusion:**

        - **Risk-Return Trade-off:**
            - Higher returns are associated with higher risk for the stock indices (S&P 500 and FTSE 100).
            - Gold offers lower returns with lower risk, making it an attractive option for conservatve investors.
        - **Diversification Benefits:**
            - Combining assets with different risk-return profiles, such as stocks and gold, can enhance porfolio diversification, reducing overall risk while maintaining acceptable returns.
        - **Extreme Movements:**
            - The high kurtosis values across all instruments highlight the potential for extreme market events. This highlights the need for robust risk management strategies to mitigate the impact of tail risks.
        """)

  # 5. Annualisation of Return and Volatility
    if sections["5. Annualisation of Return and Volatility"]:
        st.markdown("---") 
        st.header("5. Annualisation of Return and Volatility")
        
        # Create four columns for the expanded layout
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
        
        with col1:
            st.markdown("#### Instrument")
        with col2:
            st.markdown("#### Annual Return")
        with col3:
            st.markdown("#### As Percentage")
        with col4:
            st.markdown("#### Annual Volatility")
        with col5:
            st.markdown("#### As Percentage")

        # Add a divider
        st.markdown("---")

        # Display data for each instrument
        for column in log_returns.columns:
            mean = moments_summary[column]["Mean"]
            variance = moments_summary[column]["Variance"]
            annualised_return = mean * 252
            annualised_volatility = np.sqrt(variance) * np.sqrt(252)
            
            # Create columns for each row of data
            c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
            
            with c1:
                st.markdown(f"**{column}**")
            with c2:
                st.markdown(f"**{annualised_return:.6f}**")
            with c3:
                st.markdown(f"**{annualised_return:.2%}**")
            with c4:
                st.markdown(f"**{annualised_volatility:.6f}**")
            with c5:
                st.markdown(f"**{annualised_volatility:.2%}**")
                
            # Add subtle separator between rows
            st.markdown("<hr style='margin: 5px 0; opacity: 0.2'>", unsafe_allow_html=True)

               
        # Explanation of the calculations and rationale
        st.subheader("Explanation of Calculations and Annualisation Rationale")
        st.write(
            """
        - **Mean Daily Return**: Multiplied by 252 (approximate number of trading days in a year) to annualise the return.
        - **Daily Volatility**: Scaled by √252 to annualise volatility, reflecting annual risk.
        - **Rationale for Annualisation**: Annualising allows for consistent comparison of returns and risks on a yearly basis, aiding investment decisions.
            """
        )

if __name__ == "__main__":
    main()