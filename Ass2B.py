# Importing the necessary libraries for data retrieval, analysis, and modelling
import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta

def main():
    # Title of the application
    st.write("### Mean-Variance Optimisation & Machine Learning (Neural Networks)")

    # Selecting stock symbols for the analysis - choosing large, well-known technology stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=5*365)).strftime('%Y-%m-%01')

    # Sidebar options for user to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section with better visibility
    sections = {
        "1. Mean-Variance Optimisation": st.sidebar.checkbox(
            "📊 Mean-Variance Analysis",
            help="View traditional portfolio optimisation results"
        ),
        "2. Neural Network Enhancement": st.sidebar.checkbox(
            "🧠 Neural Network Analysis",
            help="View ML-enhanced portfolio optimisation"
        ),
        "3. Comparison": st.sidebar.checkbox(
            "📈 Results Comparison",
            help="Compare traditional vs ML approaches"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    # Function definitions
    def download_stock_data(tickers, start_date):
        # Retrieve adjusted closing price data for the specified tickers from the given start date until now
        data = yf.download(tickers, start=start_date, end=datetime.now())['Adj Close']
        return data

    def calculate_daily_returns(data):
        # Compute daily returns by taking the percentage change in adjusted closing prices
        return data.pct_change().dropna()

    def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
        # Calculate annualised returns, volatility, and Sharpe ratio
        returns = np.sum(mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (returns - risk_free_rate) / volatility
        return returns, volatility, sharpe_ratio

    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        # Function to be minimised in optimisation
        return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

    # Set optimisation parameters
    num_assets = len(tickers)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    # Initialise session state for storing results
    if 'mean_variance_results' not in st.session_state:
        st.session_state.mean_variance_results = None
    if 'nn_enhanced_results' not in st.session_state:
        st.session_state.nn_enhanced_results = None

    # Download data once for all sections
    stock_data = download_stock_data(tickers, start_date)
    daily_returns = calculate_daily_returns(stock_data)

    # PART 1: Traditional Mean-Variance Optimisation
    if sections["1. Mean-Variance Optimisation"]:
        st.markdown("---")
        st.header("Traditional Mean-Variance Optimisation")

        # Display historical stock prices
        st.subheader("Stock Prices (5-Year Historical)")
        fig, ax = plt.subplots()
        stock_data.plot(ax=ax)
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.title('Stock Prices for AAPL, MSFT, GOOGL')
        st.pyplot(fig)

        # Display daily returns
        st.subheader("Daily Percentage Returns")
        fig, ax = plt.subplots()
        daily_returns.plot(ax=ax)
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        plt.title('Daily Percentage Returns for AAPL, MSFT, GOOGL')
        st.pyplot(fig)

        # Show descriptive statistics
        st.subheader("Daily Returns Summary Statistics")
        st.write(daily_returns.describe())

        # Implementing Markowitz Mean-Variance Optimisation
        st.subheader("Optimisation: Markowitz Mean-Variance Optimisation")
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        risk_free_rate = 0.02

        # Run optimization
        with st.spinner('Optimising portfolio weights...'):
            optimal_result = minimize(negative_sharpe_ratio, initial_weights, 
                                   args=(mean_returns, cov_matrix, risk_free_rate),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights = optimal_result.x

        # Display optimal allocation
        st.subheader("Optimal Portfolio Weights")
        optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=['Weight'])
        st.write(optimal_weights_df)

        # Calculate and display performance
        returns, volatility, sharpe_ratio = portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)
        st.subheader("Optimal Portfolio Performance")
        st.write(f"Expected Annual Return: {returns:.2f}%")
        st.write(f"Annual Volatility: {volatility:.2f}%")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Store results
        st.session_state.mean_variance_results = (returns, volatility, sharpe_ratio)

    # PART 2: Neural Network Enhancement
    if sections["2. Neural Network Enhancement"]:
        st.markdown("---")
        st.header("Neural Network Enhanced Optimisation")

        # Prepare data for neural network
        st.write("Preparing data for MLPRegressor Neural Network...")
        lagged_returns = daily_returns.shift(1).dropna()
        daily_returns_aligned = daily_returns.loc[lagged_returns.index]
        X = lagged_returns.values
        y = daily_returns_aligned.values

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            # Train neural network
            st.write("Training MLPRegressor Neural Network...")
            model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', 
                               solver='adam', max_iter=500, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make predictions
            predicted_returns = model.predict(X_test_scaled)
            st.subheader("Predicted Returns using MLPRegressor Neural Network")
            predicted_returns_df = pd.DataFrame(predicted_returns, 
                                             columns=[f"{ticker} Predicted Return" for ticker in tickers])
            st.write("First few predictions:")
            st.write(predicted_returns_df.head())

            # Use predictions for optimisation
            mean_predicted_returns = pd.Series(np.mean(predicted_returns, axis=0), 
                                            index=tickers, name='Mean')
            st.write("Mean Predicted Returns from Neural Network:")
            st.write(mean_predicted_returns)

            # Display covariance matrix
            cov_matrix = daily_returns.cov()
            st.write("Covariance Matrix from MLPRegressor Neural Network:")
            st.write(cov_matrix)

            # Set and display risk-free rate
            risk_free_rate = 0.02
            st.write(f"Risk-Free Rate: {risk_free_rate:.2%}")

            # Optimize with predicted returns
            with st.spinner('Optimising portfolio weights using predicted returns...'):
                optimal_result_predicted = minimize(negative_sharpe_ratio, initial_weights, 
                                                 args=(mean_predicted_returns, cov_matrix, risk_free_rate),
                                                 method='SLSQP', bounds=bounds, constraints=constraints)
                optimal_weights_predicted = optimal_result_predicted.x

            # Display results
            st.subheader("Optimal Portfolio Weights using Predicted Returns")
            optimal_weights_predicted_df = pd.DataFrame(optimal_weights_predicted, 
                                                     index=tickers, columns=['Weight'])
            st.write(optimal_weights_predicted_df)

            returns_predicted, volatility_predicted, sharpe_ratio_predicted = portfolio_performance(
                optimal_weights_predicted, mean_predicted_returns, cov_matrix, risk_free_rate)

            st.subheader("Optimal Portfolio Performance using Predicted Returns")
            st.write(f"Expected Annual Return: {returns_predicted:.2f}%")
            st.write(f"Annual Volatility: {volatility_predicted:.2f}%")
            st.write(f"Sharpe Ratio: {sharpe_ratio_predicted:.2f}")

            st.session_state.nn_enhanced_results = (returns_predicted, volatility_predicted, 
                                                  sharpe_ratio_predicted)

        except Exception as e:
            st.error(f"An error occurred during the neural network analysis: {str(e)}")

    # PART 3: Comparison
    if sections["3. Comparison"]:
        st.markdown("---")
        st.header("Performance Comparison")

        if (st.session_state.mean_variance_results is not None and 
            st.session_state.nn_enhanced_results is not None):
            
            mean_variance_returns, mean_variance_volatility, mean_variance_sharpe = st.session_state.mean_variance_results
            nn_enhanced_returns, nn_enhanced_volatility, nn_enhanced_sharpe = st.session_state.nn_enhanced_results

            # Create comparison table
            comparison_data = {
                "Metric": ["Expected Return (Annualised)", "Volatility (Annualised)", "Sharpe Ratio"],
                "Mean-Variance Portfolio": [mean_variance_returns, mean_variance_volatility, 
                                         mean_variance_sharpe],
                "NN-Enhanced Portfolio": [nn_enhanced_returns, nn_enhanced_volatility, 
                                        nn_enhanced_sharpe]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.write(comparison_df)

            # Interpretation
            st.header("Discussion")
            st.write("### Interpretation of Results")
            if nn_enhanced_sharpe > mean_variance_sharpe:
                st.write(f"""The NN-Enhanced Portfolio has a higher risk-adjusted return with a Sharpe Ratio of {nn_enhanced_sharpe:.2f}, compared to the Mean-Variance Portfolio's Sharpe Ratio of {mean_variance_sharpe:.2f}. This suggests that the neural network was able to improve the Sharpe ratio by better predicting future returns.""")
            else:
                st.write(f"""The Mean-Variance Portfolio has a higher risk-adjusted return with a Sharpe Ratio of {mean_variance_sharpe:.2f}, compared to the NN-Enhanced Portfolio's Sharpe Ratio of {nn_enhanced_sharpe:.2f}. The neural network did not improve the Sharpe ratio in this case, possibly due to overfitting or other challenges.""")

            st.write("### Challenges with Machine Learning")
            st.write("""Machine learning models can suffer from issues like overfitting, where they learn patterns specific to the training set that do not generalise well to new data. Moreover, they may struggle to adapt to unexpected market conditions and can be data intensive.""")

            st.write("### Practical Considerations")
            st.write("""While neural networks can capture complex relationships, their complexity and the need for extensive tuning may not always justify their use over simpler, well-understood methods like Mean-Variance Optimisation. In many practical scenarios, a straightforward approach can be more robust, transparent, and easier to implement.""")
        else:
            st.warning("Please complete Part 1 and Part 2 before comparing results.")

if __name__ == "__main__":
    main()