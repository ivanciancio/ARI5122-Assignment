# Import necessary libraries
import yfinance as yf  # For downloading financial data from Yahoo Finance
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting graphs
import streamlit as st  # For creating the web application interface
from arch import arch_model  # For implementing the GARCH model
from sklearn.neural_network import MLPRegressor  # For the neural network model
from sklearn.preprocessing import StandardScaler  # For scaling data before training

def main():
    # Title
    st.write("### Predicting Financial Market Volatility")

    # Initialise session state variables if they do not exist
    # Using session state allows data to persist across user interactions
    for var in ['garch_mse', 'nn_mse', 'garch_params', 'nn_params', 
                'garch_predictions', 'nn_predictions', 'actual_volatility']:
        if var not in st.session_state:
            st.session_state[var] = 'N/A' if 'mse' in var else None

    # Get stock ticker input above the analyses options
    selected_stock = st.text_input("Enter Stock Ticker (e.g., AAPL, ^GSPC, BTC-USD)")

    # Sidebar options for user to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section
    sections = {
        "1. GARCH Analysis": st.sidebar.checkbox(
            "📊 GARCH Model Analysis",
            help="View GARCH(1,1) model volatility forecasting",
            key="garch_check"
        ),
        "2. Neural Network": st.sidebar.checkbox(
            "🧠 Neural Network Analysis",
            help="View neural network volatility forecasting",
            key="nn_check"
        ),
        "3. Model Comparison": st.sidebar.checkbox(
            "📈 Model Comparison",
            help="Compare GARCH and Neural Network performance",
            key="comparison"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    # Function to download data from Yahoo Finance
    @st.cache_data  # Cache the data retrieval to speed up subsequent runs
    def get_data(ticker):
        # Fetch historical price data starting from 2022-01-01 until the current date
        # This approach provides a sufficient window for volatility modelling
        data = yf.download(ticker, start="2022-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        return data

    if selected_stock:
        data = get_data(selected_stock)
        if not data.empty:
            # Confirm that data retrieval was successful before proceeding
            st.write(f"### Displaying data for: {selected_stock}")
            st.dataframe(data.tail())  # Show the last few rows of data to verify correctness

            # Data Preparation: Calculate Daily Returns
            data['Return'] = data['Adj Close'].pct_change()  # Compute percentage returns from adjusted close prices
            returns = (data['Return'].dropna()) * 100  # Convert to percentage returns and remove missing values

            # Split the Data into Training and Testing sets
            # Using an 80/20 split is standard practice to preserve unseen data for out-of-sample testing
            split_index = int(len(returns) * 0.8)
            train, test = returns[:split_index], returns[split_index:]

            # GARCH Analysis Section
            if sections["1. GARCH Analysis"]:
                st.markdown("---")
                st.header("GARCH Model Analysis")
                
                # Part 1: Employing a GARCH(1,1) model for volatility forecasting
                st.write("## Fitting GARCH(1,1) Model")
                # Set up and fit a basic GARCH(1,1) model, commonly used to model volatility clustering in finance
                model = arch_model(train, vol='Garch', p=1, q=1, rescale=False)
                model_fit = model.fit(disp='off')  # Fit the model to the training data without detailed output
                st.write(model_fit.summary())  # Display a detailed summary of the fitted GARCH model parameters

                # Explanations of the data preparation, model specification, and assumptions
                st.write("## Model Fitting Steps and Assumptions")
                st.write("### Data Preparation Steps:")
                st.markdown("""
                1. Calculated daily returns from adjusted closing prices
                2. Converted to percentage returns (multiplied by 100)
                3. Split data into 80% training and 20% testing sets
                """)

                st.write("### Model Specification:")
                st.markdown("""
                - GARCH(1,1) Parameters:
                    - p=1: One lagged GARCH term to model persistence in volatility
                    - q=1: One lagged ARCH term capturing recent volatility shocks
                - Normal error distribution assumption
                - rescale=False ensures raw returns are modeled
                """)

                st.write("### Estimation Method:")
                st.markdown("""
                - Maximum Likelihood Estimation (MLE) for parameter fitting
                - Silent output for cleaner presentation
                """)

                st.write("### Key Assumptions:")
                st.markdown("""
                1. Stationarity of returns
                2. Presence of volatility clustering
                3. Conditional normality of errors
                4. Mean-reverting volatility process
                5. Historical volatility is informative about future volatility
                """)

                # Store model parameters for later inspection or comparison
                st.session_state.garch_params = model_fit.params

                # Implemented a rolling forecast to obtain out-of-sample volatility predictions
                st.write("## Rolling Forecast")
                rolling_predictions = []
                test_size = len(test)

                st.write("### Rolling Forecast Implementation:")
                st.markdown("""
                - Perform one-step-ahead forecasts by expanding the training window incrementally
                - At each step, the model is refit to all available historical data for updated predictions
                """)

                for i in range(test_size):
                    # Expand the training dataset by one data point at a time
                    train_data = returns[:split_index + i]
                    model = arch_model(train_data, vol="Garch", p=1, q=1, rescale=False)
                    model_fit = model.fit(disp="off")
                    forecast = model_fit.forecast(horizon=1)
                    # Extract the predicted variance and take the square root to get volatility
                    rolling_predictions.append(np.sqrt(forecast.variance.values[-1, :][0]))

                # Converted rolling predictions into a Pandas Series for convenience
                rolling_predictions = pd.Series(rolling_predictions, index=test.index)
                # Actual volatility approximated as the rolling standard deviation over a 5-day window
                actual_volatility = test.rolling(window=5).std()

                # Save predictions and actual volatility for Part 3 comparison
                st.session_state.garch_predictions = rolling_predictions
                st.session_state.actual_volatility = actual_volatility

                # Plot Actual vs Predicted Volatility for visual inspection
                st.write("## Actual vs Predicted Volatility")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(actual_volatility, label="Actual Volatility", color="blue")
                ax.plot(rolling_predictions, label="GARCH(1,1) Predicted Volatility", color="red")
                ax.set_title(f"Volatility Forecasting for {selected_stock}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volatility")
                ax.legend()
                st.pyplot(fig)

                # Calculate performance using Mean Squared Error
                st.write("## Model Performance Evaluation")
                garch_mse = np.mean((rolling_predictions - actual_volatility) ** 2)
                st.session_state.garch_mse = garch_mse
                st.write(f"Mean Squared Error (MSE) of the GARCH(1,1) Model: {garch_mse:.6f}")

                # Explaining the significance of MSE in evaluating forecast accuracy
                st.write("## Analysis Explanation")
                st.write("A lower MSE indicates that the predicted volatility series is closer to the actual values, thus signifying better model performance.")

            # Neural Network Analysis Section
            if sections["2. Neural Network"]:
                st.markdown("---")
                st.header("Neural Network Analysis")

                # Here we assume that volatility relates to recent returns and apply a basic NN approach
                # Feature Preparation: Use lagged returns as features to predict the current period's volatility
                X_train = pd.DataFrame(train).shift(1).dropna()
                y_train = train.loc[X_train.index]
                X_test = pd.DataFrame(test).shift(1).dropna()
                y_test = test.loc[X_test.index]

                st.write("## Fitting Feed-Forward Neural Network Model")
                st.write("### Neural Network Architecture and Design Choices:")

                # Explaining the reasoning for the chosen network architecture and training settings
                st.markdown("""
                1. **Structure**:
                   - Input: Previous day's return
                   - Hidden Layer 1: 50 neurons
                   - Hidden Layer 2: 25 neurons
                   - Output: 1 neuron for predicted volatility

                2. **Justification**:
                   - Two hidden layers capture more complex nonlinear patterns than a single layer.
                   - Gradually reducing neurons (50 → 25) helps manage complexity and reduces overfitting risk.

                3. **Activation**:
                   - ReLU activation in hidden layers is computationally efficient and avoids vanishing gradients.

                4. **Training Parameters**:
                   - Max iterations: 1000 ensures sufficient training time.
                   - Random state: 42 ensures reproducibility.
                   - Adam optimiser: Efficient method suitable for this scale of data.
                """)

                # Scale the features to improve training stability and convergence
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Define and train the MLPRegressor as a volatility predictor
                nn_model = MLPRegressor(
                    hidden_layer_sizes=(50, 25),
                    activation='relu',
                    max_iter=1000,
                    random_state=42
                )
                nn_model.fit(X_train_scaled, y_train)

                # Store NN model parameters for later reference
                st.session_state.nn_params = nn_model.get_params()

                # Rolling Forecast to Predict Future Volatility with the Neural Network
                st.write("## Rolling Forecast")
                st.write("### Rolling Forecast Implementation:")
                st.markdown("""
                Similar to the GARCH approach, we use a rolling forecast:
                - Incrementally update the dataset as time moves forward.
                - Predict the next step volatility using the trained model.
                """)

                rolling_predictions = []
                for i in range(len(X_test_scaled)):
                    rolling_predictions.append(nn_model.predict([X_test_scaled[i]])[0])

                rolling_predictions = pd.Series(rolling_predictions, index=y_test.index)
                actual_volatility = y_test.rolling(window=5).std()

                # Save these predictions and actual values for comparison in Part 3
                st.session_state.nn_predictions = rolling_predictions
                st.session_state.actual_volatility = actual_volatility

                # Display basic statistics of the rolling predictions
                st.write("### Rolling Forecast Statistics:")
                st.write(f"""
                - Number of predictions: {len(rolling_predictions)}
                - Forecast period: {rolling_predictions.index[0].strftime('%Y-%m-%d')} to {rolling_predictions.index[-1].strftime('%Y-%m-%d')}
                - Mean predicted value: {rolling_predictions.mean():.4f}
                - Standard deviation of predictions: {rolling_predictions.std():.4f}
                """)

                # Plot Actual vs Predicted Volatility using the Neural Network model
                st.write("## Actual vs Predicted Volatility")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(actual_volatility, label="Actual Volatility", color="blue")
                ax.plot(rolling_predictions, label="Neural Network Predicted Volatility", color="green")
                ax.set_title(f"Volatility Forecasting for {selected_stock} using Neural Network")
                ax.set_xlabel("Date")
                ax.set_ylabel("Volatility")
                ax.legend()
                st.pyplot(fig)

                # Evaluate the performance of the Neural Network model using MSE
                st.write("## Model Performance Evaluation")
                nn_mse = np.mean((rolling_predictions - actual_volatility) ** 2)
                st.session_state.nn_mse = nn_mse
                st.write(f"Mean Squared Error (MSE) of the Neural Network Model: {nn_mse:.6f}")

                # How MSE measures forecast accuracy and what the values tell us about model performance
                st.write("## Analysis Explanation")
                st.write("As with the GARCH model, MSE here helps us understand how closely the neural network's predicted volatility matches the observed market volatility.")

            # Model Comparison Section
            if sections["3. Model Comparison"]:
                st.markdown("---")
                st.header("Model Comparison")

                if (st.session_state.garch_mse == 'N/A' or st.session_state.nn_mse == 'N/A'):
                    st.write("Please run both GARCH and Neural Network analyses first.")
                else:
                    if (st.session_state.garch_predictions is not None and 
                        st.session_state.nn_predictions is not None and 
                        st.session_state.actual_volatility is not None):

                        # Calculate additional comparison metrics
                        garch_mae = np.mean(np.abs(st.session_state.garch_predictions - st.session_state.actual_volatility))
                        nn_mae = np.mean(np.abs(st.session_state.nn_predictions - st.session_state.actual_volatility))

                        st.write("### Mean Squared Error (MSE):")
                        st.write(f"- GARCH(1,1) Model: **{st.session_state.garch_mse:.6f}**")
                        st.write(f"- Neural Network Model: **{st.session_state.nn_mse:.6f}**")

                        st.write("### Mean Absolute Error (MAE):")
                        st.write(f"- GARCH(1,1) Model: **{garch_mae:.6f}**")
                        st.write(f"- Neural Network Model: **{nn_mae:.6f}**")
                        

                        ## Model comparison visualisation
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(st.session_state.actual_volatility, label="Actual Volatility", color="blue")
                        ax.plot(st.session_state.garch_predictions, label="GARCH Predictions", 
                               color="red", alpha=0.7)
                        ax.plot(st.session_state.nn_predictions, label="Neural Network Predictions", 
                               color="green", alpha=0.7)
                        ax.set_title(f"Volatility Forecasting Comparison for {selected_stock}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Volatility")
                        ax.legend()
                        st.pyplot(fig)

                        # Model performance comparison
                        better_model = "GARCH(1,1)" if st.session_state.garch_mse < st.session_state.nn_mse else "Neural Network"
                        st.write(f"The {better_model} model shows better performance for this dataset.")

                        # Provide a discussion about which model is more effective and why
                        st.write("## Which approach was more effective for this dataset, and why?")
                        if st.session_state.garch_mse < st.session_state.nn_mse:
                            st.write("The **GARCH(1,1) Model** was more effective, likely due to its direct modelling of volatility clustering, a common characteristic in financial returns.")
                        else:
                            st.write("The **Neural Network Model** performed better, possibly capturing nonlinear patterns in returns that the GARCH model could not.")

                        # Summarise key findings
                        st.write("## Findings")
                        st.write("The model with the lower MSE and MAE generally provides a better fit to the observed volatility pattern. This assessment helps in choosing a suitable modelling technique for volatility forecasting.")

                        # Strengths and limitations analysis
                        st.write("## Strengths and Limitations")
                        st.write("### GARCH Model:")
                        st.markdown("""
                        - **Strengths**: Well-established for modelling volatility clustering and mean reversion.
                        - **Limitations**: Parametric assumptions may restrict the model's flexibility and adaptability.
                        """)

                        st.write("### Neural Network Model:")
                        st.markdown("""
                        - **Strengths**: Able to model complex nonlinear relationships and adapt to various patterns.
                        - **Limitations**: Requires careful tuning, may need more data, and can be prone to overfitting if not regularised.
                        """)

                        st.write("## Potential Improvements")
                        st.markdown("""
                        - **GARCH Model**: Consider EGARCH or GJR-GARCH variants to capture uneven patterns in volatility.
                        - **Neural Network Model**: Add more complex features (lagged volatilities, volumes, technical indicators) and explore LSTM architectures for better temporal modeling.
                        """)

                # Display model parameters
                st.write("## Model Parameters")
                if st.session_state.garch_params is not None:
                    st.write("### GARCH(1,1) Model Parameters:")
                    st.write(st.session_state.garch_params)

                if st.session_state.nn_params is not None:
                    st.write("### Neural Network Model Parameters:")
                    st.write(st.session_state.nn_params)

if __name__ == "__main__":
    main()