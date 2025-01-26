# Import necessary libraries
import streamlit as st  # Streamlit is used to create web-based dashboards and interactive data applications
import pandas as pd  # Pandas is employed for efficient data manipulation and analysis
from eodhdHelper import get_client  # Import EODHD client from helper module
import datetime  # The datetime module assists in handling dates and times
from dateutil.relativedelta import relativedelta  # relativedelta simplifies date arithmetic, e.g., subtracting months
import statsmodels.api as sm  # statsmodels is used for statistical and econometric modelling
import ssl  # SSL is required for creating secure, though unverified, SSL contexts
import urllib.request  # urllib is used here to open URLs for data retrieval
import io  # io helps in dealing with byte streams, which is useful when handling in-memory files
import zipfile  # zipfile allows extraction of contents from ZIP archives
from pandas.tseries.offsets import MonthEnd  # MonthEnd adjusts dates to the end of the month

def main():
    # Title
    st.write("### Factor Models & Interpreting Risk")

    # Sidebar options to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section
    sections = {
        "1. Data Collection": st.sidebar.checkbox(
            "📊 Data Collection",
            help="View data collection and preparation steps"
        ),
        "2. Factor Models": st.sidebar.checkbox(
            "📈 Factor Model Calculations",
            help="View CAPM and Fama-French model calculations"
        ),
        "3. Analysis": st.sidebar.checkbox(
            "📑 Analysis and Interpretation",
            help="View model interpretation and findings"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    # Data Collection and Preparation
    if sections["1. Data Collection"]:
        st.markdown("---")
        st.subheader("Data Collection and Preparation")
        
        # Define dates
        
        end_date = datetime.datetime(2018, 12, 31)  # End of December 2018
        start_date = end_date - relativedelta(months=59)  # Approximately 5 years prior
               
        # Stock ticker input
        selected_stock = st.text_input("Enter Stock Ticker", value="AMZN")
        
        if selected_stock:
            try:
                # Initialize EODHD client
                client = get_client()
                
                st.write(f"### Obtaining {selected_stock} Stock Data")
                st.write(f"Collecting monthly returns from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}.")
                
                # Download stock data from EODHD
                stock_data = client.download(
                    selected_stock, 
                    start=start_date.strftime('%Y-%m-%d'), 
                    end=end_date.strftime('%Y-%m-%d'), 
                    interval='1mo'
                )
                                
                # Explain the data preprocessing steps
                st.write("### Data Preprocessing")
                st.write("Below are the preprocessing steps performed on the obtained data:")
                
                # Reset index and convert dates
                stock_data.reset_index(inplace=True)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])

                # Resample to monthly data
                monthly_data = stock_data.set_index('Date').resample('ME').last()
                monthly_data.reset_index(inplace=True)

                # Calculate monthly returns
                monthly_data['Monthly Return'] = monthly_data['Adj Close'].pct_change()

                st.write("2. Converted to monthly data and computed returns.")

                # Present the preprocessed data
                st.write("### Preprocessed Amazon Stock Data")
                processed_data = monthly_data[['Date', 'Adj Close', 'Monthly Return']].dropna()
                processed_data.columns = ['Date', 'Adjusted Close', 'Monthly Return']
                processed_data['Date'] = processed_data['Date'].dt.strftime('%Y-%m-%d')
                processed_data['Monthly Return'] = processed_data['Monthly Return'] * 100  # Convert to percentage
                st.dataframe(processed_data)
                
                # Preprocessing steps description
                st.write("### Preprocessing Steps Before Calculating Factor Models")
                st.write("""
                Before calculating the factor models, the following preprocessing steps were performed to ensure data quality and consistency:

                - **Data Cleaning**: Removed any rows with missing values to prevent errors in calculations and ensure accurate results.

                - **Date Formatting**: Converted the index to a datetime format and reset it to a 'Date' column for easier merging with factor data.

                - **Calculating Returns**: Computed the monthly returns using the percentage change in adjusted closing prices, which is essential for comparing stock performance with factor returns.

                - **Data Alignment**: Although not fully executed in this part, aligning the dates of stock returns with the Fama-French factor dates is crucial. This involves adjusting the stock data dates to match the month-end dates used in the factor data.

                These preprocessing steps are vital to prepare the data for accurate factor model calculations in the subsequent analysis.
                """)

            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
                st.info("Make sure to include the correct suffix (e.g., '.US' for US stocks)")

    # Factor Model Calculations
    if sections["2. Factor Models"]:
        st.markdown("---")
        st.subheader("Factor Model Calculations")
        
        # Stock ticker input
        selected_stock = st.text_input("Enter Stock Ticker for Factor Analysis", value="AMZN")
        
        # Dates remain the same
        end_date = datetime.datetime(2018, 12, 31)
        start_date = end_date - relativedelta(months=59)
        
        try:
            # Initialize EODHD client
            client = get_client()
            
            # Download stock data from EODHD
            stock_data = client.download(
                selected_stock, 
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'), 
                interval='1mo'
            )

            # Convert index to the month-end of each period for proper alignment with factor data
            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data.index = stock_data.index.to_period('M').to_timestamp('M')
            stock_data.sort_index(inplace=True)

            # Calculate the monthly return
            stock_data['Monthly Return'] = stock_data['Adj Close'].pct_change()

            # Resample to monthly data, taking the last trading day of each month
            monthly_data = stock_data.resample('ME').last()

            # Reset index to make 'Date' a column again
            monthly_data.reset_index(inplace=True)

            # Calculate monthly returns
            monthly_data['Monthly Return'] = monthly_data['Adj Close'].pct_change() * 100  # Convert to percentage

            # Drop the first row (which will have NaN return)
            monthly_data = monthly_data.dropna()

            # Display the processed monthly return data
            st.write("### Monthly Returns Data - Amazon")
            st.dataframe(monthly_data[['Date', 'Adj Close', 'Monthly Return']])

            # SSL context and Fama-French data loading section
            ssl_context = ssl._create_unverified_context()

            st.write("### Loading Fama-French Data")

            # URL of the Fama-French 5-Factor dataset (monthly frequencies)
            ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

            # Download and extract the zipped Fama-French data
            with urllib.request.urlopen(ff_url, context=ssl_context) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as zip_file:
                    csv_filename = zip_file.namelist()[0]
                    with zip_file.open(csv_filename) as csv_file:
                        ff_data = pd.read_csv(csv_file, skiprows=3, header=None)

            # Assign suitable column names to the Fama-French dataset
            ff_data.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            ff_data = ff_data.dropna(subset=['Date'])
            ff_data['Date'] = ff_data['Date'].astype(str).str.strip()

            # Remove footer and clean data
            footer_index = ff_data[ff_data['Date'] == 'Annual Factors: January-December'].index
            if not footer_index.empty:
                ff_data = ff_data.loc[:footer_index[0]-1]

            # Process dates and filter data
            ff_data = ff_data[ff_data['Date'].str.match(r'^\d{6}$')]
            ff_data['Date'] = pd.to_datetime(ff_data['Date'], format='%Y%m') + MonthEnd(0)
            ff_data.set_index('Date', inplace=True)
            ff_data = ff_data.apply(pd.to_numeric, errors='coerce') / 100
            ff_data = ff_data.loc[start_date:end_date]

            # Display a preview of the Fama-French factor data
            st.write("### Fama-French Data Preview")
            st.dataframe(ff_data.head())

            # Join stock return data with Fama-French factors
            aligned_data = stock_data[['Monthly Return']].join(ff_data, how='inner')
            st.write("### Aligned Data Preview")
            st.write("Number of observations before dropping NaNs:", len(aligned_data))
            st.dataframe(aligned_data.head())

            # Check for missing values
            st.write("Checking for NaN values in aligned_data:")
            st.write(aligned_data.isna().sum())

            # Prepare regression data
            regression_data = aligned_data.dropna(subset=['Monthly Return', 'RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
            st.write("Number of observations after dropping NaNs:", len(regression_data))

            if len(regression_data) > 0:
                # Prepare regression variables
                excess_returns = regression_data['Monthly Return'] - regression_data['RF']
                market_premium = regression_data['Mkt-RF']
                smb = regression_data['SMB']
                hml = regression_data['HML']
                rmw = regression_data['RMW']
                cma = regression_data['CMA']

                # Run regression to calculate stock's beta and market relationship using CAPM
                st.write("### 1. CAPM Model Results")
                X_capm = sm.add_constant(market_premium)
                capm_model = sm.OLS(excess_returns, X_capm).fit()
                st.write("CAPM Regression Results:")
                st.write(capm_model.summary())
                st.write(f"Market Beta: {capm_model.params.get('Mkt-RF', float('nan')):.4f}")
                st.write(f"R-squared: {capm_model.rsquared:.4f}")

                # Fama-French 3-Factor Regression
                st.write("### 2. Fama-French 3-Factor Model Results")
                X_ff3 = sm.add_constant(pd.concat([market_premium, smb, hml], axis=1))
                ff3_model = sm.OLS(excess_returns, X_ff3).fit()
                st.write("Fama-French 3-Factor Regression Results:")
                st.write(ff3_model.summary())
                st.write("Factor Loadings:")
                for factor, beta in ff3_model.params.items():
                    st.write(f"{factor}: {beta:.4f}")

                # Fama-French 5-Factor Regression
                st.write("### 3. Fama-French 5-Factor Model Results")
                X_ff5 = sm.add_constant(pd.concat([market_premium, smb, hml, rmw, cma], axis=1))
                ff5_model = sm.OLS(excess_returns, X_ff5).fit()
                st.write("Fama-French 5-Factor Regression Results:")
                st.write(ff5_model.summary())
                st.write("Factor Loadings:")
                for factor, beta in ff5_model.params.items():
                    st.write(f"{factor}: {beta:.4f}")

                # Compare models
                st.write("### Model Comparison")
                comparison_df = pd.DataFrame({
                    'CAPM': [capm_model.rsquared, capm_model.params.get('Mkt-RF', None), None, None, None, None],
                    'FF3': [ff3_model.rsquared] + [ff3_model.params.get(factor, None) for factor in ['Mkt-RF', 'SMB', 'HML']] + [None, None],
                    'FF5': [ff5_model.rsquared] + [ff5_model.params.get(factor, None) for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
                }, index=['R-squared', 'Market Beta', 'SMB', 'HML', 'RMW', 'CMA'])
                st.write(comparison_df)

                # Display challenges and considerations
                st.write("### Challenges and Considerations")
                st.write("""
                - **Data Alignment**: Ensuring the dates in stock data and Fama-French data match. We converted stock data dates to month-end to align with factor data.

                - **Handling MultiIndex Columns**: Flattened MultiIndex columns returned by `EODHD to single-level columns for easier handling.

                - **Missing Values**: Handled missing or non-numeric data to prevent calculation errors. Converted factor returns to numeric and dropped missing values.

                - **Data Cleaning**: Removed non-data rows from the Fama-French dataset (e.g., footers and annual factors) to prevent parsing errors.

                - **Regression Data Preparation**: Dropped rows with NaN values in the relevant columns before performing regression to avoid invalid outputs.

                - **Model Assumptions**: Regression models rely on statistical assumptions. We reviewed regression outputs to assess the validity of these models.

                - **Error Handling**: Implemented try-except blocks to catch and report errors during data loading and calculations.
                         """)
            else:
                st.error("No data available for regression after dropping NaN values.")
        except Exception as e:
            st.error(f"Error in calculations: {str(e)}")
            st.write("Please review the data loading and preparation steps.")

    # Analysis and Interpretation Section
    if sections["3. Analysis"]:
        st.markdown("---")
        st.subheader("Analysis and Interpretation of Factor Models for Amazon")
        
        # Model interpretation section
        st.write("### Interpretation of Model Outputs")
        st.write("""In this part, we interpret the output from each model, focusing on the betas and other relevant statistics. The interpretation of each model's output provides insights into Amazon's sensitivity to various risk factors:""")
        
        # CAPM interpretation
        st.write("""- **CAPM (Capital Asset Pricing Model)**: The CAPM model provides a single beta coefficient that measures Amazon's sensitivity to the overall market. A beta greater than 1 indicates that Amazon's stock is more volatile compared to the market, while a beta less than 1 suggests lower volatility. The CAPM beta helps understand Amazon's exposure to market risk but does not account for other risk factors.""")
        
        # Fama-French 3-Factor Model interpretation
        st.write("""- **Fama-French 3-Factor Model**: The Fama-French 3-Factor Model extends CAPM by including two additional factors: SMB (Small Minus Big) and HML (High Minus Low). The SMB factor captures the size effect, indicating whether Amazon behaves more like a small or large company. The HML factor captures the value effect, showing whether Amazon's returns are influenced by value or growth characteristics. These additional betas provide a more nuanced understanding of Amazon's exposure to size and value risk.""")
        
        # Fama-French 5-Factor Model interpretation
        st.write("""- **Fama-French 5-Factor Model**: The Fama-French 5-Factor Model further extends the 3-Factor Model by adding RMW (Robust Minus Weak) and CMA (Conservative Minus Aggressive) factors. The RMW factor represents profitability, indicating whether Amazon's returns are influenced by high or low profitability. The CMA factor captures investment behaviour, showing whether Amazon is more conservative or aggressive in its investments. These additional factors provide a comprehensive view of Amazon's risk exposures beyond market, size, and value effects.""")
        
        # Reflective analysis
        st.write("### Reflective Analysis")
        st.write("""1. **How do the results from the CAPM differ from the Fama-French models, and what could explain these differences?**\n- The CAPM provides a single beta representing the sensitivity of Amazon's returns to the overall market. In contrast, the Fama-French models provide additional factors (SMB, HML, RMW, CMA) that capture different dimensions of risk, such as size, value, profitability, and investment patterns. Differences in results can be attributed to these additional factors, which offer a more nuanced view of Amazon's risk profile.""")
        
        st.write("""2. **Which model provides the most comprehensive view of Amazon's risk exposures, in your opinion? Justify your answer.**\n- The Fama-French 5-Factor Model provides the most comprehensive view of Amazon's risk exposures, as it incorporates multiple dimensions of risk beyond the market. This allows for a deeper understanding of the factors affecting Amazon's performance, including profitability and investment patterns. However, it is also important to recognise that more factors introduce more complexity and potential overfitting, which should be considered in the analysis.""")
        
        # Critical understanding and limitations of factor models
        st.write("### Critical Understanding Factor Models and their Limitations")
        st.write("""- While factor models such as CAPM and Fama-French provide valuable insights into the risk-return relationship, they are not without limitations. The assumptions behind these models, such as market efficiency and constant risk premiums, may not always hold in reality. Additionally, the choice of factors and the availability of quality data can significantly impact the results and their interpretation. It is crucial to understand these limitations when applying factor models in practice.""")


if __name__ == "__main__":
    main()