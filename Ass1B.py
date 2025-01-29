import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from eodhdHelper import get_client

ticker_to_company = {
    # Technology (Top 5 by market cap)
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',

    # Travel & Hospitality (Major players)
    'BKNG': 'Booking Holdings Inc.',
    'ABNB': 'Airbnb Inc.',
    'AAL': 'American Airlines Group Inc.',
    'DAL': 'Delta Air Lines Inc.',
    'MAR': 'Marriott International Inc.',

    # Leisure & Entertainment (Key companies)
    'DIS': 'The Walt Disney Company',
    'RCL': 'Royal Caribbean Cruises Ltd.',
    'CCL': 'Carnival Corporation & plc',
    'MGM': 'MGM Resorts International',
    'DKNG': 'DraftKings Inc.',

    # Financial Services (Largest US banks and payment processors)
    'JPM': 'JPMorgan Chase & Co.',
    'BAC': 'Bank of America Corporation',
    'GS': 'Goldman Sachs Group Inc.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Incorporated',

    # Healthcare (Major pharmaceutical and healthcare companies)
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    'MRNA': 'Moderna Inc.',
    'UNH': 'UnitedHealth Group Inc.',
    'CVS': 'CVS Health Corporation',
    
    # Retail (Largest US retailers)
    'WMT': 'Walmart Inc.',
    'COST': 'Costco Wholesale Corporation',
    'HD': 'The Home Depot Inc.',
    'TGT': 'Target Corporation',
    'LOW': 'Lowe\'s Companies Inc.'
}

def main():
    # Set up the title and description
    st.write("### Stock PCA Analysis")
    
    # Define lists of stock tickers organised by sector
    technology_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    travel_stocks = ['BKNG', 'ABNB', 'AAL', 'DAL', 'MAR']
    leisure_stocks = ['DIS', 'RCL', 'CCL', 'MGM', 'DKNG']
    financial_stocks = ['JPM', 'BAC', 'GS', 'V', 'MA']
    healthcare_stocks = ['JNJ', 'PFE', 'MRNA', 'UNH', 'CVS']
    retail_stocks = ['WMT', 'COST', 'HD', 'TGT', 'LOW']  

    # Combine all stock lists
    all_stocks = (technology_stocks + travel_stocks + leisure_stocks + 
                 financial_stocks + healthcare_stocks + retail_stocks)

    # Define the date range for the analysis
    start_date = '2019-01-01'
    end_date = '2021-12-31'

    # Sidebar options for user to select which analyses to display
    st.sidebar.markdown("### Available Analyses")
    st.sidebar.markdown("Select the analyses you want to view:")

    # Create checkboxes in the sidebar for each analysis section
    sections = {
        "1. Data Loading and Preprocessing": st.sidebar.checkbox(
            "📊 Load and Process Data",
            help="View the loaded and preprocessed stock price data"
        ),
        "2. Data Normalisation": st.sidebar.checkbox(
            "📈 Normalise Data",
            help="View the normalised stock price data"
        ),
        "3. PCA Application": st.sidebar.checkbox(
            "🔄 Apply PCA",
            help="View PCA results and explained variance"
        ),
        "4. COVID-19 Analysis": st.sidebar.checkbox(
            "🦠 COVID-19 Period Analysis",
            help="Analyse stock/sector contributions during COVID-19"
        ),
        "5. Results Interpretation": st.sidebar.checkbox(
            "📝 Interpret Results",
            help="View interpretation of the PCA results"
        ),
        "6. Portfolio Construction": st.sidebar.checkbox(
            "💼 Portfolio Analysis",
            help="View portfolio construction and performance"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ Tip: You can select multiple analyses to view them all at once.")

    try:
        # Initialise EODHD client
        client = get_client()

        @st.cache_data
        def load_data():
            # Download stock data and handle missing values
            data_frames = []
            for stock in all_stocks:
                try:
                    df = client.download(stock, start=start_date, end=end_date)
                    data_frames.append(df['Adj Close'])
                except Exception as e:
                    st.warning(f"Error loading data for {stock}: {str(e)}")
                    continue
            
            # Combine all data frames
            data = pd.concat(data_frames, axis=1)
            data.columns = all_stocks[:len(data_frames)]
            
            # Handle missing values
            data = data.ffill().bfill()
            return data

        @st.cache_data
        def normalise_data(data):
            scaler = StandardScaler()
            normalised = scaler.fit_transform(data)
            normalised_df = pd.DataFrame(normalised, index=data.index, columns=data.columns)
            return normalised_df

        @st.cache_data
        def apply_pca(normalised_data):
            pca = PCA()
            principal_components = pca.fit_transform(normalised_data)
            pca_columns = [f'PC{i+1}' for i in range(len(pca.explained_variance_))]
            principal_df = pd.DataFrame(principal_components, index=normalised_data.index, columns=pca_columns)
            explained_variance_ratio = pca.explained_variance_ratio_
            return pca, principal_df, explained_variance_ratio

        @st.cache_data
        def covid_analysis(data):
            covid_start_date = '2020-03-01'
            covid_end_date = '2021-12-31'
            covid_data = data.loc[covid_start_date:covid_end_date]
            
            covid_data = covid_data.ffill().bfill()
            
            nan_counts = covid_data.isnull().sum()
            if nan_counts.any():
                st.write("Stocks with NaN values after filling:")
                st.write(nan_counts[nan_counts > 0])
                st.write("Dropping stocks with insufficient data.")
                covid_data = covid_data.drop(columns=nan_counts[nan_counts > 0].index)
            else:
                st.write("No NaN values in COVID-19 data after filling.")
            
            covid_normalised = normalise_data(covid_data)
            covid_pca, covid_principal_df, covid_explained_variance_ratio = apply_pca(covid_normalised)
            
            covid_loadings = pd.DataFrame(
                covid_pca.components_.T,
                index=covid_data.columns,
                columns=[f'PC{i+1}' for i in range(len(covid_pca.explained_variance_))]
            )
            return covid_pca, covid_principal_df, covid_loadings, covid_explained_variance_ratio, covid_data

        # Load the data
        data = load_data()

        # Display sections based on checkbox selections
        if sections["1. Data Loading and Preprocessing"]:
            st.markdown("---")
            st.header('1. Load and preprocess the stock price data')
            st.write('Stock price data from January 2019 to December 2021:')
            st.dataframe(data)

        if sections["2. Data Normalisation"]:
            st.markdown("---")
            st.header('2. Normalise the data')
            normalised_data = normalise_data(data)
            st.write('Normalised data:')
            st.dataframe(normalised_data)

        if sections["3. PCA Application"]:
            st.markdown("---")
            st.header('3. Apply PCA')
            normalised_data = normalise_data(data)
            pca, principal_df, explained_variance_ratio = apply_pca(normalised_data)
            
            st.write('Explained Variance Ratio:')
            fig1, ax1 = plt.subplots()
            ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
            ax1.set_xlabel('Principal Components')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('Explained Variance Ratio by Principal Components')
            st.pyplot(fig1)
            
            cumulative_variance = np.cumsum(explained_variance_ratio)
            st.write('Cumulative Explained Variance:')
            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                    marker='o', linestyle='--')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('Cumulative Explained Variance by Principal Components')
            st.pyplot(fig2)

        if sections["4. COVID-19 Analysis"]:
            st.markdown("---")
            st.header('4. Identify stocks/sectors contribution during COVID-19 period')
            covid_pca, _, covid_loadings, covid_explained_variance_ratio, _ = covid_analysis(data)
            
            st.write('Explained Variance Ratio during COVID-19 period:')
            fig3, ax3 = plt.subplots()
            ax3.bar(range(1, len(covid_explained_variance_ratio) + 1), covid_explained_variance_ratio)
            ax3.set_xlabel('Principal Components')
            ax3.set_ylabel('Explained Variance Ratio')
            ax3.set_title('Explained Variance Ratio during COVID-19 Period')
            st.pyplot(fig3)
            
            st.write('Principal Component Loadings during COVID-19 period:')
            fig4, ax4 = plt.subplots(figsize=(20, 12))
            sns.heatmap(covid_loadings, annot=True, cmap='coolwarm', ax=ax4, fmt='.2f')
            ax4.set_title('Principal Component Loadings during COVID-19 Period')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig4)
            
            st.write('Stocks Contributing Most to Each Principal Component:')
            pcs_per_row = 3
            pcs = covid_loadings.columns.tolist()
            
            for i in range(0, len(pcs), pcs_per_row):
                cols = st.columns(pcs_per_row)
                for j in range(pcs_per_row):
                    if i + j < len(pcs):
                        pc = pcs[i + j]
                        with cols[j]:
                            st.write(f'### {pc}')
                            sorted_loadings = covid_loadings[pc].sort_values(ascending=False)
                            loadings_df = pd.DataFrame({
                                'Ticker': sorted_loadings.index,
                                'Company Name': [ticker_to_company[ticker] for ticker in sorted_loadings.index],
                                'Loading': sorted_loadings.values.round(4)
                            })
                            st.write(loadings_df)

        if sections["5. Results Interpretation"]:
            st.markdown("---")
            st.header('5. Discuss the results and interpret')
            st.write("""
        **Interpretation of PCA Results During COVID-19 Period:**
        
        The Principal Component Analysis (PCA) during the COVID-19 period helps us understand the underlying structure of the stock price movements and identify which stocks had the most significant influence.
        
        - **First Principal Component (PC1):** Our analysis shows that MGM Resorts International, Goldman Sachs Group Inc., Target Corporation, JPMorgan Chase & Co., and Alphabet Inc. had the highest positive loadings on PC1, contributing most to the overall variance during the COVID-19 period. This diverse group of leaders challenges conventional expectations about sector performance during the pandemic.
        
        - **Cross-Sector Resilience:** The distribution of top performers across leisure (MGM), financial services (GS, JPM), retail (TGT), and technology (GOOGL) sectors suggests that company-specific factors were more important than sector membership in determining market influence during this period.
        
        - **Performance Insights:** The PC1 loadings reveal that successful companies during the COVID-19 period came from various sectors, indicating that adaptability and individual company strategies were more crucial than sector-wide advantages.
        
        **Conclusion:**
        
        The PCA results reveal that market leadership during the COVID-19 period was not concentrated in any single sector, but rather distributed across companies that demonstrated strong individual resilience and adaptability. Understanding these company-specific contributions, rather than sector-wide patterns, can help investors make more informed decisions about portfolio diversification and risk management.
        """)

        if sections["6. Portfolio Construction"]:
            st.markdown("---")
            st.header('6. Portfolio Construction')
            covid_pca, _, covid_loadings, _, covid_data = covid_analysis(data)
            # 1. Identify top 5 stocks with highest loadings on PC1 (assumed resilience/growth)

            pc1_loadings = covid_loadings['PC1']
            top_stocks = pc1_loadings.sort_values(ascending=False).head(5)
            
            top_stocks_df = pd.DataFrame({
                'Ticker': top_stocks.index,
                'Company Name': [ticker_to_company[ticker] for ticker in top_stocks.index],
                'PC1 Loading': top_stocks.values.round(4)
            })
            st.subheader('Top 5 Stocks Exhibiting Resilience/Growth During COVID-19 (PC1 Loadings)')
            st.write(top_stocks_df)
            
            # 2. Construct an equally-weighted portfolio from the top stocks
            portfolio_stocks = top_stocks.index.tolist()
            st.subheader('Constructed Portfolio Based on PCA Analysis')
            st.write(portfolio_stocks)
            
            portfolio_stocks = top_stocks.index.tolist()
            portfolio_data = covid_data[portfolio_stocks]
            portfolio_returns = portfolio_data.pct_change().dropna()
            portfolio_cum_returns = (1 + portfolio_returns.mean(axis=1)).cumprod()
            
            try:
                # Retrieve S&P 500 data for comparison
                sp500_data = client.download('^GSPC', start='2020-03-01', end='2021-12-31')
                sp500_returns = sp500_data['Adj Close'].pct_change().dropna()
                sp500_cum_returns = (1 + sp500_returns).cumprod()
                
                # 3. Compare cumulative returns between the portfolio and S&P 500
                st.subheader('Cumulative Returns Comparison')
                fig, ax = plt.subplots()
                portfolio_cum_returns.plot(ax=ax, label='PCA-Based Portfolio')
                sp500_cum_returns.plot(ax=ax, label='S&P 500')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Returns')
                ax.set_title('Cumulative Returns: PCA-Based Portfolio vs. S&P 500')
                ax.legend()
                st.pyplot(fig)
                
                # Calculate performance metrics
                st.subheader('Performance Metrics')
                
                # Total returns
                portfolio_total_return = portfolio_cum_returns.iloc[-1] - 1
                sp500_total_return = sp500_cum_returns.iloc[-1] - 1
                portfolio_volatility = portfolio_returns.std().mean() * np.sqrt(252)
                sp500_volatility = sp500_returns.std() * np.sqrt(252)
                portfolio_drawdown = (portfolio_cum_returns / portfolio_cum_returns.cummax() - 1).min()
                sp500_drawdown = (sp500_cum_returns / sp500_cum_returns.cummax() - 1).min()
                
                metrics = pd.DataFrame({
                    'Metric': ['Total Return', 'Annualised Volatility', 'Maximum Drawdown'],
                    'PCA-Based Portfolio': [
                        f"{portfolio_total_return:.2%}",
                        f"{portfolio_volatility:.2%}",
                        f"{portfolio_drawdown:.2%}"
                    ],
                    'S&P 500': [
                        f"{sp500_total_return:.2%}",
                        f"{sp500_volatility:.2%}",
                        f"{sp500_drawdown:.2%}"
                    ]
                })
                
                st.table(metrics)
                
                st.subheader('Individual Stock Contributions to Portfolio Returns')
                fig, ax = plt.subplots(figsize=(10, 6))
                portfolio_returns.plot(ax=ax)
                plt.title('Daily Returns of Portfolio Stocks')
                plt.xlabel('Date')
                plt.ylabel('Daily Returns')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in S&P 500 comparison: {str(e)}")
                st.write("Unable to load S&P 500 data for comparison. Showing portfolio performance only.")
                
                st.subheader('Portfolio Performance')
                fig, ax = plt.subplots()
                portfolio_cum_returns.plot(ax=ax, label='PCA-Based Portfolio')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Returns')
                ax.set_title('Cumulative Returns of PCA-Based Portfolio')
                ax.legend()
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure your EODHD API key is configured correctly and you have access to the required data.")
        return

if __name__ == "__main__":
    main()