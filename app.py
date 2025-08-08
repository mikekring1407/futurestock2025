import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from pycaret.time_series import setup, compare_models, plot_model, pull
from darts import TimeSeries
from darts.models import LSTM
from transformers import pipeline
import matplotlib.pyplot as plt
import logging
import os

# Suppress verbose logging from libraries to keep the output clean
logging.getLogger("pycaret").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


# --- App Configuration ---
st.set_page_config(
    page_title="Multi-Model Stock Forecaster",
    page_icon="📈",
    layout="wide"
)

# --- Caching Functions for Performance ---
# Caching ensures that slow functions are not re-run every time a widget is changed.

@st.cache_data
def load_data(ticker):
    """Downloads stock data from Yahoo Finance."""
    data = yf.download(ticker, start='2020-01-01', end='2024-12-31')
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

@st.cache_resource
def get_sentiment_pipeline():
    """Loads the FinBERT sentiment analysis model from Hugging Face."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --- Main App ---
st.title("📈 Multi-Model Stock Forecaster")
st.write("Analyze stock trends using Prophet, PyCaret (XGBoost, etc.), and Darts (LSTM).")

# --- Sidebar for User Input ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")
forecast_days = st.sidebar.slider("Days to Forecast", 30, 365, 90)
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

# --- Main Analysis Area ---
if analyze_button:
    # --- 1. Data Loading and Preparation ---
    with st.spinner(f"Downloading data for {ticker}..."):
        data = load_data(ticker)
    
    if data is None:
        st.error(f"Could not download data for {ticker}. Please check the ticker symbol.")
    else:
        st.success(f"Data for {ticker} downloaded successfully!")
        st.dataframe(data.tail())

        # --- 2. Sentiment Analysis (using sample data) ---
        with st.spinner("Performing sentiment analysis..."):
            sentiment_pipeline = get_sentiment_pipeline()
            # Sample news for demonstration
            news_data = {
                'Date': pd.to_datetime(['2023-05-15', '2023-08-10', '2024-01-20']),
                'headline': [
                    "Market surges as inflation fears ease.",
                    "New government regulations could negatively impact major industries.",
                    "Company reports record profits, beating all expectations."
                ]
            }
            news_df = pd.DataFrame(news_data)
            sentiment = news_df['headline'].apply(lambda x: sentiment_pipeline(x)[0])
            news_df['sentiment_score'] = sentiment.apply(lambda x: x['score'] * (1 if x['label'] == 'positive' else -1))
            
            # Merge sentiment with price data
            data['Date'] = pd.to_datetime(data['Date'])
            data = pd.merge(data, news_df[['Date', 'sentiment_score']], on='Date', how='left').ffill().bfill()
            st.write("Sentiment scores merged with price data.")

        # --- 3. Create Tabs for Different Models ---
        prophet_tab, pycaret_tab, darts_tab = st.tabs(["Prophet Forecast", "PyCaret Model Comparison", "Darts (LSTM) Forecast"])

        # --- Prophet Tab ---
        with prophet_tab:
            st.header("Prophet Time-Series Forecast")
            with st.spinner("Training Prophet model and making predictions..."):
                df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                m = Prophet()
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)
                
                st.write("Forecast Data:")
                st.dataframe(forecast.tail())
                
                st.write("Interactive Forecast Plot:")
                fig = plot_plotly(m, forecast)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Forecast Components:")
                fig2 = m.plot_components(forecast)
                st.pyplot(fig2)

        # --- PyCaret Tab ---
        with pycaret_tab:
            st.header("PyCaret Model Comparison (XGBoost, LightGBM, etc.)")
            with st.spinner("Setting up PyCaret and comparing models... This may take a few minutes."):
                df_pycaret = data[['Date', 'Close', 'sentiment_score']].set_index('Date')
                
                setup(data=df_pycaret, target='Close', fh=forecast_days, session_id=123, use_gpu=False, verbose=False)
                
                best_model = compare_models(sort='MAPE')
                
                st.write("Model Comparison Leaderboard (sorted by MAPE):")
                st.dataframe(pull())
                
                st.write("Plotting the best model's forecast...")
                # The plot_model function saves the file locally, so we need to check for its existence
                plot_file_path = 'Forecast.png'
                if os.path.exists(plot_file_path):
                    os.remove(plot_file_path) # Remove old plot if it exists
                plot_model(best_model, plot='forecast', data_kwargs={'fh': forecast_days}, save=True)
                st.image(plot_file_path)

        # --- Darts Tab ---
        with darts_tab:
            st.header("Darts LSTM Deep Learning Forecast")
            with st.spinner("Training LSTM model... This can be slow."):
                series = TimeSeries.from_dataframe(data, 'Date', 'Close', fill_missing_dates=True, freq='B')
                
                train, val = series[:-forecast_days], series[-forecast_days:]
                
                model_lstm = LSTM(input_chunk_length=30, output_chunk_length=1, n_epochs=50, random_state=42)
                model_lstm.fit(train)
                
                prediction = model_lstm.predict(n=forecast_days)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                series.plot(label='Actual', ax=ax)
                prediction.plot(label='Forecast', ax=ax)
                plt.legend()
                st.pyplot(fig)

        st.success("Analysis Complete!")
        st.balloons()

# --- Disclaimer ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer:** This is an educational tool and should not be used for real financial decisions. "
    "Stock market prediction is inherently complex and risky."
)
