import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from pycaret.time_series import setup, compare_models, plot_model, predict_model
from darts import TimeSeries
from darts.models import LSTM
from transformers import pipeline
import matplotlib.pyplot as plt

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
        data = load_d
