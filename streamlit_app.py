import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Financial Forecasting App", layout="wide")

st.title("📈 Financial Forecasting Dashboard")
st.markdown("Multivariate LSTM Forecasting + GARCH Risk Estimation")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_nse_data.csv")
    df = df.dropna().reset_index(drop=True)
    df['Returns'] = df['Returns'].fillna(0)
    df, minmax_scaler, standard_scaler = preprocess_data(df)
    return df, minmax_scaler, standard_scaler

df, minmax_scaler, standard_scaler = load_data()

tab1, tab2 = st.tabs(["📊 LSTM Forecasting", "⚠️ Risk Forecasting (GARCH)"])

with tab1:
    st.subheader("Train LSTM on NSE Data")
    features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
    X, y = create_sequences(df[features], target_col='Close')

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=16,
              validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)

    st.success("✅ Model trained successfully!")

    preds = model.predict(X_test).flatten()
    st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})

with tab2:
    st.subheader("Risk Forecasting Using GARCH")
    vol_forecast, var_1d = forecast_garch_var(df)
    st.metric(label="📉 1-Day 95% Value at Risk (VaR)", value=f"{var_1d:.2f}%")
    st.line_chart(vol_forecast.values)
