# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from data_loader import load_raw, preprocess_hourly
from anomaly_detector import AnomalyDetector

# ─── CONFIG ──────────────────────────────────────────────
RAW_PATH   = 'C:/Users/Hp/Desktop/energie/data/household_power_consumption.txt'
MODEL_PATH = 'C:/Users/Hp/Desktop/energie/checkpoints/lstm_best.h5'
LOOKBACK   = 48
FEATURES   = [
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'hour',
    'weekday',
    'month',
    'is_weekend'
]
TARGET     = 'Global_active_power'
CONTAM     = 0.01
# ──────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df_raw = load_raw(RAW_PATH)
    df     = preprocess_hourly(df_raw)
    return df

@st.cache_resource
def load_models():
    lstm = load_model(MODEL_PATH)
    det  = AnomalyDetector(contamination=CONTAM)
    return lstm, det

# ─── STREAMLIT UI ─────────────────────────────────────────
st.set_page_config(page_title="Energy Forecast & Anomalies", layout="wide")
st.title("⚡ Energy Forecasting & Anomaly Detection")

# Load data & models
df = load_data()
lstm, detector = load_models()

# Sidebar controls
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 72, 24)

st.sidebar.header("Anomaly Detection")
contam = st.sidebar.slider("Contamination rate", 0.001, 0.05, CONTAM, step=0.001)
detector.contamination = contam

# 1) Display historical series
st.subheader("Historic Global Active Power")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df['datetime'], df[TARGET], color='blue', lw=0.6)
ax.set_xlabel("Datetime")
ax.set_ylabel("kW")
st.pyplot(fig)

# 2) Detect anomalies
st.subheader("Detected Anomalies")
detector.fit(df, [TARGET])
df_anom = detector.predict(df, [TARGET])

fig2, ax2 = plt.subplots(figsize=(10, 3))
normal = df_anom[df_anom['anomaly'] == 0]
anoms  = df_anom[df_anom['anomaly'] == 1]
ax2.plot(normal['datetime'], normal[TARGET], color='blue', lw=0.6, label='Normal')
ax2.scatter(anoms['datetime'], anoms[TARGET], color='red', s=15, label='Anomaly')
ax2.set_xlabel("Datetime")
ax2.set_ylabel("kW")
ax2.legend()
st.pyplot(fig2)

# 3) Forecast next hours (auto-regressive)
st.subheader(f"Forecast Next {horizon} Hours")

# Prepare rolling history of dates and features (without target)
history_dates = df['datetime'].tail(LOOKBACK).tolist()
history_feats = df[FEATURES].tail(LOOKBACK).values.tolist()

forecasts = []
for _ in range(horizon):
    # build input array (1, LOOKBACK, n_features)
    X_input = np.array(history_feats[-LOOKBACK:])[np.newaxis, ...]
    # predict next value
    y_pred = lstm.predict(X_input, verbose=0)[0, 0]
    forecasts.append(y_pred)

    # compute next timestamp
    next_time = history_dates[-1] + pd.Timedelta(hours=1)
    last_feats = history_feats[-1]
    # build next feature vector
    new_feats = [
        last_feats[0],  # Global_reactive_power
        last_feats[1],  # Voltage
        last_feats[2],  # Global_intensity
        last_feats[3],  # Sub_metering_1
        last_feats[4],  # Sub_metering_2
        last_feats[5],  # Sub_metering_3
        next_time.hour,
        next_time.weekday(),
        next_time.month,
        int(next_time.weekday() >= 5)
    ]
    # append for next iteration
    history_dates.append(next_time)
    history_feats.append(new_feats)

# Build forecast DataFrame
future_idx = pd.to_datetime(history_dates[-horizon:])
df_fcst = pd.DataFrame({
    'datetime': future_idx,
    'forecast': forecasts
})

# Plot forecast versus last history
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(df['datetime'].tail(200), df[TARGET].tail(200), label='History', lw=0.8)
ax3.plot(df_fcst['datetime'], df_fcst['forecast'], label='Forecast', lw=2, color='orange')
ax3.set_xlabel("Datetime")
ax3.set_ylabel("kW")
ax3.legend()
st.pyplot(fig3)
# ─── END OF STREAMLIT APP ───────────────────────────────