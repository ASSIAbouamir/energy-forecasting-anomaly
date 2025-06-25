# src/train_forecast.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_loader import load_raw, preprocess_hourly
from forecasting_model import build_lstm, get_lstm_callbacks, build_prophet

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RAW_PATH    = 'C:/Users/Hp/Desktop/energie/data/household_power_consumption.txt'
# Format HDF5 pour le ModelCheckpoint
CHECKPOINT  = 'C:/Users/Hp/Desktop/energie/checkpoints/lstm_best.h5'
LOOKBACK    = 48     # fenêtre glissante en heures
BATCH_SIZE  = 64
EPOCHS      = 30
TEST_RATIO  = 0.15
VAL_RATIO   = 0.1765  # pour que val ~ 15% du total
# ────────────────────────────────────────────────────────────────────────────────

def create_sequences(data: pd.DataFrame, features: list, target: str, lookback: int):
    X, y = [], []
    arr = data[features + [target]].values
    for i in range(lookback, len(arr)):
        X.append(arr[i-lookback:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)

def plot_history(history, save_path=None):
    loss = history.history['loss']
    val  = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val,  label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    os.makedirs('checkpoints', exist_ok=True)

    # 1️⃣ Chargement & pré-processing
    df_raw = load_raw(RAW_PATH)
    df     = preprocess_hourly(df_raw)
    print(f"Data ready: {df.shape[0]} hourly samples, columns: {df.columns.tolist()}")

    # 2️⃣ Définir target et features
    TARGET   = 'Global_active_power'
    FEATURES = [c for c in df.columns if c not in ['datetime', TARGET]]
    print("Features used:", FEATURES)

    # 3️⃣ Split train / val / test (sans shuffling)
    df_trainval, df_test = train_test_split(df,  test_size=TEST_RATIO,  shuffle=False)
    df_train,    df_val  = train_test_split(df_trainval, test_size=VAL_RATIO, shuffle=False)
    print(f"Train: {df_train.shape[0]}  Val: {df_val.shape[0]}  Test: {df_test.shape[0]}")

    # 4️⃣ Création des séquences pour LSTM
    X_train, y_train = create_sequences(df_train, FEATURES, TARGET, LOOKBACK)
    X_val,   y_val   = create_sequences(df_val,   FEATURES, TARGET, LOOKBACK)
    X_test,  y_test  = create_sequences(df_test,  FEATURES, TARGET, LOOKBACK)
    print(f"Shapes → X_train:{X_train.shape}, y_train:{y_train.shape}")

    # 5️⃣ Construction et entraînement du LSTM
    model   = build_lstm(input_shape=(LOOKBACK, len(FEATURES)))
    cbs     = get_lstm_callbacks(checkpoint_path=CHECKPOINT)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    # 6️⃣ Visualisation des pertes
    plot_history(history, save_path='C:/Users/Hp/Desktop/energie/checkpoints/loss_curve.png')

    # 7️⃣ Évaluation sur le test set
    y_pred = model.predict(X_test).flatten()
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    print(f"\nLSTM Test RMSE: {rmse:.4f} kW  |  MAE: {mae:.4f} kW")

    # 8️⃣ Évaluation rapide de Prophet
    df_prophet = df[['datetime', TARGET]].rename(columns={'datetime':'ds','Global_active_power':'y'})
    m = build_prophet(country_holidays='FR')
    cutoff = int(len(df_prophet)*(1-TEST_RATIO))
    m.fit(df_prophet.iloc[:cutoff])
    future = m.make_future_dataframe(periods=LOOKBACK, freq='H')
    fcst   = m.predict(future)
    actual = df_prophet['y'].values[-LOOKBACK:]
    pred   = fcst['yhat'].values[-LOOKBACK:]
    rmse_p = mean_squared_error(actual, pred, squared=False)
    mae_p  = mean_absolute_error(actual, pred)
    print(f"Prophet Last {LOOKBACK}h RMSE: {rmse_p:.4f} kW  |  MAE: {mae_p:.4f} kW")

if __name__ == "__main__":
    main()
