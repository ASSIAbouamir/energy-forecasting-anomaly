# src/detect_anomalies.py

import os
import matplotlib.pyplot as plt

from data_loader import load_raw, preprocess_hourly
from anomaly_detector import AnomalyDetector

def main():
    os.makedirs('anomaly_results', exist_ok=True)

    # 1. Charger et préparer les données
    df_raw = load_raw('C:/Users/Hp/Desktop/energie/data/household_power_consumption.txt')
    df     = preprocess_hourly(df_raw)

    # 2. Choisir les colonnes pour la détection d'anomalies
    FEATURES = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity'
    ]

    # 3. Entraîner le détecteur
    detector = AnomalyDetector(contamination=0.01)
    detector.fit(df, FEATURES)

    # 4. Prédire les anomalies
    df_anom = detector.predict(df, FEATURES)
    print("Total anomalies détectées:", df_anom['anomaly'].sum())

    # 5. Visualisation
    normal = df_anom[df_anom['anomaly'] == 0]
    anoms  = df_anom[df_anom['anomaly'] == 1]

    plt.figure(figsize=(12,4))
    plt.plot(normal['datetime'], normal['Global_active_power'], label='Normal', lw=0.8)
    plt.scatter(anoms['datetime'], anoms['Global_active_power'],
                color='red', s=15, label='Anomaly')
    plt.legend()
    plt.title("Anomaly Detection on Global Active Power")
    plt.xlabel("Datetime")
    plt.ylabel("Global Active Power (kW)")
    plt.tight_layout()
    plt.savefig('anomaly_results/anomalies_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
