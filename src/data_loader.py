# src/data_loader.py

import pandas as pd

def load_raw(path: str = 'C:/Users/Hp/Desktop/energie/data/household_power_consumption.txt') -> pd.DataFrame:
    """
    Charge le fichier brut et retourne un DataFrame pandas avec :
      - datetime : fusion de Date + Time
      - colonnes converties en float, valeurs manquantes en NaN
    """
    df = pd.read_csv(
        path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        dayfirst=True,
        na_values='?',
        low_memory=False
    )
    return df

def preprocess_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    À partir d'un DataFrame minute par minute :
      1. Mettre 'datetime' en index
      2. Interpoler linéairement les NaN dans le temps
      3. Resampler à l'heure (moyenne)
      4. Ré-interpoler les éventuels NaN dus au resample
      5. Ajouter des features temporelles : hour, weekday, month, is_weekend
    """
    # 1. Index temporel
    df_ts = df.set_index('datetime')

    # 2. Interpolation dans le temps
    df_interp = df_ts.interpolate(method='time')  

    # 3. Resampling horaire (moyenne)
    df_hourly = df_interp.resample('H').mean()

    # 4. Ré-interpolation post-resample si nécessaire
    df_hourly = df_hourly.interpolate(method='time')

    # 5. Remettre 'datetime' en colonne et feature engineering
    df_hourly = df_hourly.reset_index()
    df_hourly['hour']       = df_hourly['datetime'].dt.hour
    df_hourly['weekday']    = df_hourly['datetime'].dt.weekday
    df_hourly['month']      = df_hourly['datetime'].dt.month
    df_hourly['is_weekend'] = df_hourly['weekday'].isin([5, 6]).astype(int)

    return df_hourly

if __name__ == "__main__":
    # Petit test
    df_raw = load_raw()
    print("Raw data shape:", df_raw.shape)
    df_hourly = preprocess_hourly(df_raw)
    print("Hourly data shape:", df_hourly.shape)
    print(df_hourly.head())
