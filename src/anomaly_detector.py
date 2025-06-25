# src/anomaly_detector.py

import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """
    Détecteur d'anomalies basé sur IsolationForest (pyod).
    """

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Args:
          contamination: fraction estimée d’anomalies dans les données
          random_state: graine pour la reproductibilité
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame, features: list):
        """
        Entraîne l'IsolationForest sur les features fournies.

        Args:
          df: DataFrame pré-processed (par ex. df_hourly)
          features: liste de colonnes numériques à utiliser
        """
        # 1) Standardisation
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[features])

        # 2) IsolationForest
        self.model = IForest(contamination=self.contamination,
                             random_state=self.random_state)
        self.model.fit(X)

    def predict(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Prédit les anomalies sur un nouveau jeu de données.

        Args:
          df: DataFrame à tester
          features: mêmes features que pour fit()

        Returns:
          Copie du DataFrame avec une colonne 'anomaly' (1 = anomalie, 0 = normal)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Appeler d'abord `fit()` avant `predict()`")

        X = self.scaler.transform(df[features])
        # pyod IForest.predict: 0 = inlier, 1 = outlier
        df_out = df.copy()
        df_out['anomaly'] = self.model.predict(X)
        return df_out

    def decision_scores(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Retourne les scores de décision (plus petit = plus anormal).

        Args:
          df: DataFrame à tester
          features: mêmes features que pour fit()

        Returns:
          Copie du DataFrame avec une colonne 'anomaly_score'
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Appeler d'abord `fit()` avant `decision_scores()`")

        X = self.scaler.transform(df[features])
        scores = self.model.decision_function(X)
        df_out = df.copy()
        df_out['anomaly_score'] = scores
        return df_out
