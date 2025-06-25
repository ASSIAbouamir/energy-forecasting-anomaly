# src/forecasting_model.py

from typing import TYPE_CHECKING, Optional, Tuple, List

# Pour que Pylance connaisse ProphetType au check-time, sans l’importer à runtime
if TYPE_CHECKING:
    from prophet import Prophet as ProphetType
    import tensorflow as tf
    from tensorflow.keras.callbacks import Callback as KerasCallback
    from tensorflow.keras.models import Model as KerasModel

# —— Option LSTM (TensorFlow) ——
try:
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks, optimizers
except ImportError:
    tf = None

def build_lstm(
    input_shape: Tuple[int, int],
    lstm_units: List[int] = [64, 32],
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> "KerasModel":
    """
    Construit un modèle LSTM configurable pour la prévision.

    Returns:
      Un tf.keras.Model compilé.
    """
    if tf is None:
        raise ImportError("TensorFlow non installé. pip install tensorflow")

    inputs = layers.Input(shape=input_shape)
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(
            units,
            return_sequences=(i < len(lstm_units) - 1),
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )(x)

    outputs = layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def get_lstm_callbacks(
    checkpoint_path: str = 'checkpoints/best_model.h5',
    patience: int = 5,
    monitor: str = 'val_loss'
) -> List["KerasCallback"]:
    """
    Retourne les callbacks pour l'entraînement LSTM.
    """
    if tf is None:
        return []

    return [
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor=monitor),
        callbacks.EarlyStopping(patience=patience, monitor=monitor, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-6)
    ]


# —— Option Prophet ——
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
except ImportError:
    Prophet = None

def build_prophet(
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    seasonality_mode: str = 'additive',
    country_holidays: Optional[str] = None
) -> "ProphetType":
    """
    Construit un modèle Prophet configurable.
    """
    if Prophet is None:
        raise ImportError("Prophet non installé. pip install prophet")

    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode
    )
    if country_holidays:
        m.add_country_holidays(country_name=country_holidays)
    return m

def save_prophet_model(model: "ProphetType", filepath: str) -> None:
    """Sauvegarde le modèle Prophet au format JSON."""
    with open(filepath, 'w') as f:
        f.write(model_to_json(model))

def load_prophet_model(filepath: str) -> "ProphetType":
    """Charge un modèle Prophet depuis un fichier JSON."""
    with open(filepath, 'r') as f:
        return model_from_json(f.read())


# —— Test rapide ——
if __name__ == "__main__":
    print("=== Test LSTM ===")
    if tf:
        m = build_lstm((24, 6))
        m.summary()
        cbs = get_lstm_callbacks()
        print("Callbacks:", cbs)
    else:
        print("Skip LSTM: TensorFlow non disponible.")

    print("\n=== Test Prophet ===")
    if Prophet:
        pm = build_prophet(country_holidays='FR')
        print("Prophet instance:", pm)
    else:
        print("Skip Prophet: librairie non disponible.")
