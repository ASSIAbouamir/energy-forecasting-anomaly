# ⚡ Energy Forecasting & Anomaly Detection in Household Power Consumption

**Auteur : Assia Bouamir**  
_Projet personnel d'intelligence artificielle_

## 📌 Objectif

Ce projet a pour but de construire un pipeline complet pour :

- Prévoir la **consommation électrique horaire** d’un logement à court terme (1 à 72 heures).
- Détecter automatiquement les **consommations anormales** (pics ou dysfonctionnements).

## 🗃️ Données

Les données proviennent de l’ensemble **Individual Household Electric Power Consumption** (UCI ML Repository) :
- Environ 2 millions de mesures à la **minute** (2006–2010)
- Colonnes utilisées : `Global_active_power`, `Voltage`, `Global_reactive_power`, `Global_intensity`, `Sub_metering_1/2/3`

## ⚙️ Étapes du pipeline

### 1. 🔧 Préparation des données

- Lecture du fichier `.txt` avec parsing `Date + Time`
- Interpolation & **resampling horaire**
- Feature engineering : `hour`, `weekday`, `is_weekend`, etc.
- Résultat : **34 589 enregistrements horaires** prêts à l'emploi

### 2. 📈 Prévision de la consommation

#### 🧠 Modèle LSTM (TensorFlow)
- Architecture : 2 couches LSTM + Dense
- Entraînement avec `EarlyStopping`, `ReduceLROnPlateau`
- RMSE : **0.50 kW** sur 48 h de prévision

#### 📊 Benchmark : Prophet
- Modèle additif avec saisonnalités
- RMSE : **0.73 kW** sur 48 h

✅ **LSTM surpasse Prophet** grâce à sa capacité à capturer les motifs journaliers.

### 3. 🚨 Détection d’anomalies

- Utilisation de `IsolationForest` (via PyOD)
- Sur les features : `Global_active_power`, `Voltage`, etc.
- Affichage interactif des **pics suspects** via une interface

### 4. 🌐 Application interactive (Streamlit)

Une interface simple et interactive permet :
- Visualisation des séries historiques
- Prévisions ajustables (1 à 72 h)
- Détection d’anomalies avec un slider pour le taux de détection

📷 _Exemples d’écrans disponibles dans le repo_

## 🔭 Améliorations futures

- Ajouter des variables exogènes : météo, température
- Hyperparameter tuning (Keras Tuner, Bayesian search)
- Test d'autres modèles d’anomalie (AutoEncoder, LOF, etc.)
- Déploiement via FastAPI + Docker

## 📚 Références

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- [TensorFlow LSTM Guide](https://www.tensorflow.org/tutorials)
- [Facebook Prophet](https://facebook.github.io/prophet/)
- [PyOD Documentation](https://pyod.readthedocs.io/en/latest/)

---

🧠 _N'hésitez pas à me contacter si vous souhaitez en savoir plus ou discuter IA !_

