import streamlit as st
import pandas as pd
import pickle

# Chargement du modèle KNN avec les features
with open("mobile_KNN.pkl", "rb") as f:
    knn_model, knn_features = pickle.load(f)

# Chargement du modèle de régression logistique
with open("logistic_model.pkl", "rb") as f:
    loaded_log = pickle.load(f)

# Vérifier si c’est un tuple (modèle, features) ou juste le modèle
if isinstance(loaded_log, tuple):
    logistic_model, logistic_features = loaded_log
else:
    logistic_model = loaded_log
    logistic_features = ["battery_power", "ram", "px_height", "px_width", "int_memory", "mobile_wt"]  # à ajuster selon ton entraînement

# Titre de l'application
st.title("Mobile Price Prediction App 📱💰")

# Choix du modèle
model_choice = st.selectbox("Choisissez un modèle :", ["KNN", "Régression Logistique"])

# Entrée des données utilisateur
st.subheader("Entrez les caractéristiques du téléphone :")

battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1000)
ram = st.slider("RAM (en MB)", 256, 4096, 2048)
px_height = st.slider("Pixel Height", 0, 1960, 1000)
px_width = st.slider("Pixel Width", 0, 2000, 1000)
int_memory = st.slider("Mémoire interne (en GB)", 2, 128, 32)
mobile_wt = st.slider("Poids du téléphone (g)", 80, 250, 150)

# Convertir les données en DataFrame
user_input = pd.DataFrame([[battery_power, ram, px_height, px_width, int_memory, mobile_wt]],
                          columns=["battery_power", "ram", "px_height", "px_width", "int_memory", "mobile_wt"])

# Sélection des features et du modèle
if model_choice == "KNN":
    expected_features = knn_features
    model = knn_model
else:
    expected_features = logistic_features
    model = logistic_model

# Ajouter les colonnes manquantes avec 0
for col in expected_features:
    if col not in user_input.columns:
        user_input[col] = 0

# Réorganiser les colonnes dans le bon ordre
user_input = user_input[expected_features]

# Prédiction
if st.button("Prédire le prix"):
    prediction = model.predict(user_input)
    st.success(f"Classe de prix prédite : {prediction[0]}")
