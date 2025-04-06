import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des modèles
with open("mobile_KNN.pkl", "rb") as f:
    knn_model, knn_features = pickle.load(f)

with open("logistic_model.pkl", "rb") as f:
    loaded_log = pickle.load(f)

if isinstance(loaded_log, tuple):
    logistic_model, logistic_features = loaded_log
else:
    logistic_model = loaded_log
    logistic_features = ["battery_power", "ram", "px_height", "px_width", "int_memory", "mobile_wt"]

# Titre principal
st.title("📱 Mobile Price Prediction App")

# Choix du modèle
model_choice = st.selectbox("Choisissez un modèle :", ["KNN", "Régression Logistique"])
model = knn_model if model_choice == "KNN" else logistic_model
expected_features = knn_features if model_choice == "KNN" else logistic_features

# Section : Prédiction individuelle
st.header("🧮 Prédiction Individuelle")
battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1000)
ram = st.slider("RAM (en MB)", 256, 4096, 2048)
px_height = st.slider("Pixel Height", 0, 1960, 1000)
px_width = st.slider("Pixel Width", 0, 2000, 1000)
int_memory = st.slider("Mémoire interne (en GB)", 2, 128, 32)
mobile_wt = st.slider("Poids du téléphone (g)", 80, 250, 150)

user_input = pd.DataFrame([[battery_power, ram, px_height, px_width, int_memory, mobile_wt]],
                          columns=["battery_power", "ram", "px_height", "px_width", "int_memory", "mobile_wt"])

for col in expected_features:
    if col not in user_input.columns:
        user_input[col] = 0
user_input = user_input[expected_features]

if st.button("Prédire le prix individuel"):
    prediction = model.predict(user_input)
    st.success(f"💡 Classe de prix prédite : {prediction[0]}")

# Section : Upload CSV
st.header("📂 Prédictions à partir d’un fichier CSV")

uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")
if uploaded_file is not None:
    # Charger le fichier CSV
    csv_data = pd.read_csv(uploaded_file)
    
    # Vérifier si les colonnes attendues sont présentes dans les données
    for col in expected_features:
        if col not in csv_data.columns:
            csv_data[col] = 0  # Ajouter des colonnes manquantes avec 0
    
    # Sélectionner les colonnes nécessaires
    csv_data = csv_data[expected_features]
    
    # Prédiction sur les données du fichier CSV
    predictions = model.predict(csv_data)
    csv_data["Classe de prix prédite"] = predictions
    
    # Afficher les résultats des prédictions
    st.write("Prédictions basées sur les données importées :")
    st.write(csv_data)
