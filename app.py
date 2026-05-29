import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_FILE = os.path.join(os.path.dirname(__file__), "lyon_full_history.csv")

st.set_page_config(page_title="Lyon ML Miner", page_icon=None, layout="wide")

# --- Custom Style, for blue panel color  ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #26619C;
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #26619C !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Lyon Data Mining & Prediction")

if not os.path.exists(DATA_FILE):
    st.warning("Aucune donnée trouvée")
    st.stop()


@st.cache_data
def load_and_mine():
    df = pd.read_csv(DATA_FILE)
    df['time'] = pd.to_datetime(df['time']) # On convertit en format date, important!
    df['target_aqi'] = (df['pm10'] * 0.5 + df['pm2_5'] * 1.5 + df['nitrogen_dioxide'] * 0.2)#calcul du target_aqi
    return df

df = load_and_mine()

# Entraînement du modèle de Linear Regression simple 
@st.cache_resource
def train_professional_model(data):
    X = data[['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']].values
    y = data['target_aqi'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_professional_model(df)

# --- TABS SELECTION ---
tab1, tab2 = st.tabs(["Analyse Historique", "Prédiction en Direct"])

with tab1:
    # LA SAISONNALITÉ
    st.subheader("Analyse Saisonnière")
    st.write("Moyenne de l'AQI par mois (2022-2026)")
    df['mois'] = df['time'].dt.month
    monthly_avg = df.groupby('mois')['target_aqi'].mean()
    st.bar_chart(monthly_avg)
    st.caption("1 = Janvier, 12 = Décembre")
    
    st.divider()
    st.subheader("Évolution Temporelle")
    st.write("Aperçu des 500 derniers relevés (AQI vs Température)")
    chart_data = df.tail(500)
    st.line_chart(chart_data.set_index('time')[['target_aqi', 'temperature_2m']])

    st.divider()
    st.subheader("Analyse de Corrélation")
    st.write("Relation entre la Vitesse du Vent et la Pollution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['wind_speed_10m'], df['target_aqi'], alpha=0.3, color='#26619C')
    ax.set_xlabel("Vitesse du Vent (km/h)")
    ax.set_ylabel("AQI")
    ax.set_title("Impact du Vent sur la Qualité de l'Air")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    st.caption("On observe graphiquement que les AQI les plus élevés arrivent principalement lorsque le vent est faible.")

with tab2:
    st.subheader("Simulateur de Qualité de l'Air")
    st.write(f"Modèle entraîné sur **{len(df)}** relevés horaires.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        t = st.slider("Température (°C)", -10, 40, 20)
        h = st.slider("Humidité (%)", 0, 100, 50)
        w = st.slider("Vent (km/h)", 0, 100, 15)
        
    with col2:
        pred = model.predict([[t, h, w]])[0]
        pred = max(0, pred)
        
        st.metric("AQI Prédit", f"{pred:.1f}")
        
        # Interpretation simplifié 
        if pred < 50:
            st.success("L'air est sain. Idéal pour les activités extérieures.")
        elif pred < 100:
            st.info("La qualité de l'air est moyenne.")
        elif pred < 150:
            st.warning("Qualité dégradée. Les personnes sensibles, attention!")
        else:
            st.error("Mauvaise qualité de l'air ! Évitez les sorties.")

# Guide d'interprétation de l'indice de qualité de l'air
with st.sidebar:
    st.header(" Comprendre IQA ")
    st.markdown("""
    **0 - 50 : Bon**
    Air sain, aucun risque.
    
    **51 - 100 : Moyen**
    Qualité acceptable.
    
    **101 - 150 : Dégradé**
    Mauvais pour les personnes sensibles.
    
    **151 - 200 : Mauvais**
    Effets sur la santé pour tous.
    
    **201+ : Très Mauvais**
    Alerte santé !
    """)
    st.divider()
    st.caption("par Malik HOUNI")
