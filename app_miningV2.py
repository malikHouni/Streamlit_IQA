import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuration des villes (doit correspondre au data_minerV2)
CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522},
    "Marseille": {"lat": 43.2965, "lon": 5.3698},
    "Lyon": {"lat": 45.75, "lon": 4.85},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442},
    "Nice": {"lat": 43.7102, "lon": 7.2620},
    "Nantes": {"lat": 47.2184, "lon": -1.5536},
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521},
    "Montpellier": {"lat": 43.6108, "lon": 3.8767},
    "Bordeaux": {"lat": 44.8378, "lon": -0.5792},
    "Lille": {"lat": 50.6292, "lon": 3.0573}
}

st.set_page_config(page_title="France ML Miner V2", page_icon=None, layout="wide")

# --- Custom Styling ---
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

st.title("France Data Mining & Prediction V2")

# --- CITY SELECTION ---
selected_city = st.selectbox("Sélectionnez une ville pour l'analyse et la prédiction", list(CITIES.keys()), index=2)
st.divider()

# Path relative to the script
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "storage", f"{selected_city.lower()}_full_history.csv")

if not os.path.exists(DATA_FILE):
    st.warning(f"Aucune donnée trouvée pour {selected_city}. Veuillez utiliser le Data Miner V2 pour extraire les données de cette ville.")
    st.stop()

@st.cache_data
def load_and_mine(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['target_aqi'] = (df['pm10'] * 0.5 + df['pm2_5'] * 1.5 + df['nitrogen_dioxide'] * 0.2)
    return df

df = load_and_mine(DATA_FILE)

# Entraînement du modèle
@st.cache_resource
def train_professional_model(data):
    X = data[['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']].values
    y = data['target_aqi'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_professional_model(df)

# --- TABS SELECTION ---
tab1, tab2, tab3 = st.tabs(["Analyse Historique", "Prédiction en Direct", "Carte de France"])

with tab1:
    st.subheader(f"Analyse Saisonnière - {selected_city}")
    st.write("Moyenne de l'AQI par mois (2022-2024)")
    df['mois'] = df['time'].dt.month
    monthly_avg = df.groupby('mois')['target_aqi'].mean()
    st.bar_chart(monthly_avg)
    
    st.divider()
    st.subheader("Évolution Temporelle")
    chart_data = df.tail(500)
    st.line_chart(chart_data[['target_aqi', 'temperature_2m']])

    st.divider()
    st.subheader("Analyse de Corrélation")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['wind_speed_10m'], df['target_aqi'], alpha=0.3, color='#26619C')
    ax.set_xlabel("Vitesse du Vent (km/h)")
    ax.set_ylabel("AQI")
    ax.set_title(f"Impact du Vent sur la Qualité de l'Air ({selected_city})")
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

with tab2:
    st.subheader(f"Simulateur de Qualité de l'Air - {selected_city}")
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
        
        if pred < 50:
            st.success("L'air est sain. Idéal pour les activités extérieures.")
        elif pred < 100:
            st.info("La qualité de l'air est moyenne.")
        elif pred < 150:
            st.warning("Qualité dégradée. Les personnes sensibles, attention!")
        else:
            st.error("Mauvaise qualité de l'air ! Évitez les sorties.")

with tab3:
    st.subheader("Situation Nationale")
    
    map_list = []
    for city, coords in CITIES.items():
        path = os.path.join(os.path.dirname(__file__), "..", "storage", f"{city.lower()}_full_history.csv")
        if os.path.exists(path):
            try:
                # On lit seulement la dernière ligne pour la performance
                temp_df = pd.read_csv(path).tail(1).copy()
                aqi = (temp_df['pm10'].iloc[0] * 0.5 + temp_df['pm2_5'].iloc[0] * 1.5 + temp_df['nitrogen_dioxide'].iloc[0] * 0.2)
                # Significant scaling for visual impact
                map_list.append({
                    "city": city,
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "aqi": round(aqi, 1),
                    "size": (aqi ** 1.5) * 10 
                })
            except:
                pass
    
    if map_list:
        map_df = pd.DataFrame(map_list)
        
        # Mapping colors for Pydeck compatibility (RGBA)
        def get_color_rgb(aqi):
            if aqi < 50: return [0, 128, 0, 160]      # Green
            elif aqi < 100: return [255, 255, 0, 160]   # Yellow
            elif aqi < 150: return [255, 165, 0, 160]   # Orange
            else: return [255, 0, 0, 160]              # Red

        map_df['color'] = map_df['aqi'].apply(get_color_rgb)
        
        import pydeck as pdk
        
        # We use a very large radius (in meters) to make the dots big and visible
        # aqi 50 -> 50,000 meters (50km radius)
        # aqi 150 -> 150,000 meters (150km radius)
        
        st.pydeck_chart(pdk.Deck(
            map_provider='carto',
            map_style='light',
            initial_view_state=pdk.ViewState(
                latitude=46.2276,
                longitude=2.2137,
                zoom=5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_df,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_radius='aqi * 1000', 
                    pickable=True,
                ),
            ],
            tooltip={"text": "{city}\nAQI: {aqi}"}
        ))
        
        st.caption("La taille et la couleur des cercles representent le niveau de pollution (Plus le cercle est grand et rouge, plus l'air est pollue).")
    else:
        st.info("Aucune donnée disponible pour la carte. Veuillez miner les données dans le Data Miner V2.")

# Guide d'interprétation en barre latérale (Sidebar)
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
