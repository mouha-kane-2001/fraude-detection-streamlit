import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Configuration de la Page ---
st.set_page_config(
    page_title="FraudGuard | Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Personnalisé Premium ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fe 0%, #eef2f7 100%);
        padding: 0 !important;
    }
    
    .header-glass {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        padding: 25px 40px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 30px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
    }
    
    .header-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-title h1 {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    
    .header-title p {
        color: #4a5568;
        font-size: 16px;
        font-weight: 400;
        -webkit-text-fill-color: #4a5568;
    }
    
    .header-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .modern-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .modern-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.12);
    }
    
    .card-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }
    
    .card-value {
        font-size: 32px;
        font-weight: 800;
        color: #1a202c;
        margin: 5px 0;
    }
    
    .card-value-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .card-label {
        font-size: 13px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-sub {
        font-size: 14px;
        font-weight: 500;
        margin-top: 5px;
    }
    
    .card-sub-success {
        color: #48bb78;
    }
    
    .card-sub-danger {
        color: #f56565;
    }
    
    .card-sub-warning {
        color: #ed8936;
    }
    
    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        background: #f7fafc;
    }
    
    .stat-badge-success {
        background: #c6f6d5;
        color: #22543d;
    }
    
    .stat-badge-danger {
        background: #fed7d7;
        color: #9b2c2c;
    }
    
    .stat-badge-warning {
        background: #feebc8;
        color: #7b341e;
    }
    
    .nav-tabs {
        background: white;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 30px;
        display: flex;
        gap: 8px;
    }
    
    .nav-tab {
        flex: 1;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 600;
        color: #4a5568;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        border: none;
        background: transparent;
        font-size: 15px;
    }
    
    .nav-tab:hover {
        background: rgba(102, 126, 234, 0.08);
    }
    
    .nav-tab-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .upload-zone {
        border: 3px dashed #cbd5e0;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: white;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.03);
        transform: scale(1.02);
    }
    
    .upload-zone-icon {
        font-size: 48px;
        margin-bottom: 10px;
    }
    
    .upload-zone-title {
        font-size: 18px;
        font-weight: 700;
        color: #2d3748;
    }
    
    .upload-zone-sub {
        color: #718096;
        font-size: 14px;
        margin-top: 5px;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: #edf2f7;
        overflow: hidden;
        margin-top: 10px;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    .progress-fill-success {
        background: linear-gradient(90deg, #48bb78, #38a169);
    }
    
    .progress-fill-danger {
        background: linear-gradient(90deg, #fc8181, #f56565);
    }
    
    .progress-fill-warning {
        background: linear-gradient(90deg, #f6ad55, #ed8936);
    }
    
    .info-box {
        background: #ebf8ff;
        border-radius: 12px;
        padding: 15px 20px;
        border-left: 4px solid #3182ce;
        margin-bottom: 20px;
        color: #2c5282;
    }
    
    .prediction-box {
        border-radius: 16px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .prediction-safe {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #48bb78;
    }
    
    .prediction-fraud {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #f56565;
    }
    
    .prediction-title {
        font-size: 28px;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .footer-premium {
        text-align: center;
        padding: 30px;
        color: #718096;
        font-size: 13px;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 40px;
        background: white;
        border-radius: 20px 20px 0 0;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    .control-group {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("❌ Fichiers `fraud_model.pkl` ou `scaler.pkl` introuvables dans le dossier `model/`.")
    st.stop()

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
st.markdown("""
<div class="header-glass">
    <div class="header-title">
        <div>
            <h1>🛡️ FraudGuard</h1>
            <p style="margin-top: 5px;">Système intelligent de détection de fraudes bancaires</p>
        </div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <span class="header-badge">⚡ IA en temps réel</span>
            <span class="header-badge" style="background: #48bb78;">✅ Modèle actif</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# NAVIGATION
# ==============================================================================
st.markdown("""
<div class="nav-tabs">
    <button class="nav-tab nav-tab-active" onclick="window.location.href='?mode=lot'">
        📊 Analyse par Lot
    </button>
    <button class="nav-tab" onclick="window.location.href='?mode=single'">
        🔍 Transaction Unique
    </button>
</div>
""", unsafe_allow_html=True)

# Gestion du mode via query params
query_params = st.query_params
mode = query_params.get("mode", "lot")

if mode == "single":
    mode_selected = "🔍 Transaction Unique"
else:
    mode_selected = "📊 Analyse par Lot"

# ==============================================================================
# FONCTION POUR AFFICHER LES STATISTIQUES
# ==============================================================================
def display_stats(df_results, predictions, col_target=None):
    """Affiche toutes les statistiques de manière cohérente"""
    
    total_tx = len(df_results)
    total_fraudes = int((predictions == 1).sum())
    total_legitimes = total_tx - total_fraudes
    taux_fraude = (total_fraudes / total_tx) * 100
    taux_legitime = 100 - taux_fraude
    
    col_montant = next((c for c in df_results.columns if 'montant' in c.lower() or 'amount' in c.lower()), None)
    montant_fraude = df_results[df_results["Prediction"] == 1][col_montant].sum() if col_montant else 0
    montant_total = df_results[col_montant].sum() if col_montant else 0
    
    st.markdown("""
    <h3 style="font-size: 20px; font-weight: 700; color: #1a202c; margin-bottom: 20px;">
        📈 Indicateurs de Performance
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">📊</div>
            <div class="card-label">Volume Total</div>
            <div class="card-value card-value-gradient">{total_tx:,}</div>
            <div class="card-sub" style="color: #718096;">Transactions analysées</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">🚨</div>
            <div class="card-label">Fraudes Détectées</div>
            <div class="card-value" style="color: #f56565;">{total_fraudes:,}</div>
            <div class="card-sub card-sub-danger">{taux_fraude:.1f}% du total</div>
            <div class="progress-bar">
                <div class="progress-fill progress-fill-danger" style="width: {taux_fraude}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">✅</div>
            <div class="card-label">Transactions Légitimes</div>
            <div class="card-value" style="color: #48bb78;">{total_legitimes:,}</div>
            <div class="card-sub card-sub-success">{taux_legitime:.1f}% du total</div>
            <div class="progress-bar">
                <div class="progress-fill progress-fill-success" style="width: {taux_legitime}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if col_montant:
            st.markdown(f"""
            <div class="modern-card">
                <div class="card-icon">💰</div>
                <div class="card-label">Montant Suspect</div>
                <div class="card-value" style="color: #ed8936;">{montant_fraude:,.0f}</div>
                <div class="card-sub card-sub-warning">{((montant_fraude/montant_total)*100 if montant_total > 0 else 0):.1f}% du volume total</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="modern-card">
                <div class="card-icon">📊</div>
                <div class="card-label">Taux de Risque</div>
                <div class="card-value card-value-gradient">{taux_fraude:.1f}%</div>
                <div class="card-sub" style="color: #718096;">Transactions suspectes</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <h3 style="font-size: 20px; font-weight: 700; color: #1a202c; margin-bottom: 20px;">
        🎯 Métriques Avancées
    </h3>
    """, unsafe_allow_html=True)
    
    if col_target:
        y_true = df_results[col_target]
        y_pred = predictions
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-label">🎯 Exactitude</span>
                    <span class="stat-badge stat-badge-success">{acc:.1%}</span>
                </div>
                <div class="card-value card-value-gradient" style="font-size: 28px;">{acc:.1%}</div>
                <div class="card-sub" style="color: #718096;">Proportion de prédictions correctes</div>
                <div class="progress-bar">
                    <div class="progress-fill progress-fill-success" style="width: {acc*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-label">🎯 Précision</span>
                    <span class="stat-badge stat-badge-success">{prec:.1%}</span>
                </div>
                <div class="card-value card-value-gradient" style="font-size: 28px;">{prec:.1%}</div>
                <div class="card-sub" style="color: #718096;">Parmi les fraudes prédites, combien sont réelles</div>
                <div class="progress-bar">
                    <div class="progress-fill progress-fill-success" style="width: {prec*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-label">🎯 Rappel (Recall)</span>
                    <span class="stat-badge stat-badge-success">{rec:.1%}</span>
                </div>
                <div class="card-value card-value-gradient" style="font-size: 28px;">{rec:.1%}</div>
                <div class="card-sub" style="color: #718096;">Quel % des vraies fraudes a été détecté</div>
                <div class="progress-bar">
                    <div class="progress-fill progress-fill-success" style="width: {rec*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            st.markdown(f"""
            <div class="modern-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="card-label">🎯 F1-Score</span>
                    <span class="stat-badge stat-badge-success">{f1:.1%}</span>
                </div>
                <div class="card-value card-value-gradient" style="font-size: 28px;">{f1:.1%}</div>
                <div class="card-sub" style="color: #718096;">Moyenne harmonique précision/rappel</div>
                <div class="progress-bar">
                    <div class="progress-fill progress-fill-success" style="width: {f1*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            st.markdown("""
            <h4 style="font-weight: 600; color: #1a202c;">📊 Matrice de Confusion</h4>
            """, unsafe_allow_html=True)
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Prédit", y="Réel", color="Nombre"),
                x=['Légitime', 'Fraude'],
                y=['Légitime', 'Fraude'],
                color_continuous_scale="Blues",
                title=""
            )
            fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col_cm2:
            st.markdown("""
            <h4 style="font-weight: 600; color: #1a202c;">📈 Distribution des Risques</h4>
            """, unsafe_allow_html=True)
            risk_counts = df_results["Niveau_Risque"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.45,
                marker=dict(colors=['#48bb78', '#ed8936', '#f56565']),
                textinfo='label+percent',
                textposition='outside'
            )])
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("""
            <h4 style="font-weight: 600; color: #1a202c;">📈 Distribution des Risques</h4>
            """, unsafe_allow_html=True)
            risk_counts = df_results["Niveau_Risque"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.45,
                marker=dict(colors=['#48bb78', '#ed8936', '#f56565']),
                textinfo='label+percent',
                textposition='outside'
            )])
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.markdown("""
            <h4 style="font-weight: 600; color: #1a202c;">📊 Distribution des Probabilités</h4>
            """, unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_results["Probabilité"],
                nbinsx=30,
                marker=dict(color='#667eea', line=dict(color='white', width=1)),
                opacity=0.8
            ))
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Seuil 50%", annotation_position="top")
            fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), 
                                 xaxis_title="Probabilité", yaxis_title="Nombre")
            st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# FONCTION POUR GÉNÉRER DES DONNÉES D'EXEMPLE
# ==============================================================================
def generate_sample_data():
    """Génère des données d'exemple pour l'affichage initial"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'montant': np.random.exponential(50000, n_samples) + 1000,
        'heure': np.random.randint(0, 24, n_samples),
        'localisation_non_habituelle': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'target': np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    }
    df = pd.DataFrame(data)
    
    predictions = df['target'].copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    predictions.iloc[error_indices] = 1 - predictions.iloc[error_indices]
    
    probas = []
    for i in range(n_samples):
        if predictions.iloc[i] == 1:
            probas.append(np.random.uniform(0.6, 0.95))
        else:
            probas.append(np.random.uniform(0.01, 0.35))
    
    df_results = df.copy()
    df_results["Prediction"] = predictions
    df_results["Statut"] = ["🚨 Fraude" if p == 1 else "✅ Légitime" for p in predictions]
    df_results["Probabilité"] = probas
    df_results["Niveau_Risque"] = pd.cut(probas, bins=[-0.1, 0.3, 0.7, 1.0], 
                                        labels=["🟢 Faible", "🟡 Moyen", "🔴 Élevé"])
    
    return df_results, np.array(predictions)

# ==============================================================================
# FONCTION POUR PRÉDIRE UNE TRANSACTION UNIQUE
# ==============================================================================
def predict_transaction(montant, heure, localisation_non, canal):
    """Prédit le risque d'une transaction unique"""
    num_features = scaler.n_features_in_
    features = np.zeros((1, num_features))
    
    # Assurez-vous que les indices correspondent à ceux utilisés lors de l'entraînement
    features[0, 0] = montant
    features[0, 1] = heure
    features[0, 2] = localisation_non
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]
    
    return prediction, proba

# ==============================================================================
# MODE 1 : ANALYSE PAR LOT
# ==============================================================================
if mode_selected == "📊 Analyse par Lot":
    
    st.markdown("""
    <div style="margin-bottom: 25px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #1a202c;">📊 Analyse de Fraude par Lot</h2>
        <p style="color: #718096; font-size: 15px;">Importez un fichier CSV pour analyser les transactions en masse</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==============================================================
    # SECTION INTERACTIVE - TEST DE TRANSACTION
    # ==============================================================
    st.markdown("""
    <div class="info-box">
        🎯 <strong>Testez une transaction en temps réel</strong> - Ajustez les paramètres ci-dessous pour voir la prédiction instantanément
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="control-group">', unsafe_allow_html=True)
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 15px;">
            ⚙️ Paramètres de la transaction
        </h4>
        """, unsafe_allow_html=True)
        
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            montant_test = st.number_input(
                "💰 Montant (FCFA)", 
                min_value=0.0, 
                value=75000.0, 
                step=5000.0,
                key="montant_test_lot"
            )
        
        with col_input2:
            heure_test = st.slider(
                "🕐 Heure de l'opération", 
                0, 23, 14,
                key="heure_test_lot"
            )
        
        with col_input3:
            localisation_test = st.radio(
                "📍 Localisation habituelle ?",
                ["Oui", "Non"],
                horizontal=True,
                key="localisation_test_lot"
            )
        
        canal_test = st.selectbox(
            "📱 Canal de paiement",
            ["Mobile Money", "Carte bancaire", "Virement", "ATM"],
            key="canal_test_lot"
        )
        
        # Prédiction automatique
        localisation_non = 1 if localisation_test == "Non" else 0
        prediction, proba = predict_transaction(montant_test, heure_test, localisation_non, canal_test)
        
        # Affichage du résultat en temps réel
        st.markdown("---")
        st.markdown("""
        <h5 style="font-weight: 600; color: #1a202c; margin-bottom: 10px;">
            📊 Résultat de l'analyse
        </h5>
        """, unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            if prediction == 1:
                st.markdown("""
                <div style="background: #fee2e2; border-radius: 12px; padding: 15px; text-align: center;">
                    <div style="font-size: 32px;">🚨</div>
                    <div style="font-weight: 700; color: #991b1b;">FRAUDE</div>
                    <div style="font-size: 12px; color: #7f1d1d;">Risque élevé</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #d1fae5; border-radius: 12px; padding: 15px; text-align: center;">
                    <div style="font-size: 32px;">✅</div>
                    <div style="font-weight: 700; color: #065f46;">LÉGITIME</div>
                    <div style="font-size: 12px; color: #065f46;">Risque faible</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_res2:
            st.metric("📊 Probabilité", f"{proba:.1%}")
        
        with col_res3:
            if proba < 0.3:
                risk_level = "🟢 Faible"
            elif proba < 0.7:
                risk_level = "🟡 Moyen"
            else:
                risk_level = "🔴 Élevé"
            st.metric("🎯 Niveau", risk_level)
        
        with col_res4:
            st.metric("📱 Canal", canal_test)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ==============================================================
    # STATISTIQUES PAR DÉFAUT
    # ==============================================================
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        ℹ️ <strong>Statistiques démonstratives</strong> - Importez votre propre fichier CSV pour analyser vos données réelles
    </div>
    """, unsafe_allow_html=True)
    
    df_demo, preds_demo = generate_sample_data()
    col_target_demo = next((c for c in df_demo.columns if c.lower() in ['target', 'fraude', 'label', 'is_fraud']), None)
    display_stats(df_demo, preds_demo, col_target_demo)
    
    # ==============================================================
    # UPLOAD ZONE
    # ==============================================================
    st.markdown("---")
    st.markdown("""
    <h3 style="font-size: 18px; font-weight: 700; color: #1a202c; margin-bottom: 15px;">
        📤 Importer vos données réelles
    </h3>
    """, unsafe_allow_html=True)
    
    col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
    with col_upload2:
        with st.container():
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-zone-icon">📂</div>
                <div class="upload-zone-title">Déposez votre fichier CSV</div>
                <div class="upload-zone-sub">Support des formats standard • Sécurisé</div>
            </div>
            """, unsafe_allow_html=True)
            fichier = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    
    if fichier is not None:
        try:
            df_raw = pd.read_csv(fichier, sep=None, engine="python")
            
            with st.expander("👁️ Aperçu des Données", expanded=False):
                st.dataframe(df_raw.head(10), use_container_width=True, height=300)
            
            if st.button("🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True):
                
                with st.spinner("🔍 Analyse en cours..."):
                    expected_features = getattr(scaler, "feature_names_in_", None)
                    df_processed = pd.get_dummies(df_raw)
                    
                    if expected_features is not None:
                        df_processed = df_processed.reindex(columns=expected_features, fill_value=0)
                    
                    X_scaled = scaler.transform(df_processed)
                    predictions = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)[:, 1]
                    
                    df_results = df_raw.copy()
                    df_results["Prediction"] = predictions
                    df_results["Statut"] = ["🚨 Fraude" if p == 1 else "✅ Légitime" for p in predictions]
                    df_results["Probabilité"] = probas
                    df_results["Niveau_Risque"] = pd.cut(probas, bins=[-0.1, 0.3, 0.7, 1.0], 
                                                        labels=["🟢 Faible", "🟡 Moyen", "🔴 Élevé"])
                    
                    col_target = next((c for c in df_results.columns if c.lower() in ['target', 'fraude', 'label', 'is_fraud']), None)
                
                st.markdown("---")
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                
                display_stats(df_results, predictions, col_target)
                
                st.markdown("---")
                st.markdown("""
                <h3 style="font-size: 20px; font-weight: 700; color: #1a202c; margin-bottom: 20px;">
                    📋 Détail des Transactions
                </h3>
                """, unsafe_allow_html=True)
                
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    filter_risk = st.selectbox("🎯 Filtrer par risque", ["Tous", "🟢 Faible", "🟡 Moyen", "🔴 Élevé"])
                with col_f2:
                    filter_status = st.selectbox("📌 Filtrer par statut", ["Tous", "✅ Légitime", "🚨 Fraude"])
                with col_f3:
                    search_col = st.text_input("🔍 Rechercher", placeholder="Rechercher...")
                
                df_display = df_results.copy()
                if filter_risk != "Tous":
                    df_display = df_display[df_display["Niveau_Risque"] == filter_risk]
                if filter_status != "Tous":
                    df_display = df_display[df_display["Statut"] == filter_status]
                if search_col:
                    mask = df_display.astype(str).apply(lambda x: x.str.contains(search_col, case=False)).any(axis=1)
                    df_display = df_display[mask]
                
                def style_dataframe(df):
                    def highlight_risk(row):
                        if row['Niveau_Risque'] == '🔴 Élevé':
                            return ['background-color: #fee2e2; color: #991b1b; font-weight: 600;' for _ in row]
                        elif row['Niveau_Risque'] == '🟡 Moyen':
                            return ['background-color: #fef3c7; color: #92400e; font-weight: 600;' for _ in row]
                        else:
                            return ['background-color: #d1fae5; color: #065f46; font-weight: 600;' for _ in row]
                    return df.style.apply(highlight_risk, axis=1)
                
                cols_display = [c for c in df_display.columns if c not in ["Prediction"]]
                st.dataframe(style_dataframe(df_display[cols_display]), use_container_width=True, height=400)
                
                st.markdown("---")
                col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
                with col_exp2:
                    csv_data = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Télécharger le Rapport Complet (CSV)",
                        data=csv_data,
                        file_name=f"rapport_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement : {e}")

# ==============================================================================
# MODE 2 : TRANSACTION UNIQUE
# ==============================================================================
else:
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="font-size: 24px; font-weight: 700; color: #1a202c;">🔍 Analyse Transaction Unique</h2>
        <p style="color: #718096; font-size: 15px;">Évaluez instantanément le risque d'une transaction spécifique</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.06);">
        """, unsafe_allow_html=True)
        
        with st.form("single_tx_form"):
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                montant = st.number_input("💰 Montant (FCFA)", min_value=0.0, value=75000.0, step=5000.0,
                                         help="Montant de la transaction en Francs CFA")
                localisation_habituelle = st.radio("📍 Localisation habituelle du client ?", ["Oui", "Non"], horizontal=True)
            
            with col_input2:
                heure = st.slider("🕐 Heure de l'opération", 0, 23, 14,
                                 help="Heure à laquelle la transaction a été effectuée")
                canal = st.selectbox("📱 Canal de paiement", 
                                    ["Mobile Money", "Carte bancaire", "Virement", "ATM"],
                                    help="Moyen de paiement utilisé")
            
            st.markdown("---")
            btn_submit = st.form_submit_button("🛡️ Évaluer le Risque", type="primary", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    if btn_submit:
        with st.spinner("🔍 Analyse en cours..."):
            localisation_non = 1 if localisation_habituelle == "Non" else 0
            prediction, proba = predict_transaction(montant, heure, localisation_non, canal)
        
        st.markdown("---")
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fee2e2, #fecaca); border-radius: 20px; padding: 30px; border: 2px solid #f56565;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="background: #f56565; color: white; padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 14px;">🚨 RISQUE ÉLEVÉ</span>
                        <h2 style="margin: 15px 0 5px 0; color: #9b2c2c; font-weight: 800;">Transaction Suspecte</h2>
                        <p style="color: #742a2a; font-size: 16px;">Cette transaction présente des caractéristiques anormales.</p>
                    </div>
                    <div style="font-size: 72px;">🚨</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-radius: 20px; padding: 30px; border: 2px solid #48bb78;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span style="background: #48bb78; color: white; padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 14px;">✅ RISQUE FAIBLE</span>
                        <h2 style="margin: 15px 0 5px 0; color: #065f46; font-weight: 800;">Transaction Légitime</h2>
                        <p style="color: #065f46; font-size: 16px;">Aucun signe de fraude détecté.</p>
                    </div>
                    <div style="font-size: 72px;">✅</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("📊 Probabilité de Fraude", f"{proba:.1%}", 
                     delta=f"Seuil: 50%", delta_color="off")
        
        with col_res2:
            if proba < 0.3:
                status_text = "🟢 Faible"
            elif proba < 0.7:
                status_text = "🟡 Moyen"
            else:
                status_text = "🔴 Élevé"
            st.metric("🎯 Niveau de Risque", status_text)
        
        with col_res3:
            st.metric("📱 Canal", canal)
        
        st.markdown("---")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Indicateur de Risque", 'font': {'size': 24, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'steps': [
                    {'range': [0, 30], 'color': '#48bb78'},
                    {'range': [30, 70], 'color': '#ed8936'},
                    {'range': [70, 100], 'color': '#f56565'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer-premium">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 10px;">
        <span>🛡️ FraudGuard v2.0</span>
        <span>⚡ Propulsé par Machine Learning</span>
        <span>🔒 Données sécurisées</span>
    </div>
    <p style="margin: 0; font-size: 12px;">© 2026 - Système de Détection de Fraude Bancaire</p>
</div>
""", unsafe_allow_html=True)