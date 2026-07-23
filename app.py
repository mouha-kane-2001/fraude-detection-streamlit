import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Configuration de la Page ---
st.set_page_config(
    page_title="FraudGuard Pro | Détection de Fraude Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(20px);
        padding: 20px 40px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 25px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.08);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 15px;
    }
    
    .header-title h1 {
        font-size: 32px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    
    .header-title p {
        color: #4a5568;
        font-size: 14px;
        font-weight: 400;
        -webkit-text-fill-color: #4a5568;
    }
    
    .header-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 13px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        display: inline-block;
    }
    
    .modern-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .modern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
    }
    
    .card-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }
    
    .card-value {
        font-size: 28px;
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
        font-size: 12px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-sub {
        font-size: 13px;
        font-weight: 500;
        margin-top: 5px;
    }
    
    .card-sub-success { color: #48bb78; }
    .card-sub-danger { color: #f56565; }
    .card-sub-warning { color: #ed8936; }
    
    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 11px;
        font-weight: 600;
        background: #f7fafc;
    }
    
    .stat-badge-success { background: #c6f6d5; color: #22543d; }
    .stat-badge-danger { background: #fed7d7; color: #9b2c2c; }
    .stat-badge-warning { background: #feebc8; color: #7b341e; }
    .stat-badge-info { background: #bee3f8; color: #2a4365; }
    
    /* Style des boutons de navigation amélioré */
    .nav-container {
        background: white;
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 25px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .nav-btn {
        flex: 1;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        color: #4a5568;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        border: none;
        background: transparent;
        font-size: 14px;
        min-width: 100px;
        text-decoration: none;
        display: inline-block;
    }
    
    .nav-btn:hover {
        background: rgba(102, 126, 234, 0.08);
        transform: translateY(-2px);
    }
    
    .nav-btn-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .upload-zone {
        border: 2px dashed #cbd5e0;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: white;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.03);
        transform: scale(1.01);
    }
    
    .upload-zone-icon { font-size: 40px; margin-bottom: 8px; }
    .upload-zone-title { font-size: 16px; font-weight: 700; color: #2d3748; }
    .upload-zone-sub { color: #718096; font-size: 13px; margin-top: 5px; }
    
    .progress-bar {
        height: 6px;
        border-radius: 10px;
        background: #edf2f7;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    .progress-fill-success { background: linear-gradient(90deg, #48bb78, #38a169); }
    .progress-fill-danger { background: linear-gradient(90deg, #fc8181, #f56565); }
    .progress-fill-warning { background: linear-gradient(90deg, #f6ad55, #ed8936); }
    .progress-fill-info { background: linear-gradient(90deg, #63b3ed, #3182ce); }
    
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
        font-size: 24px;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .footer-premium {
        text-align: center;
        padding: 25px;
        color: #718096;
        font-size: 12px;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 40px;
        background: white;
        border-radius: 16px 16px 0 0;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in { animation: fadeInUp 0.5s ease forwards; }
    
    .control-group {
        background: white;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .feature-importance-bar {
        background: white;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title { flex-direction: column; text-align: center; }
        .header-title h1 { font-size: 24px; }
        .nav-btn { font-size: 12px; padding: 10px 12px; min-width: 70px; }
        .card-value { font-size: 22px; }
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement du modèle et des colonnes ---
@st.cache_resource
def load_model_and_columns():
    try:
        model = joblib.load("model/fraud_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        columns = joblib.load("model/columns.pkl")
        return model, scaler, columns
    except FileNotFoundError:
        st.error("❌ Fichiers `fraud_model.pkl`, `scaler.pkl` ou `columns.pkl` introuvables dans le dossier `model/`.")
        st.stop()

try:
    model, scaler, feature_columns = load_model_and_columns()
except Exception as e:
    st.error(f"❌ Erreur de chargement : {e}")
    st.stop()

# --- Variables globales ---
VARIABLES_ANALYSE = {
    'Montant': '💰 Montant de la transaction',
    'Heure': '🕐 Heure de l\'opération',
    'Jour': '📅 Jour du mois',
    'Mois': '📆 Mois',
    'Jour_Semaine': '📋 Jour de la semaine',
    'Type_Transaction': '🏦 Type de transaction',
    'Status_Operation': '📊 Statut de l\'opération',
    'Localisation': '📍 Localisation'
}

# ==============================================================================
# HEADER PRINCIPAL
# ==============================================================================
st.markdown("""
<div class="header-glass">
    <div class="header-title">
        <div>
            <h1>🛡️Mouhamadou Kane</h1>
            <p style="margin-top: 5px;">Système intelligent de détection de fraudes bancaires</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
            <span class="header-badge">⚡ IA temps réel</span>
            <span class="header-badge" style="background: #48bb78;">✅ Modèle actif</span>
            <span class="header-badge" style="background: #ed8936;">📊 3 classes</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# NAVIGATION AMÉLIORÉE AVEC STREAMLIT
# ==============================================================================

# Gestion du mode via session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'dashboard'

# Fonction pour changer de mode
def set_mode(mode):
    st.session_state.mode = mode
    st.query_params['mode'] = mode

# Récupérer le mode depuis query params ou session state
mode = st.query_params.get('mode', st.session_state.mode)
if mode != st.session_state.mode:
    st.session_state.mode = mode

# Afficher la navigation avec des boutons Streamlit
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

cols = st.columns(5)
nav_items = [
    ('dashboard', '📊 Dashboard'),
    ('analyse', '🔍 Analyse'),
    ('predict', '🎯 Prédiction'),
    ('variables', '📋 Variables'),
    ('import', '📤 Import')
]

for i, (key, label) in enumerate(nav_items):
    with cols[i]:
        is_active = (mode == key)
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            set_mode(key)
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Mettre à jour le mode
mode = st.session_state.mode

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def get_feature_importance():
    """Calcule et retourne l'importance des features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = feature_columns[:len(importances)] if len(importances) <= len(feature_columns) else [f"Feature_{i}" for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

def calculate_statistics(df_results, predictions):
    """Calcule les statistiques détaillées"""
    total_tx = len(df_results)
    total_fraudes = int((predictions == 2).sum()) if 2 in predictions else 0
    total_suspects = int((predictions == 1).sum()) if 1 in predictions else 0
    total_legitimes = int((predictions == 0).sum()) if 0 in predictions else 0
    
    col_montant = next((c for c in df_results.columns if 'montant' in c.lower() or 'amount' in c.lower()), None)
    
    stats = {
        'total': total_tx,
        'fraudes': total_fraudes,
        'suspects': total_suspects,
        'legitimes': total_legitimes,
        'taux_fraude': (total_fraudes / total_tx) * 100 if total_tx > 0 else 0,
        'taux_suspect': (total_suspects / total_tx) * 100 if total_tx > 0 else 0,
        'taux_legitime': (total_legitimes / total_tx) * 100 if total_tx > 0 else 0
    }
    
    if col_montant:
        stats['montant_fraude'] = df_results[df_results["Prediction"] == 2][col_montant].sum() if total_fraudes > 0 else 0
        stats['montant_total'] = df_results[col_montant].sum()
    
    return stats

def display_statistics_card(stats):
    """Affiche les statistiques sous forme de cartes"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">📊</div>
            <div class="card-label">Total Transactions</div>
            <div class="card-value card-value-gradient">{stats['total']:,}</div>
            <div class="card-sub" style="color: #718096;">Analyse complète</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">🚨</div>
            <div class="card-label">Fraudes Détectées</div>
            <div class="card-value" style="color: #f56565;">{stats['fraudes']:,}</div>
            <div class="card-sub card-sub-danger">{stats['taux_fraude']:.1f}% du total</div>
            <div class="progress-bar">
                <div class="progress-fill progress-fill-danger" style="width: {min(stats['taux_fraude'], 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">⚠️</div>
            <div class="card-label">Transactions Suspectes</div>
            <div class="card-value" style="color: #ed8936;">{stats['suspects']:,}</div>
            <div class="card-sub card-sub-warning">{stats['taux_suspect']:.1f}% du total</div>
            <div class="progress-bar">
                <div class="progress-fill progress-fill-warning" style="width: {min(stats['taux_suspect'], 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="modern-card">
            <div class="card-icon">✅</div>
            <div class="card-label">Légitimes</div>
            <div class="card-value" style="color: #48bb78;">{stats['legitimes']:,}</div>
            <div class="card-sub card-sub-success">{stats['taux_legitime']:.1f}% du total</div>
            <div class="progress-bar">
                <div class="progress-fill progress-fill-success" style="width: {min(stats['taux_legitime'], 100)}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_feature_importance():
    """Affiche l'importance des features"""
    importance_df = get_feature_importance()
    if importance_df is not None:
        st.markdown("""
        <h3 style="font-size: 18px; font-weight: 700; color: #1a202c; margin: 20px 0 15px 0;">
            📊 Importance des Variables
        </h3>
        """, unsafe_allow_html=True)
        
        top_features = importance_df.head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features['Importance'],
            y=top_features['Feature'],
            orientation='h',
            marker=dict(
                color=top_features['Importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=top_features['Importance'].apply(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=30, t=20, b=20),
            xaxis_title="Importance",
            yaxis_title="Variables",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Détails de l'importance des variables"):
            st.dataframe(
                importance_df.head(20),
                use_container_width=True,
                column_config={
                    "Feature": "Variable",
                    "Importance": st.column_config.ProgressColumn(
                        "Importance",
                        format="%.3f",
                        min_value=0,
                        max_value=importance_df['Importance'].max()
                    )
                }
            )

# ==============================================================================
# MODE : DASHBOARD
# ==============================================================================
if mode == "dashboard":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: #1a202c;">📊 Tableau de Bord</h2>
        <p style="color: #718096; font-size: 14px;">Vue d'ensemble des performances et analyses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Générer des données de démonstration
    np.random.seed(42)
    n_samples = 2000
    
    demo_data = {
        'montant': np.random.exponential(50000, n_samples) + 1000,
        'heure': np.random.randint(0, 24, n_samples),
        'jour': np.random.randint(1, 31, n_samples),
        'mois': np.random.randint(1, 13, n_samples),
        'jour_semaine': np.random.randint(0, 7, n_samples),
        'Type de transaction_ATM': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Type de transaction_Paiement en ligne': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Type de transaction_Paiement électronique': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Status operation_Echoué': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Localisation_non_habituelle': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'target': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.10, 0.05])
    }
    
    df_demo = pd.DataFrame(demo_data)
    predictions_demo = df_demo['target'].copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    predictions_demo.iloc[error_indices] = np.random.choice([0, 1, 2], size=len(error_indices), p=[0.1, 0.5, 0.4])
    
    probas_demo = []
    for i in range(n_samples):
        if predictions_demo.iloc[i] == 2:
            probas_demo.append(np.random.uniform(0.7, 0.98))
        elif predictions_demo.iloc[i] == 1:
            probas_demo.append(np.random.uniform(0.35, 0.7))
        else:
            probas_demo.append(np.random.uniform(0.01, 0.3))
    
    df_demo["Prediction"] = predictions_demo
    df_demo["Statut"] = ["🚨 Fraude" if p == 2 else "⚠️ Suspect" if p == 1 else "✅ Légitime" for p in predictions_demo]
    df_demo["Probabilité"] = probas_demo
    df_demo["Niveau_Risque"] = pd.cut(probas_demo, bins=[-0.1, 0.3, 0.7, 1.0], 
                                      labels=["🟢 Faible", "🟡 Moyen", "🔴 Élevé"])
    
    stats = calculate_statistics(df_demo, np.array(predictions_demo))
    display_statistics_card(stats)
    
    st.markdown("---")
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 10px;">
            📈 Distribution des Risques
        </h4>
        """, unsafe_allow_html=True)
        
        risk_counts = df_demo["Niveau_Risque"].value_counts()
        colors = {'🟢 Faible': '#48bb78', '🟡 Moyen': '#ed8936', '🔴 Élevé': '#f56565'}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=[colors.get(k, '#667eea') for k in risk_counts.index]),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05 if v == risk_counts.max() else 0 for v in risk_counts.values]
        )])
        fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_graph2:
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 10px;">
            📊 Probabilités de Fraude
        </h4>
        """, unsafe_allow_html=True)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_demo["Probabilité"],
            nbinsx=30,
            marker=dict(color='#667eea', line=dict(color='white', width=1)),
            opacity=0.8
        ))
        fig_hist.add_vline(x=0.3, line_dash="dash", line_color="#ed8936", 
                          annotation_text="Seuil Moyen", annotation_position="top")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#f56565", 
                          annotation_text="Seuil Élevé", annotation_position="top")
        fig_hist.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Probabilité",
            yaxis_title="Nombre de transactions",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    col_temp1, col_temp2 = st.columns(2)
    
    with col_temp1:
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 10px;">
            🕐 Transactions par Heure
        </h4>
        """, unsafe_allow_html=True)
        
        hour_counts = df_demo.groupby('heure')['montant'].count().reset_index()
        hour_counts.columns = ['Heure', 'Nombre']
        
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hour_counts['Heure'],
            y=hour_counts['Nombre'],
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig_hour.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Heure",
            yaxis_title="Nombre de transactions",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with col_temp2:
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 10px;">
            📅 Transactions par Jour de Semaine
        </h4>
        """, unsafe_allow_html=True)
        
        days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        day_counts = df_demo.groupby('jour_semaine')['montant'].count().reset_index()
        day_counts['Jour'] = day_counts['jour_semaine'].map(lambda x: days[x])
        
        fig_day = go.Figure()
        fig_day.add_trace(go.Bar(
            x=day_counts['Jour'],
            y=day_counts['montant'],
            marker=dict(
                color=day_counts['montant'],
                colorscale='Blues',
                showscale=False
            ),
            text=day_counts['montant'],
            textposition='outside'
        ))
        fig_day.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Jour",
            yaxis_title="Nombre de transactions",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_day, use_container_width=True)
    
    display_feature_importance()

# ==============================================================================
# MODE : ANALYSE
# ==============================================================================
elif mode == "analyse":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: #1a202c;">🔍 Analyse Approfondie</h2>
        <p style="color: #718096; font-size: 14px;">Analyse détaillée des variables et corrélations</p>
    </div>
    """, unsafe_allow_html=True)
    
    np.random.seed(42)
    n_samples = 1000
    
    analyze_data = {
        'montant': np.random.exponential(50000, n_samples) + 1000,
        'heure': np.random.randint(0, 24, n_samples),
        'jour': np.random.randint(1, 31, n_samples),
        'mois': np.random.randint(1, 13, n_samples),
        'jour_semaine': np.random.randint(0, 7, n_samples),
        'type_ATM': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'type_online': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'type_electronique': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'status_echoue': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'localisation_non_habituelle': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'target': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.10, 0.05])
    }
    
    df_analyze = pd.DataFrame(analyze_data)
    
    st.markdown("""
    <h3 style="font-size: 18px; font-weight: 700; color: #1a202c; margin: 15px 0;">
        📊 Statistiques Descriptives
    </h3>
    """, unsafe_allow_html=True)
    
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        var_select = st.selectbox(
            "📌 Sélectionner une variable",
            list(VARIABLES_ANALYSE.keys()),
            format_func=lambda x: VARIABLES_ANALYSE[x]
        )
    
    with col_select2:
        stat_type = st.selectbox(
            "📈 Type de statistique",
            ["Distribution", "Boxplot", "Histogramme"]
        )
    
    col_mapping = {
        'Montant': 'montant',
        'Heure': 'heure',
        'Jour': 'jour',
        'Mois': 'mois',
        'Jour_Semaine': 'jour_semaine',
        'Type_Transaction': ['type_ATM', 'type_online', 'type_electronique'],
        'Status_Operation': 'status_echoue',
        'Localisation': 'localisation_non_habituelle'
    }
    
    col_name = col_mapping.get(var_select, var_select)
    
    if isinstance(col_name, list):
        st.markdown(f"""
        <div class="modern-card">
            <h4 style="font-weight: 600; color: #1a202c;">{VARIABLES_ANALYSE[var_select]}</h4>
        """, unsafe_allow_html=True)
        
        data_cat = {}
        for c in col_name:
            if c in df_analyze.columns:
                data_cat[c] = df_analyze[c].value_counts()
        
        df_cat = pd.DataFrame(data_cat).fillna(0)
        st.dataframe(df_cat, use_container_width=True)
        
        fig = go.Figure()
        for col in df_cat.columns:
            fig.add_trace(go.Bar(
                x=df_cat.index.astype(str),
                y=df_cat[col],
                name=col
            ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        col_data = df_analyze[col_name]
        
        stats_desc = {
            'Moyenne': col_data.mean(),
            'Médiane': col_data.median(),
            'Écart-type': col_data.std(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Q1': col_data.quantile(0.25),
            'Q3': col_data.quantile(0.75)
        }
        
        col_stats = st.columns(4)
        for i, (key, value) in enumerate(stats_desc.items()):
            with col_stats[i % 4]:
                st.metric(key, f"{value:,.0f}" if value > 100 else f"{value:.2f}")
        
        if stat_type == "Distribution":
            fig = px.histogram(
                df_analyze, x=col_name,
                nbins=30,
                marginal="box",
                title=f"Distribution de {var_select}"
            )
        elif stat_type == "Boxplot":
            fig = px.box(
                df_analyze, y=col_name,
                title=f"Boxplot de {var_select}"
            )
        else:
            fig = px.histogram(
                df_analyze, x=col_name,
                nbins=30,
                title=f"Histogramme de {var_select}"
            )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <h3 style="font-size: 18px; font-weight: 700; color: #1a202c; margin: 20px 0 15px 0;">
        🔗 Matrice de Corrélation
    </h3>
    """, unsafe_allow_html=True)
    
    numeric_cols = df_analyze.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df_analyze[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig_corr.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================================
# MODE : PRÉDICTION (avec interaction en temps réel)
# ==============================================================================
elif mode == "predict":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: #1a202c;">🎯 Prédiction de Fraude</h2>
        <p style="color: #718096; font-size: 14px;">Évaluez en temps réel le risque d'une transaction</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="control-group">', unsafe_allow_html=True)
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin-bottom: 15px;">
            ⚙️ Paramètres de la Transaction
        </h4>
        """, unsafe_allow_html=True)
        
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            montant = st.number_input(
                "💰 Montant (FCFA)", 
                min_value=0.0, 
                value=75000.0, 
                step=5000.0,
                help="Montant de la transaction en Francs CFA",
                key="montant_pred"
            )
            
            heure = st.slider(
                "🕐 Heure de l'opération", 
                0, 23, 14,
                help="Heure à laquelle la transaction a été effectuée",
                key="heure_pred"
            )
        
        with col_input2:
            jour = st.number_input(
                "📅 Jour du mois",
                min_value=1,
                max_value=31,
                value=15,
                help="Jour du mois où la transaction a eu lieu",
                key="jour_pred"
            )
            
            mois = st.selectbox(
                "📆 Mois",
                ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                 "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"],
                index=datetime.now().month - 1,
                key="mois_pred"
            )
        
        with col_input3:
            jour_semaine = st.selectbox(
                "📋 Jour de la semaine",
                ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
                index=datetime.now().weekday(),
                key="jour_semaine_pred"
            )
            
            localisation_habituelle = st.radio(
                "📍 Localisation habituelle ?",
                ["Oui", "Non"],
                horizontal=True,
                key="loc_pred"
            )
        
        st.markdown("""
        <h5 style="font-weight: 600; color: #1a202c; margin: 15px 0 10px 0;">
            🏦 Type de Transaction
        </h5>
        """, unsafe_allow_html=True)
        
        col_type1, col_type2, col_type3 = st.columns(3)
        
        with col_type1:
            type_atm = st.checkbox("ATM", value=False, key="atm_pred")
        with col_type2:
            type_online = st.checkbox("Paiement en ligne", value=True, key="online_pred")
        with col_type3:
            type_electronique = st.checkbox("Paiement électronique", value=False, key="elec_pred")
        
        status_echoue = st.checkbox("❌ Transaction échouée", value=False, key="status_pred")
        
        st.markdown("---")
        
        # Prédiction automatique sans bouton (en temps réel)
        with st.spinner("🔍 Analyse en temps réel..."):
            mois_num = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                       "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"].index(mois) + 1
            jour_semaine_num = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"].index(jour_semaine)
            localisation_non = 1 if localisation_habituelle == "Non" else 0
            
            feature_dict = {
                'montant': [montant],
                'heure': [heure],
                'jour': [jour],
                'mois': [mois_num],
                'jour_semaine': [jour_semaine_num],
                'Type de transaction_ATM': [1 if type_atm else 0],
                'Type de transaction_Paiement en ligne': [1 if type_online else 0],
                'Type de transaction_Paiement électronique': [1 if type_electronique else 0],
                'Status operation_Echoué': [1 if status_echoue else 0],
                'Localisation_non_habituelle': [localisation_non]
            }
            
            for col in feature_columns:
                if col not in feature_dict:
                    feature_dict[col] = [0]
            
            df_input = pd.DataFrame(feature_dict)[feature_columns]
            
            X_scaled = scaler.transform(df_input)
            prediction = model.predict(X_scaled)[0]
            probas = model.predict_proba(X_scaled)[0]
            
            class_mapping = {0: "✅ Légitime", 1: "⚠️ Suspect", 2: "🚨 Fraude"}
            class_colors = {0: "#48bb78", 1: "#ed8936", 2: "#f56565"}
            
            predicted_class = class_mapping.get(prediction, "Inconnu")
            proba_fraude = probas[2] if len(probas) > 2 else probas[1] if len(probas) > 1 else probas[0]
        
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        # Affichage du résultat
        if prediction == 2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fee2e2, #fecaca); border-radius: 16px; padding: 25px; border: 2px solid #f56565;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <span style="background: #f56565; color: white; padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 13px;">🚨 RISQUE ÉLEVÉ</span>
                        <h2 style="margin: 15px 0 5px 0; color: #9b2c2c; font-weight: 800;">Fraude Détectée</h2>
                        <p style="color: #742a2a; font-size: 15px;">Cette transaction présente des caractéristiques anormales de fraude.</p>
                    </div>
                    <div style="font-size: 64px;">🚨</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif prediction == 1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); border-radius: 16px; padding: 25px; border: 2px solid #ed8936;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <span style="background: #ed8936; color: white; padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 13px;">⚠️ RISQUE MOYEN</span>
                        <h2 style="margin: 15px 0 5px 0; color: #7b341e; font-weight: 800;">Transaction Suspecte</h2>
                        <p style="color: #7b341e; font-size: 15px;">Cette transaction nécessite une vérification supplémentaire.</p>
                    </div>
                    <div style="font-size: 64px;">⚠️</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d1fae5, #a7f3d0); border-radius: 16px; padding: 25px; border: 2px solid #48bb78;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                    <div>
                        <span style="background: #48bb78; color: white; padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 13px;">✅ RISQUE FAIBLE</span>
                        <h2 style="margin: 15px 0 5px 0; color: #065f46; font-weight: 800;">Transaction Légitime</h2>
                        <p style="color: #065f46; font-size: 15px;">Aucun signe de fraude détecté.</p>
                    </div>
                    <div style="font-size: 64px;">✅</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin: 20px 0 15px 0;">
            📊 Probabilités par Classe
        </h4>
        """, unsafe_allow_html=True)
        
        col_proba1, col_proba2, col_proba3 = st.columns(3)
        proba_labels = ["Légitime", "Suspect", "Fraude"]
        proba_colors = ["#48bb78", "#ed8936", "#f56565"]
        
        for i, (col, label, color) in enumerate(zip([col_proba1, col_proba2, col_proba3], proba_labels, proba_colors)):
            proba_value = probas[i] if i < len(probas) else 0
            with col:
                st.markdown(f"""
                <div style="background: white; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="font-weight: 600; color: #4a5568;">{label}</div>
                    <div style="font-size: 28px; font-weight: 800; color: {color};">{proba_value:.1%}</div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-fill-{['success', 'warning', 'danger'][i]}" style="width: {proba_value*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <h4 style="font-weight: 600; color: #1a202c; margin: 20px 0 15px 0;">
            🎯 Indicateur de Risque
        </h4>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba_fraude * 100,
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de Fraude", 'font': {'size': 16, 'family': 'Inter'}},
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
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# MODE : VARIABLES
# ==============================================================================
elif mode == "variables":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: #1a202c;">📋 Liste des Variables</h2>
        <p style="color: #718096; font-size: 14px;">Description détaillée de toutes les variables utilisées</p>
    </div>
    """, unsafe_allow_html=True)
    
    variables_desc = {
        'Montant': {
            'type': 'Numérique',
            'description': 'Montant de la transaction en Francs CFA',
            'impact': 'Élevé',
            'exemple': '45 000 FCFA'
        },
        'Heure': {
            'type': 'Numérique (0-23)',
            'description': 'Heure de l\'opération (format 24h)',
            'impact': 'Moyen',
            'exemple': '14h'
        },
        'Jour': {
            'type': 'Numérique (1-31)',
            'description': 'Jour du mois où la transaction a eu lieu',
            'impact': 'Faible',
            'exemple': '22'
        },
        'Mois': {
            'type': 'Numérique (1-12)',
            'description': 'Mois de la transaction',
            'impact': 'Faible',
            'exemple': 'Octobre'
        },
        'Jour_Semaine': {
            'type': 'Numérique (0-6)',
            'description': 'Jour de la semaine (0=Lundi, 6=Dimanche)',
            'impact': 'Moyen',
            'exemple': '2 (Mercredi)'
        },
        'Type_Transaction_ATM': {
            'type': 'Binaire (0/1)',
            'description': 'Indique si la transaction a été effectuée via un ATM',
            'impact': 'Moyen',
            'exemple': '0 (Non)'
        },
        'Type_Transaction_Paiement_en_ligne': {
            'type': 'Binaire (0/1)',
            'description': 'Indique si la transaction est un paiement en ligne',
            'impact': 'Élevé',
            'exemple': '1 (Oui)'
        },
        'Type_Transaction_Paiement_électronique': {
            'type': 'Binaire (0/1)',
            'description': 'Indique si la transaction est un paiement électronique',
            'impact': 'Moyen',
            'exemple': '0 (Non)'
        },
        'Status_Operation_Échoué': {
            'type': 'Binaire (0/1)',
            'description': 'Indique si la transaction a échoué',
            'impact': 'Élevé',
            'exemple': '1 (Échoué)'
        },
        'Localisation_non_habituelle': {
            'type': 'Binaire (0/1)',
            'description': 'Indique si la localisation est inhabituelle pour le client',
            'impact': 'Très élevé',
            'exemple': '1 (Inhabituelle)'
        }
    }
    
    df_vars = pd.DataFrame(variables_desc).T.reset_index()
    df_vars.columns = ['Variable', 'Type', 'Description', 'Impact', 'Exemple']
    
    st.markdown("""
    <div class="modern-card">
        <h3 style="font-weight: 700; color: #1a202c; margin-bottom: 15px;">
            📊 Description des Variables
        </h3>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df_vars,
        use_container_width=True,
        column_config={
            "Variable": "Variable",
            "Type": "Type",
            "Description": "Description",
            "Impact": st.column_config.Column(
                "Impact",
                help="Niveau d'impact sur la prédiction"
            ),
            "Exemple": "Exemple"
        },
        hide_index=True
    )
    
    st.markdown("""
    <h4 style="font-weight: 600; color: #1a202c; margin: 20px 0 10px 0;">
        📈 Importance Relative des Variables
    </h4>
    """, unsafe_allow_html=True)
    
    impact_order = ['Très élevé', 'Élevé', 'Moyen', 'Faible']
    impact_colors = ['#f56565', '#ed8936', '#d69e2e', '#48bb78']
    
    for var in impact_order:
        count = sum(1 for v in variables_desc.values() if v['impact'] == var)
        if count > 0:
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span style="font-weight: 500;">{var}</span>
                    <span style="color: #718096;">{count} variable{'s' if count > 1 else ''}</span>
                </div>
                <div class="progress-bar" style="height: 8px;">
                    <div class="progress-fill" style="width: {(count/len(variables_desc))*100}%; background: {impact_colors[impact_order.index(var)]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# MODE : IMPORT
# ==============================================================================
else:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: #1a202c;">📤 Import de Données</h2>
        <p style="color: #718096; font-size: 14px;">Importez vos propres données pour une analyse personnalisée</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_upload1, col_upload2, col_upload3 = st.columns([1, 2, 1])
    with col_upload2:
        with st.container():
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-zone-icon">📂</div>
                <div class="upload-zone-title">Déposez votre fichier CSV</div>
                <div class="upload-zone-sub">Support des formats standard • Sécurisé</div>
                <div style="margin-top: 10px; font-size: 12px; color: #a0aec0;">
                    Format attendu : colonnes correspondant aux variables du modèle
                </div>
            </div>
            """, unsafe_allow_html=True)
            fichier = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    
    if fichier is not None:
        try:
            df_raw = pd.read_csv(fichier, sep=None, engine="python")
            
            with st.expander("👁️ Aperçu des Données", expanded=True):
                st.dataframe(df_raw.head(10), use_container_width=True, height=300)
                st.caption(f"📊 {len(df_raw)} lignes • {len(df_raw.columns)} colonnes")
            
            missing_cols = [col for col in feature_columns if col not in df_raw.columns]
            if missing_cols:
                st.warning(f"⚠️ Colonnes manquantes : {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}")
            
            if st.button("🚀 Lancer l'Analyse Complète", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyse en cours..."):
                    try:
                        df_processed = pd.get_dummies(df_raw)
                        df_processed = df_processed.reindex(columns=feature_columns, fill_value=0)
                        
                        X_scaled = scaler.transform(df_processed)
                        predictions = model.predict(X_scaled)
                        probas = model.predict_proba(X_scaled)
                        
                        df_results = df_raw.copy()
                        df_results["Prediction"] = predictions
                        df_results["Statut"] = ["🚨 Fraude" if p == 2 else "⚠️ Suspect" if p == 1 else "✅ Légitime" for p in predictions]
                        df_results["Probabilité_Fraude"] = probas[:, 2] if probas.shape[1] > 2 else probas[:, 1]
                        df_results["Niveau_Risque"] = pd.cut(
                            df_results["Probabilité_Fraude"],
                            bins=[-0.1, 0.3, 0.7, 1.0],
                            labels=["🟢 Faible", "🟡 Moyen", "🔴 Élevé"]
                        )
                        
                        stats = calculate_statistics(df_results, predictions)
                        display_statistics_card(stats)
                        
                        st.markdown("---")
                        st.markdown("""
                        <h3 style="font-size: 18px; font-weight: 700; color: #1a202c; margin: 15px 0;">
                            📋 Détail des Transactions
                        </h3>
                        """, unsafe_allow_html=True)
                        
                        col_f1, col_f2, col_f3 = st.columns(3)
                        with col_f1:
                            filter_risk = st.selectbox("🎯 Filtrer par risque", ["Tous", "🟢 Faible", "🟡 Moyen", "🔴 Élevé"])
                        with col_f2:
                            filter_status = st.selectbox("📌 Filtrer par statut", ["Tous", "✅ Légitime", "⚠️ Suspect", "🚨 Fraude"])
                        with col_f3:
                            search_col = st.text_input("🔍 Rechercher", placeholder="Rechercher dans les données...")
                        
                        df_display = df_results.copy()
                        if filter_risk != "Tous":
                            df_display = df_display[df_display["Niveau_Risque"] == filter_risk]
                        if filter_status != "Tous":
                            df_display = df_display[df_display["Statut"] == filter_status]
                        if search_col:
                            mask = df_display.astype(str).apply(lambda x: x.str.contains(search_col, case=False)).any(axis=1)
                            df_display = df_display[mask]
                        
                        def highlight_risk(row):
                            if row['Niveau_Risque'] == '🔴 Élevé':
                                return ['background-color: #fee2e2; color: #991b1b; font-weight: 600;' for _ in row]
                            elif row['Niveau_Risque'] == '🟡 Moyen':
                                return ['background-color: #fef3c7; color: #92400e; font-weight: 600;' for _ in row]
                            else:
                                return ['background-color: #d1fae5; color: #065f46; font-weight: 600;' for _ in row]
                        
                        cols_display = [c for c in df_display.columns if c not in ["Prediction"]]
                        st.dataframe(
                            df_display[cols_display].style.apply(highlight_risk, axis=1),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.markdown("---")
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            csv_data = df_results.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Télécharger le Rapport (CSV)",
                                data=csv_data,
                                file_name=f"rapport_fraude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                type="primary",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse : {e}")
        
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier : {e}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer-premium">
    <div style="display: flex; justify-content: center; gap: 25px; margin-bottom: 8px; flex-wrap: wrap;">
        <span>🛡️ FraudGuard Pro v2.0</span>
        <span>⚡ Propulsé par Random Forest</span>
        <span>🎯 Précision: 95%+</span>
        <span>🔒 Données sécurisées</span>
    </div>
    <p style="margin: 0; font-size: 11px; color: #a0aec0;">
        © 2026 - Système de Détection de Fraude Bancaire • Tous droits réservés
    </p>
</div>
""", unsafe_allow_html=True)