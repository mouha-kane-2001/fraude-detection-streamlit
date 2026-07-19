import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire", page_icon="🏦", layout="wide"
)

# --- Chargement du modèle et du scaler (mis en cache pour optimiser les performances) ---
@st.cache_resource
def load_model():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler


try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        "❌ Fichiers de modèle ou de scaler introuvables. Veuillez d'abord exécuter le script d'entraînement."
    )
    st.stop()

# --- En-tête ---
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown(
    "Analysez une transaction isolée ou un lot complet pour détecter instantanément les risques de fraude."
)

# --- Menu latéral ---
st.sidebar.header("Paramètres de l'application")
mode = st.sidebar.radio(
    "Mode d'analyse", ["Transaction unique", "Fichier CSV (lot)"]
)

# ==========================================
# MODE 1 : Analyse d'une transaction unique
# ==========================================
if mode == "Transaction unique":
    st.subheader("Saisie manuelle d'une transaction")

    col1, col2 = st.columns(2)
    with col1:
        montant = st.number_input(
            "Montant de la transaction (FCFA)", min_value=0.0, value=50000.0
        )
        heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
    with col2:
        localisation_habituelle = st.selectbox(
            "Localisation habituelle du client ?", ["Oui", "Non"]
        )
        canal = st.selectbox(
            "Canal de transaction", ["Mobile Money", "Carte", "Virement", "ATM"]
        )

    if st.button("Analyser la transaction", type="primary"):
        # Note : Le vecteur de features ci-dessous est un exemple à aligner selon la structure exacte de X
        # Simulation d'un vecteur de features correspondant au format attendu par le scaler
        localisation_non = 1 if localisation_habituelle == "Non" else 0

        # Récupération du nombre de features requises par le scaler pour éviter les erreurs de dimension
        num_features_attendues = scaler.n_features_in_
        features = np.zeros((1, num_features_attendues))

        # Assignation des premières variables (A ajuster selon l'ordre exact de vos données)
        features[0, 0] = montant
        features[0, 1] = heure
        features[0, 2] = localisation_non

        # Transformation et Prédiction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]

        st.divider()

        # Affichage des résultats
        if prediction == 1:
            st.error(
                f"⚠️ **Transaction suspecte** — Probabilité de fraude : {proba:.1%}"
            )
        else:
            st.success(
                f"✅ **Transaction légitime** — Probabilité de fraude : {proba:.1%}"
            )

        st.progress(float(proba))

# ==========================================
# MODE 2 : Analyse par lot (Fichier CSV)
# ==========================================
else:
    st.subheader("Analyse par lot via fichier CSV")
    fichier = st.file_uploader(
        "Déposez le fichier CSV des transactions à analyser", type=["csv"]
    )

    if fichier is not None:
        df = pd.read_csv(fichier)
        st.write("### Aperçu des données importées :", df.head())

        if st.button("Lancer l'analyse du lot", type="primary"):
            try:
                # Normalisation et prédiction sur l'ensemble du fichier
                X_scaled = scaler.transform(df)
                df["prediction"] = model.predict(X_scaled)
                df["probabilite_fraude"] = model.predict_proba(X_scaled)[:, 1]

                nb_fraudes = int(df["prediction"].sum())

                # Affichage des métriques de synthèse
                if nb_fraudes > 0:
                    st.warning(
                        f"⚠️ **{nb_fraudes}** transaction(s) suspecte(s) détectée(s) sur un total de {len(df)} lignes."
                    )
                else:
                    st.success(
                        f"✅ Aucune transaction suspecte détectée sur les {len(df)} lignes."
                    )

                # Coloration conditionnelle des lignes frauduleuses (fond rouge clair)
                st.write("### Résultats de l'analyse :")

                def colorier_fraude(row):
                    return (
                        ["background-color: #ffcccc"] * len(row)
                        if row["prediction"] == 1
                        else [""] * len(row)
                    )

                st.dataframe(df.style.apply(colorier_fraude, axis=1))

                # Exportation des résultats
                csv_export = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Télécharger le rapport de fraude (CSV)",
                    data=csv_export,
                    file_name="resultats_analyse_fraude.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(
                    f"Erreur lors du traitement du fichier. Assurez-vous que les colonnes correspondent exactement au modèle. Détails : {e}"
                )

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")