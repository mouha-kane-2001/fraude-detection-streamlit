import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Créer le dossier 'model' s'il n'existe pas
os.makedirs("model", exist_ok=True)

# 1. Chargement des données
print("Chargement des données...")
chemin = "data/transactions.csv"
df = pd.read_csv(chemin, sep=";")

# Nettoyage des espaces cachés dans les noms de colonnes
df.columns = df.columns.str.strip()

# Supprimer les lignes où la cible est absente à la source
df = df.dropna(subset=["Target"])

# 2. Préparation des données (Preprocessing)
df["Date"] = pd.to_datetime(df["Date"])
df["heure"] = df["Date"].dt.hour
df["jour"] = df["Date"].dt.day
df["mois"] = df["Date"].dt.month
df["jour_semaine"] = df["Date"].dt.dayofweek

# Uniformisation textuelle de la cible
df["Target"] = df["Target"].astype(str).str.strip().str.capitalize()

# Mapping parfait à 3 classes : Normal=0, Suspect=1, Fraude=2
df["Target"] = df["Target"].map({"Normal": 0, "Suspect": 1, "Fraude": 2})

# Nettoyage de sécurité si une valeur bizarre s'est glissée dans le CSV
df = df.dropna(subset=["Target"])
df["Target"] = df["Target"].astype(int)

# On supprime les colonnes d'identifiants uniques non prédictives
df = df.drop(columns=["ID Clients", "Numero de compte", "Identifiant operation", "Date"])

# Encodage des variables catégorielles (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=["Type de transaction", "Status operation", "Localisation"], dtype=int)

# 3. Séparation features (X) / cible (y)
X = df_encoded.drop("Target", axis=1)   
y = df_encoded["Target"]

# Sauvegarde des colonnes pour l'application Streamlit (vital pour que l'app ne plante pas)
joblib.dump(X.columns.tolist(), "model/columns.pkl")

# 4. Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Division entraînement / test avec stratification multi-classes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Entraînement du modèle multiclasse (Poids ajustés automatiquement pour gérer le déséquilibre)
print("Entraînement du modèle Random Forest (3 classes)...")
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    class_weight="balanced", 
    random_state=42
)
model.fit(X_train, y_train)

# 7. Évaluation complète des 3 classes
y_pred = model.predict(X_test)
print("\nRapport de classification complet :")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Fraude"]))

# 8. Sauvegarde définitive des fichiers de production
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("\n[SUCCÈS] Modèle multiclasse et Scaler sauvegardés et prêts pour l'application Streamlit !")