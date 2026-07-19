# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Définir le dossier de travail
WORKDIR /app

# Copier uniquement le fichier des dépendances
COPY requirements.txt .

# Installer les packages Python requis
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]