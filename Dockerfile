# Base Python image
FROM python:3.10-slim

# Créer un dossier dans le conteneur
WORKDIR /app

# Copier tout le contenu de ton projet dans ce dossier
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de Flask
EXPOSE 8000

# Lancer ton app Flask
CMD ["python", "main.py"]
