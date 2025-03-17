# Utiliser une image de base Python
FROM python:3.11.8

# Définir le répertoire de travail
WORKDIR /Docker_training

# Copier le fichier requirements.txt et installer les dépendances
COPY requirements.txt /Docker_training/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le fichier script d'entraînement dans le conteneur
COPY Classification.py .
COPY ecg.csv .

# Exécuter le script d'entraînement
CMD ["python", "Classification.py"]

