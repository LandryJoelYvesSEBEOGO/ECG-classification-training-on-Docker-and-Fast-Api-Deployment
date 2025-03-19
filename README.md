# Classification des ECG

Ce projet est une application de classification des ECG (électrocardiogrammes) utilisant l'apprentissage profond. Il est composé de deux parties principales : l'entraînement du modèle et l'API de prédiction.

## Architecture du Projet

Le projet est structuré en deux services Docker principaux :

1. **Service d'Entraînement** (`Docker_training/`)
   - Contient le script d'entraînement du modèle
   - Utilise TensorFlow pour créer et entraîner un réseau de neurones
   - Sauvegarde le modèle entraîné

2. **Service API** (`app/`)
   - Fournit une API REST pour les prédictions
   - Utilise Flask pour exposer les endpoints
   - Charge le modèle entraîné pour faire des prédictions

## Prérequis

- Docker
- Docker Compose

## Installation

1. Cloner le repository
2. Exécuter la commande suivante pour démarrer les services :
```bash
docker-compose up --build
```

## Structure des Données

Le modèle est entraîné sur un dataset d'ECG (`ecg.csv`) avec les caractéristiques suivantes :
- Features : F1, F2, ..., Fn (caractéristiques extraites des ECG)
- Label : classe de l'ECG

## Modèle

Le modèle est un réseau de neurones avec l'architecture suivante :
- Couche d'entrée : dimension adaptée aux features
- Première couche cachée : 128 neurones avec activation ReLU
- Deuxième couche cachée : 64 neurones avec activation ReLU
- Couche de sortie : 1 neurone avec activation sigmoïde (classification binaire) ou softmax (classification multi-classes)

## API Endpoints

### 1. Prédiction
- **URL**: `/predict`
- **Méthode**: POST
- **Description**: Fait une prédiction sur de nouvelles données ECG
- **Format de la requête**: JSON contenant les features
- **Format de la réponse**: 
```json
{
    "predicted_class": int,
    "probability": float
}
```

### 2. Health Check
- **URL**: `/health`
- **Méthode**: GET
- **Description**: Vérifie l'état de l'API
- **Format de la réponse**:
```json
{
    "status": "healthy"
}
```

## Utilisation

1. L'API est accessible sur le port 5000
2. Pour faire une prédiction, envoyer une requête POST à `http://localhost:5000/predict` avec les features au format JSON
3. Le modèle retournera la classe prédite et la probabilité associée

## Performance

Le modèle est évalué sur un ensemble de test avec :
- Matrice de confusion
- Rapport de classification détaillé
- Précision globale

## Maintenance

- Le modèle est automatiquement entraîné au démarrage du service d'entraînement
- Les prédictions sont effectuées en temps réel via l'API
- Le système est conteneurisé pour une déploiement facile et une isolation des dépendances 