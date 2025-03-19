# ECG Classification in a Docker Container and API Implementation

This project is an ECG (Electrocardiogram) classification application using deep learning. It consists of two main phases: the first phase involves training the model in a Docker container, and after training, we move to the second phase which involves implementing an API to perform various predictions.

## Project Architecture

The project is structured into two main Docker services:

1. **Training Service** (`Docker_training/`)
   - Contains the model training script
   - Uses TensorFlow to create and train a neural network
   - Saves the trained model

2. **API Service** (`app/`)
   - Provides a REST API for predictions
   - Uses Flask to expose the endpoints
   - Loads the trained model to make predictions

## Prerequisites

- Docker
- Docker Compose

## Installation

1. Clone the repository
2. Run the following command to start the services:
```bash
docker-compose up --build
```

## Structure des Données

The model is trained on an ECG dataset (ecg.csv) with the following features:

- Features: F1, F2, ..., Fn (features extracted from ECGs)
- Label: ECG class

## Modèle

The model is a neural network with the following architecture:

- Input layer: dimension adapted to the features
- First hidden layer: 128 neurons with ReLU activation
- Second hidden layer: 64 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation (binary classification) or softmax (multi-class classification)

## API Endpoints

### 1. Prédiction
- **URL**: `/predict`
- **Méthode**: POST
- **Description**: Fait une prédiction sur de nouvelles données ECG
- **Format de la requête**: JSON containing the features
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
- **Description**: Makes a prediction on new ECG data
- **Format de la réponse**:
```json
{
    "status": "healthy"
}
```

## Usage

1. The API is accessible on port 5000
2. To make a prediction, send a POST request to http://localhost:5000/predict with the features in JSON format
3. The model will return the predicted class and the associated probability

## Performance

The model is evaluated on a test set with:

- Confusion matrix
- Detailed classification report
- Overall accuracy

## Maintenance

- The model is automatically trained at the start of the training service
- Predictions are made in real-time via the API
- The system is containerized for easy deployment and dependency isolation
