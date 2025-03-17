from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle et le scaler au démarrage de l'application
def load_model_and_scaler():
    
        model = joblib.load('model.pkl')
        #scaler = joblib.load('scaler.pkl')
        return model

model = load_model_and_scaler()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON de la requête
        content = request.json
        
        if not content or not isinstance(content, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object with features."}), 400

        # Convertir les features en tableau NumPy
        features = np.array(list(content.values())).reshape(1, -1)
        
        # Vérifier si le nombre de features correspond à ce qu'attend le modèle
        #expected_features = scaler.n_features_in_
        #if features.shape[1] != expected_features:
        #    return jsonify({
        #        "error": f"Invalid number of features. Expected {expected_features}, got {features.shape[1]}"
        #    }), 400

        # Normaliser les données
        #features_scaled = scaler.transform(features)

        # Faire la prédiction
        prediction = model.predict(features)

        # Traiter la sortie en fonction du type de classification
        if prediction.shape[1] == 1:  # Classification binaire
            predicted_class = int(prediction[0][0] > 0.5)
            probability = float(prediction[0][0])
        else:  # Classification multi-classes
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            probability = float(np.max(prediction))

        # Préparer et renvoyer la réponse
        response = {
            "predicted_class": predicted_class,
            "probability": probability
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)