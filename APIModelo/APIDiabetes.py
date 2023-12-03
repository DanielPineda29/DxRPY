from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Cargar el modelo y el escalador
logreg = joblib.load('modelo_logreg.joblib')
scaler = joblib.load('escalador.joblib')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Transformar los datos según el escalador utilizado durante el entrenamiento
    features = np.array([
        data['cholesterol'],
        data['glucose'],
        data['hdlChol'],
        data['cholHdlRatio'],
        data['age'],
        data['gender'],
        data['height'],
        data['weight'],
        data['bmi'],
        data['systolicBP'],
        data['diastolicBP'],
        data['waist'],
        data['hip'],
        data['waistHipRatio'],
    ])

    features_scaled = scaler.transform(features.reshape(1, -1))
    print('Datos normalizados:', features_scaled)

    # Realizar la predicción usando el modelo entrenado
    prediction = logreg.predict(features_scaled)
    
    # Después de la predicción, imprime las probabilidades de cada clase
    prediction_probabilities = logreg.predict_proba(features_scaled)
    print('Probabilidades:', prediction_probabilities)
    
    # Ajusta el umbral según tus necesidades
    threshold = 0.5  # Puedes ajustar este valor según sea necesario
    
    # Mapea la predicción de nuevo a las categorías originales con el nuevo umbral
    prediction_label = "Diabetes" if prediction_probabilities[0, 1] > threshold else "No Diabetes"
    print('Resultado: ', prediction_label)
    
    # Mapear la predicción de nuevo a las categorías originales
    #prediction_label = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    #print('Resultado: ', prediction_label)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
