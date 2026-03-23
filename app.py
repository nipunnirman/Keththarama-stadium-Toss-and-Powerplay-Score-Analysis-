from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pp_score_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    le_team = joblib.load(os.path.join(BASE_DIR, 'le_team.pkl'))
    le_opp = joblib.load(os.path.join(BASE_DIR, 'le_opp.pkl'))
    le_decision = joblib.load(os.path.join(BASE_DIR, 'le_decision.pkl'))
    print("Model and Encoders loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Prediction model is not loaded properly'}), 500

    try:
        data = request.json
        team = data.get('team')
        opponent = data.get('opponent')
        innings = int(data.get('innings', 1))
        toss_won = int(data.get('toss_won', 0))
        toss_decision = data.get('toss_decision', 'Bat')

        if not team or not opponent:
             return jsonify({'error': 'Team and Opponent are required'}), 400

        # Encode inputs safely
        try:
            team_encoded = le_team.transform([team])[0]
        except ValueError:
            # handle unseen labels gracefully
            team_encoded = 0 
        
        try:
            opp_encoded = le_opp.transform([opponent])[0]
        except ValueError:
             opp_encoded = 0
             
        try:
            decision_encoded = le_decision.transform([toss_decision])[0]
        except ValueError:
             decision_encoded = 0

        features = pd.DataFrame([{
            'Team_encoded': team_encoded,
            'Opponent_encoded': opp_encoded,
            'Innings': innings,
            'Toss_Won': toss_won,
            'Decision_encoded': decision_encoded
        }])

        prediction = model.predict(features)[0]

        return jsonify({'predicted_pp_score': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Make templates dir if missing
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5001)
