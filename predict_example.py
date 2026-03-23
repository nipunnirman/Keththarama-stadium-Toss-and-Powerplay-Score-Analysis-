import joblib
import pandas as pd

def predict_pp_score(team, opponent, innings, toss_won, team_toss_decision):
    # Load model and encoders
    model = joblib.load('pp_score_model.pkl')
    le_team = joblib.load('le_team.pkl')
    le_opp = joblib.load('le_opp.pkl')
    le_decision = joblib.load('le_decision.pkl')
    
    # Encode inputs
    team_encoded = le_team.transform([team])[0]
    opp_encoded = le_opp.transform([opponent])[0]
    decision_encoded = le_decision.transform([team_toss_decision])[0]
    
    # Create feature dataframe
    features = pd.DataFrame([{
        'Team_encoded': team_encoded,
        'Opponent_encoded': opp_encoded,
        'Innings': innings,
        'Toss_Won': toss_won,
        'Decision_encoded': decision_encoded
    }])
    
    # Predict
    predicted_score = model.predict(features)[0]
    return predicted_score

if __name__ == '__main__':
    # Example Prediction
    print("--- Example Prediction ---")
    team = 'SL'
    opponent = 'IND'
    innings = 1
    toss_won = 1
    toss_decision = 'Bat'
    
    predicted_pp_score = predict_pp_score(team, opponent, innings, toss_won, toss_decision)
    print(f"When {team} wins the toss and decides to {toss_decision} against {opponent} in Inning {innings}:")
    print(f"Predicted Powerplay Score: {predicted_pp_score:.2f} runs")
