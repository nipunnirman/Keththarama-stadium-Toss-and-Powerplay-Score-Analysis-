import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess_data(csv_path):
    # Read preserving raw bytes, replace CR with LF
    with open(csv_path, 'rb') as f:
        content = f.read().replace(b'\r', b'\n')
    with open('temp.csv', 'wb') as f:
        f.write(content)
        
    df = pd.read_csv('temp.csv')
    df.columns = df.columns.str.strip()
    
    # Exclude TIED matches if any or missing Winner
    df = df[df['Winner'] != 'TIED']
    df = df.dropna(subset=['Winner'])
    
    innings_data = []
    
    for idx, row in df.iterrows():
        try:
            teams = row['Match'].split(' vs ')
            if len(teams) != 2:
                continue
            
            t1, t2 = teams[0].strip(), teams[1].strip()
            winner = row['Winner'].strip() if pd.notna(row['Winner']) else None
            loser = t1 if winner == t2 else (t2 if winner == t1 else None)
            
            if not winner or not loser:
                continue
                
            toss_winner = row['Toss_Win'].strip() if pd.notna(row['Toss_Win']) else None
            decision = row['Decision'].strip() if pd.notna(row['Decision']) else None
            
            winner_pp = row['Winner_PP']
            loser_pp = row['Loser_PP']
            
            # Extract who batted first
            if decision == 'Bat':
                bat_first = toss_winner
                bat_second = t1 if toss_winner == t2 else t2
            else:
                bat_second = toss_winner
                bat_first = t1 if toss_winner == t2 else t2
            
            # Prepare row for Team 1 (Inning 1 or 2)
            for team in [t1, t2]:
                is_winner = (team == winner)
                opponent = t2 if team == t1 else t1
                innings = 1 if team == bat_first else 2
                toss_won = 1 if team == toss_winner else 0
                
                toss_decision = decision if toss_won == 1 else (
                    'Bowl' if decision == 'Bat' else 'Bat' # opposite perspective
                )
                
                pp_score = winner_pp if is_winner else loser_pp
                
                # We want to predict PP Score given these details
                innings_data.append({
                    'Team': team,
                    'Opponent': opponent,
                    'Innings': innings,
                    'Toss_Won': toss_won,
                    'Team_Toss_Decision': 'Bat' if toss_won and decision == 'Bat' or not toss_won and decision == 'Bowl' else 'Bowl',
                    'PP_Score': float(pp_score)
                })
        except Exception as e:
            pass

    innings_df = pd.DataFrame(innings_data)
    return innings_df

def build_model(df):
    # Encoding Categorical Variables
    le_team = LabelEncoder()
    le_opp = LabelEncoder()
    le_decision = LabelEncoder()

    # Fit and transform
    df['Team_encoded'] = le_team.fit_transform(df['Team'])
    df['Opponent_encoded'] = le_opp.fit_transform(df['Opponent'])
    df['Decision_encoded'] = le_decision.fit_transform(df['Team_Toss_Decision'])

    features = ['Team_encoded', 'Opponent_encoded', 'Innings', 'Toss_Won', 'Decision_encoded']
    X = df[features]
    y = df['PP_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model Evaluation on Test Data:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    return model, le_team, le_opp, le_decision, mae

if __name__ == '__main__':
    csv_file = 'keththarama.csv'
    df = load_and_preprocess_data(csv_file)
    print(f"Total innings parsed for training: {len(df)}")
    
    if len(df) > 0:
        model, le_team, le_opp, le_decision, mae = build_model(df)
        
        # Save models and encoders
        joblib.dump(model, 'pp_score_model.pkl')
        joblib.dump(le_team, 'le_team.pkl')
        joblib.dump(le_opp, 'le_opp.pkl')
        joblib.dump(le_decision, 'le_decision.pkl')
        
        print("\nModel saved to 'pp_score_model.pkl' successfully!")
    else:
        print("Not enough data to train the model. Please check the dataset format.")
