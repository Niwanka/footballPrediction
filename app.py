import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__, static_url_path='/static')

# Load the model and dummy columns
model, dummy_columns = joblib.load('ensemble_best_model_teams.pkl')

# Define the feature columns
val_X_columns = [
    'MatchWeek', 'HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals',
    'HomeTeamShots', 'AwayTeamShots', 'HomeTeamShotsOnTarget', 'AwayTeamShotsOnTarget',
    'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'year', 'month', 'HomeTeamForm', 'AwayTeamForm'
] + dummy_columns


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        print("Received Data:", data)

       
        HomeTeam = data['HomeTeam']
        AwayTeam = data['AwayTeam']
        year = int(data['year'])
        month = int(data['month'])
        mWeek = int(data['mWeek'])
        HTHG = int(data['HTHG']) 
        HTAG = int(data['HTAG']) 
        HShots = int(data['HShots'])
        AShots = int(data['AShots'])
        HShotsOT = int(data['HShotsOT'])
        AShotsOT = int(data['AShotsOT'])
        BHome = float(data['betHome'])
        BDraw = float(data['betDraw'])
        BAway = float(data['betAway'])
        Hform = int(data['htForm'])
        Aform = int(data['atForm'])

        
        dataframe = pd.DataFrame(0, index=[0], columns=dummy_columns)

        
        dataframe['HalfTimeHomeTeamGoals'] = HTHG
        dataframe['HalfTimeAwayTeamGoals'] = HTAG
        dataframe['year'] = year
        dataframe['month'] = month
        dataframe['MatchWeek'] = mWeek
        dataframe['HomeTeamShots'] = HShots
        dataframe['AwayTeamShots'] = AShots
        dataframe['HomeTeamShotsOnTarget'] = HShotsOT
        dataframe['AwayTeamShotsOnTarget'] = AShotsOT
        dataframe['B365HomeTeam'] = BHome
        dataframe['B365Draw'] = BDraw
        dataframe['B365AwayTeam'] = BAway
        dataframe['HomeTeamForm'] = Hform
        dataframe['AwayTeamForm'] = Aform

        # Set the appropriate dummy columns for home and away teams
        for team_col in dummy_columns:
            if team_col == f'HomeTeam_{HomeTeam}':
                dataframe[team_col] = 1
            elif team_col == f'AwayTeam_{AwayTeam}':
                dataframe[team_col] = 1

       

        # Reorder DataFrame columns to match model input
        dataframe = dataframe[val_X_columns]  # Use val_X_columns to reorder

        # Debugging: print the reordered DataFrame
        
        dataframe.rename(columns={'month': 'Month', 'year': 'Year'}, inplace=True)
        # Make the prediction
        prediction = model.predict_proba(dataframe)

        # Interpret prediction
        response = {
    'home_win_probability': float(prediction[0][0]),  # Convert to float
    'draw_probability': float(prediction[0][1]),       # Convert to float
    'away_win_probability': float(prediction[0][2]),   # Convert to float
    'predicted_outcome': int(prediction[0].argmax())   # Convert to int
}


        # Return the prediction as a JSON response
        return jsonify(response)

    except Exception as e:
        print("Error:", str(e))  # Debugging line to see errors
        return jsonify({'error': str(e)}), 400



@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == "__main__":
    app.run(debug=True)

