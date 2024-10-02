import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__, static_url_path='/static')

# Load the model and dummy columns
model, dummy_columns = joblib.load('ensemble_best_model_teams.pkl')

# Define the feature columns
val_X_columns = [
    'MatchWeek','HalfTimeHomeTeamGoals', 'HalfTimeAwayTeamGoals', 
    'HomeTeamShots', 'AwayTeamShots','HomeTeamShotsOnTarget' , 'AwayTeamShotsOnTarget' ,    
    'B365HomeTeam', 'B365Draw', 'B365AwayTeam', 'year', 'month','HomeTeamForm', 'AwayTeamForm'
] + dummy_columns


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    print("Received Data:", request.data)

    try:
        # Extract and convert form data
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
        BHome = float(request.form.get('betHome', 0))  # Provide a default value if not present
        BDraw = float(request.form.get('betDraw', 0))
        BAway = float(request.form.get('betAway', 0))
        Hform = int(data['htForm'])
        Aform = int(data['atForm'])

        # Create a DataFrame for the input data
        dataframe = pd.DataFrame(0, index=[0], columns=dummy_columns)

        # Fill the DataFrame with input data
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
        dataframe = dataframe[val_X_columns]

        # Make the prediction
        prediction = model.predict_proba(dataframe)

        # Interpret prediction
        response = {
            'home_win_probability': prediction[0][0],
            'draw_probability': prediction[0][1],
            'away_win_probability': prediction[0][2],
            'predicted_outcome': prediction[0].argmax()  # 0: Home Win, 1: Draw, 2: Away Win
        }

        # Return the prediction as a JSON response
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)
