import flask
from flask import request, jsonify, send_from_directory, session
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from datetime import timedelta

# --- 1. Initialization (Unchanged) ---
app = flask.Flask(__name__)
app.secret_key = 'your_very_secret_key_change_this'
app.permanent_session_lifetime = timedelta(days=7)

# --- Database Setup (Unchanged) ---
DB_PATH = '../cricket_final_data_by_format.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            player TEXT, venue TEXT, opposition TEXT, match_format TEXT,
            predicted_runs_bin TEXT, predicted_wickets_bin TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()
    print("--> User and History tables initialized.")

# --- 2. Load All Assets on Startup (Unchanged) ---
print("--- Server is starting up. Loading all assets... ---")
init_db()

try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    batting_df_full = pd.read_sql_query("SELECT * FROM batting_stats", conn)
    bowling_df_full = pd.read_sql_query("SELECT * FROM bowling_stats", conn)
    batting_df_full['date'] = pd.to_datetime(batting_df_full['date'])
    bowling_df_full['date'] = pd.to_datetime(bowling_df_full['date'])
    conn.close()
    print("--> Database loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load database. Error: {e}")
    batting_df_full = pd.DataFrame()
    bowling_df_full = pd.DataFrame()

MODELS = {"batting": {}, "bowling": {}}
FORMATS = ["t20", "odi", "test"]

for format_type in FORMATS:
    try:
        MODELS['batting'][format_type] = {
            'model': joblib.load(f'../models/{format_type}_batting_classifier.joblib'),
            'columns': joblib.load(f'../models/{format_type}_batting_columns.joblib')
        }
        MODELS['bowling'][format_type] = {
            'model': joblib.load(f'../models/{format_type}_bowling_classifier.joblib'),
            'columns': joblib.load(f'../models/{format_type}_bowling_columns.joblib')
        }
        print(f"--> Models for {format_type.upper()} loaded successfully.")
    except FileNotFoundError:
        print(f"--> WARNING: Model files for {format_type.upper()} not found.")

ALL_PLAYERS = sorted(list(pd.concat([batting_df_full['player'], bowling_df_full['player']]).unique()))
ALL_VENUES = sorted(list(pd.concat([batting_df_full['venue'], bowling_df_full['venue']]).unique()))
match_teams = batting_df_full.groupby('match_id')['team'].unique().apply(list).to_dict()
all_teams_set = set()
for teams in match_teams.values():
    if len(teams) == 2:
        all_teams_set.add(teams[0])
        all_teams_set.add(teams[1])
ALL_TEAMS = sorted(list(all_teams_set))
print("--> Autocomplete directories created.")
print("--- Server startup complete. Ready for requests. ---")


# --- 3. API Endpoints ---
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# --- AUTHENTICATION API (Unchanged) ---
@app.route('/api/register', methods=['POST'])
def register():
    # This code is identical to your provided file
    data = request.get_json()
    email = data['email']
    password = data['password']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user:
        return jsonify({'success': False, 'message': 'Email already exists.'}), 409
    password_hash = generate_password_hash(password)
    cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'User registered successfully.'})

@app.route('/api/login', methods=['POST'])
def login():
    # This code is identical to your provided file
    data = request.get_json()
    email = data['email']
    password = data['password']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if user and check_password_hash(user[2], password):
        session.permanent = True
        session['user_id'] = user[0]
        session['user_email'] = user[1]
        return jsonify({'success': True, 'email': user[1]})
    return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})
    
@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'email': session['user_email']})
    return jsonify({'logged_in': False})

# --- PREDICTION AND HISTORY API (Unchanged) ---
@app.route('/api/predict', methods=['POST'])
def predict():
    # This code is identical to your provided file
    data = request.get_json()
    player_name = data['player']
    venue = data.get('venue', '').strip()
    opposition = data.get('opposition', '').strip()
    match_format = data['match_format']
    format_key = match_format.lower()
    predicted_runs = f"Model for {match_format} not available."
    if format_key in MODELS['batting']:
        try:
            b_model_info = MODELS['batting'][format_key]
            b_model, b_columns = b_model_info['model'], b_model_info['columns']
            player_history = batting_df_full[(batting_df_full['player'] == player_name) & (batting_df_full['match_type'] == match_format)].sort_values(by='date')
            if player_history.empty: predicted_runs = f"No {match_format} batting history found."
            else:
                career_stats=player_history.iloc[-1]
                form_avg_last_10=player_history['runs'].tail(10).mean()
                career_avg=np.where(career_stats.get('career_innings',0)>0,career_stats.get('career_runs',0)/career_stats.get('career_innings'),0)
                career_sr=np.where(career_stats.get('career_balls_faced',0)>0,(career_stats.get('career_runs',0)/career_stats.get('career_balls_faced',0))*100,0)
                input_template=pd.DataFrame(columns=b_columns,index=[0]).fillna(0)
                input_template.loc[0,'career_avg']=career_avg; input_template.loc[0,'career_sr']=career_sr; input_template.loc[0,'career_innings']=career_stats.get('career_innings',0); input_template.loc[0,'form_avg_last_10']=form_avg_last_10
                player_col,venue_col,opposition_col=f'player_{player_name}',f'venue_{venue}',f'against_team_{opposition}'
                if player_col in input_template.columns:input_template.loc[0,player_col]=1
                if venue and venue_col in input_template.columns:input_template.loc[0,venue_col]=1
                if opposition and opposition_col in input_template.columns:input_template.loc[0,opposition_col]=1
                prediction=b_model.predict(input_template)
                predicted_runs=prediction[0]
        except Exception as e: predicted_runs = f"An error occurred: {str(e)}"
    predicted_wickets = f"Model for {match_format} not available."
    if format_key in MODELS['bowling']:
        try:
            bw_model_info=MODELS['bowling'][format_key]
            bw_model, bw_columns=bw_model_info['model'], bw_model_info['columns']
            player_bowling_history=bowling_df_full[(bowling_df_full['player']==player_name)&(bowling_df_full['match_type']==match_format)].sort_values(by='date')
            if player_bowling_history.empty: predicted_wickets=f"No {match_format} bowling history found."
            else:
                career_stats_bw=player_bowling_history.iloc[-1]
                form_wickets_last_10=player_bowling_history['wickets'].tail(10).mean()
                career_bowling_avg=np.where(career_stats_bw.get('career_wickets',0)>0,career_stats_bw.get('career_runs_conceded',0)/career_stats_bw.get('career_wickets'),0)
                career_bowling_sr=np.where(career_stats_bw.get('career_wickets',0)>0,career_stats_bw.get('career_balls_bowled',0)/career_stats_bw.get('career_wickets'),0)
                input_template_bw=pd.DataFrame(columns=bw_columns,index=[0]).fillna(0)
                input_template_bw.loc[0,'career_bowling_avg']=career_bowling_avg; input_template_bw.loc[0,'career_bowling_sr']=career_bowling_sr; input_template_bw.loc[0,'form_wickets_last_10']=form_wickets_last_10
                player_col_bw,venue_col_bw,opposition_col_bw=f'player_{player_name}',f'venue_{venue}',f'against_team_{opposition}'
                if player_col_bw in input_template_bw.columns:input_template_bw.loc[0,player_col_bw]=1
                if venue and venue_col_bw in input_template_bw.columns:input_template_bw.loc[0,venue_col_bw]=1
                if opposition and opposition_col_bw in input_template_bw.columns:input_template_bw.loc[0,opposition_col_bw]=1
                prediction_bw=bw_model.predict(input_template_bw)
                predicted_wickets=prediction_bw[0]
        except Exception as e: predicted_wickets = f"An error occurred: {str(e)}"
    if 'user_id' in session:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO history (user_id, player, venue, opposition, match_format, predicted_runs_bin, predicted_wickets_bin) VALUES (?, ?, ?, ?, ?, ?, ?)', (session['user_id'], player_name, venue, opposition, match_format, str(predicted_runs), str(predicted_wickets)))
        conn.commit()
        conn.close()
    return jsonify({'predicted_runs_bin': str(predicted_runs), 'predicted_wickets_bin': str(predicted_wickets)})

@app.route('/api/history', methods=['GET'])
def get_history():
    # This code is identical to your provided file
    if 'user_id' not in session: return jsonify({'success': False, 'message': 'Not logged in'}), 401
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    history_data = cursor.execute("SELECT * FROM history WHERE user_id = ? ORDER BY timestamp DESC", (session['user_id'],)).fetchall()
    conn.close()
    history_list = [dict(row) for row in history_data]
    return jsonify({'success': True, 'history': history_list})

# --- OTHER APIS (Unchanged) ---
@app.route('/api/search_terms', methods=['GET'])
def get_search_terms():
    return jsonify({ 'players': ALL_PLAYERS, 'venues': ALL_VENUES, 'teams': ALL_TEAMS })

@app.route('/api/recent_stats', methods=['POST'])
def get_recent_stats():
    # This code is identical to your provided file
    data = request.get_json()
    player_name = data['player']
    match_format = data['match_format']
    player_batting = batting_df_full[(batting_df_full['player'] == player_name) & (batting_df_full['match_type'] == match_format)]
    player_bowling = bowling_df_full[(bowling_df_full['player'] == player_name) & (bowling_df_full['match_type'] == match_format)]
    merged_stats = pd.merge(player_batting[['match_id', 'date', 'runs']], player_bowling[['match_id', 'date', 'wickets']], on=['match_id', 'date'], how='outer')
    recent_stats = merged_stats.sort_values(by='date', ascending=False).head(5)
    recent_stats = recent_stats.fillna(0)
    stats_list = recent_stats.to_dict(orient='records')
    return jsonify(stats_list)

# --- BEST 11 API (This is the new, updated part) ---
def _predict_player_performance(player_name, venue, opposition, match_format):
    """Helper function to run prediction for a single player."""
    # This function contains the same logic as your /api/predict route
    format_key = match_format.lower()
    predicted_runs = "N/A"
    if format_key in MODELS['batting']:
        try:
            # Batting prediction logic...
            b_model_info = MODELS['batting'][format_key]
            b_model, b_columns = b_model_info['model'], b_model_info['columns']
            player_history = batting_df_full[(batting_df_full['player'] == player_name) & (batting_df_full['match_type'] == match_format)].sort_values(by='date')
            if not player_history.empty:
                career_stats=player_history.iloc[-1]
                form_avg_last_10=player_history['runs'].tail(10).mean()
                career_avg=np.where(career_stats.get('career_innings',0)>0,career_stats.get('career_runs',0)/career_stats.get('career_innings'),0)
                career_sr=np.where(career_stats.get('career_balls_faced',0)>0,(career_stats.get('career_runs',0)/career_stats.get('career_balls_faced',0))*100,0)
                input_template=pd.DataFrame(columns=b_columns,index=[0]).fillna(0)
                input_template.loc[0,'career_avg']=career_avg; input_template.loc[0,'career_sr']=career_sr; input_template.loc[0,'career_innings']=career_stats.get('career_innings',0); input_template.loc[0,'form_avg_last_10']=form_avg_last_10
                player_col,venue_col,opposition_col=f'player_{player_name}',f'venue_{venue}',f'against_team_{opposition}'
                if player_col in input_template.columns:input_template.loc[0,player_col]=1
                if venue and venue_col in input_template.columns:input_template.loc[0,venue_col]=1
                if opposition and opposition_col in input_template.columns:input_template.loc[0,opposition_col]=1
                prediction=b_model.predict(input_template)
                predicted_runs=str(prediction[0])
        except Exception: predicted_runs = "Error"
    predicted_wickets = "N/A"
    if format_key in MODELS['bowling']:
        try:
            # Bowling prediction logic...
            bw_model_info=MODELS['bowling'][format_key]
            bw_model, bw_columns=bw_model_info['model'], bw_model_info['columns']
            player_bowling_history=bowling_df_full[(bowling_df_full['player']==player_name)&(bowling_df_full['match_type']==match_format)].sort_values(by='date')
            if not player_bowling_history.empty:
                career_stats_bw=player_bowling_history.iloc[-1]
                form_wickets_last_10=player_bowling_history['wickets'].tail(10).mean()
                career_bowling_avg=np.where(career_stats_bw.get('career_wickets',0)>0,career_stats_bw.get('career_runs_conceded',0)/career_stats_bw.get('career_wickets'),0)
                career_bowling_sr=np.where(career_stats_bw.get('career_wickets',0)>0,career_stats_bw.get('career_balls_bowled',0)/career_stats_bw.get('career_wickets'),0)
                input_template_bw=pd.DataFrame(columns=bw_columns,index=[0]).fillna(0)
                input_template_bw.loc[0,'career_bowling_avg']=career_bowling_avg; input_template_bw.loc[0,'career_bowling_sr']=career_bowling_sr; input_template_bw.loc[0,'form_wickets_last_10']=form_wickets_last_10
                player_col_bw,venue_col_bw,opposition_col_bw=f'player_{player_name}',f'venue_{venue}',f'against_team_{opposition}'
                if player_col_bw in input_template_bw.columns:input_template_bw.loc[0,player_col_bw]=1
                if venue and venue_col_bw in input_template_bw.columns:input_template_bw.loc[0,venue_col_bw]=1
                if opposition and opposition_col_bw in input_template_bw.columns:input_template_bw.loc[0,opposition_col_bw]=1
                prediction_bw=bw_model.predict(input_template_bw)
                predicted_wickets=str(prediction_bw[0])
        except Exception: predicted_wickets = "Error"
    return {'runs_bin': predicted_runs, 'wickets_bin': predicted_wickets}

def _classify_player_role(player_name, match_format):
    """Classifies player role based on career stats."""
    player_bat_stats = batting_df_full[(batting_df_full['player'] == player_name) & (batting_df_full['match_type'] == match_format)]
    player_bowl_stats = bowling_df_full[(bowling_df_full['player'] == player_name) & (bowling_df_full['match_type'] == match_format)]
    
    is_batsman = False
    if not player_bat_stats.empty:
        latest_bat = player_bat_stats.sort_values('date').iloc[-1]
        if latest_bat['career_innings'] > 20 and (latest_bat['career_runs'] / latest_bat['career_innings']) > 15:
            is_batsman = True
            
    is_bowler = False
    if not player_bowl_stats.empty:
        latest_bowl = player_bowl_stats.sort_values('date').iloc[-1]
        if latest_bowl['career_balls_bowled'] > 300 and latest_bowl['career_wickets'] > 10:
            is_bowler = True

    if is_batsman and is_bowler: return "All-Rounder"
    if is_batsman: return "Batsman"
    if is_bowler: return "Bowler"
    
    # Fallback for inexperienced players
    if not player_bat_stats.empty: return "Batsman"
    if not player_bowl_stats.empty: return "Bowler"
    return "Unknown" # Should be rare

def parse_prediction_bin(bin_str, is_runs=True):
    """Converts prediction string like '15-30' to an average number."""
    if not isinstance(bin_str, str) or '-' not in bin_str:
        return 0
    try:
        low, high = map(int, bin_str.split('-'))
        return (low + high) / 2
    except ValueError:
        return 0

@app.route('/api/best11', methods=['POST'])
def get_best11():
    data = request.get_json()
    squad = data.get('squad', [])
    venue = data.get('venue', '')
    opposition = data.get('opposition', '')
    match_format = data.get('match_format', 'T20')

    if len(squad) < 11:
        return jsonify({'error': 'Please provide a squad of at least 11 players.'}), 400

    player_data = []
    for player_name in squad:
        prediction = _predict_player_performance(player_name, venue, opposition, match_format)
        role = _classify_player_role(player_name, match_format)
        
        runs_score = parse_prediction_bin(prediction['runs_bin'])
        wickets_score = parse_prediction_bin(prediction['wickets_bin'])
        
        player_data.append({
            'name': player_name,
            'role': role,
            'runs_bin': prediction['runs_bin'],
            'wickets_bin': prediction['wickets_bin'],
            'runs_score': runs_score,
            'wickets_score': wickets_score,
            'impact_score': runs_score + (wickets_score * 25) # Wickets are more valuable
        })
    
    # Sort by roles
    batsmen = sorted([p for p in player_data if p['role'] == 'Batsman'], key=lambda x: x['runs_score'], reverse=True)
    bowlers = sorted([p for p in player_data if p['role'] == 'Bowler'], key=lambda x: x['wickets_score'], reverse=True)
    all_rounders = sorted([p for p in player_data if p['role'] == 'All-Rounder'], key=lambda x: x['impact_score'], reverse=True)

    best_11 = []
    
    # Selection logic
    best_11.extend(all_rounders[:4]) # Take up to 4 best all-rounders
    
    bowling_options = bowlers + [p for p in all_rounders if p not in best_11]
    bowling_options_sorted = sorted(bowling_options, key=lambda x: x['wickets_score'], reverse=True)
    
    # Fill up to 5 main bowling options
    needed_bowlers = 5 - len([p for p in best_11 if p['role'] == 'All-Rounder' or p['role'] == 'Bowler'])
    best_11.extend([p for p in bowling_options_sorted if p not in best_11][:needed_bowlers])
    
    batting_options = batsmen + [p for p in all_rounders if p not in best_11]
    batting_options_sorted = sorted(batting_options, key=lambda x: x['runs_score'], reverse=True)
    
    # Fill remaining spots with best batsmen
    remaining_spots = 11 - len(best_11)
    best_11.extend([p for p in batting_options_sorted if p not in best_11][:remaining_spots])
    
    # If still not 11, fill with best of the rest
    if len(best_11) < 11:
        all_players_sorted = sorted(player_data, key=lambda x: x['impact_score'], reverse=True)
        needed = 11 - len(best_11)
        best_11.extend([p for p in all_players_sorted if p not in best_11][:needed])

    return jsonify(best_11[:11])


# --- 4. Run the App (Unchanged) ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

