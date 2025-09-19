import os
import json
import pandas as pd
import sqlite3
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

# --- Configuration ---
DATA_FOLDER = '../data'
DATABASE_NAME = 'cricket_final_data_by_format.db' # The definitive database
NUM_PROCESSES = max(1, cpu_count() - 1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_match_type(raw_type):
    if raw_type == 'IT20': return 'T20'
    if raw_type == 'ODM': return 'ODI'
    if raw_type == 'MDM': return 'Test'
    return raw_type

def parse_match_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        info = data['info']
        match_id = os.path.splitext(os.path.basename(file_path))[0]
        match_type = clean_match_type(info.get('match_type', 'Unknown'))
        
        all_batting_stats = []
        all_bowling_stats = []
        
        for inning_data in data.get('innings', []):
            team_for_inning = inning_data['team']
            player_stats = {}
            
            for over in inning_data.get('overs', []):
                for delivery in over.get('deliveries', []):
                    batter, bowler = delivery['batter'], delivery['bowler']
                    if batter not in player_stats: player_stats[batter] = {'runs': 0, 'balls_faced': 0}
                    if bowler not in player_stats: player_stats[bowler] = {'wickets': 0, 'runs_conceded': 0, 'balls_bowled': 0}

                    player_stats[batter]['runs'] += delivery['runs']['batter']
                    if 'wides' not in delivery.get('extras', {}):
                        player_stats[batter]['balls_faced'] += 1
                    
                    player_stats[bowler]['runs_conceded'] += delivery['runs']['total']
                    if 'wides' not in delivery.get('extras', {}) and 'noballs' not in delivery.get('extras', {}):
                        player_stats[bowler]['balls_bowled'] += 1
                    
                    if 'wickets' in delivery:
                        for wicket in delivery['wickets']:
                            non_bowler_wickets = ['run out', 'retired hurt', 'obstructing the field', 'retired out']
                            if wicket.get('kind') not in non_bowler_wickets:
                                player_stats[bowler]['wickets'] += 1

            for player, stats in player_stats.items():
                if 'runs' in stats:
                    all_batting_stats.append({
                        'match_id': match_id, 'date': info['dates'][0], 'match_type': match_type, 'venue': info.get('venue', 'N/A'),
                        'player': player, 'team': team_for_inning, 'runs': stats['runs'], 'balls_faced': stats['balls_faced']
                    })
                if 'wickets' in stats:
                    all_bowling_stats.append({
                        'match_id': match_id, 'date': info['dates'][0], 'match_type': match_type, 'venue': info.get('venue', 'N/A'),
                        'player': player, 'team': "Opponent", 
                        'wickets': stats['wickets'], 'runs_conceded': stats['runs_conceded'], 'balls_bowled': stats['balls_bowled']
                    })
        return all_batting_stats, all_bowling_stats
    except Exception:
        return [], []

def calculate_career_stats_by_format(df, is_batting=True):
    df = df.sort_values(by=['player', 'date'])
    
    # --- CRUCIAL CHANGE: Group by both player AND match_type ---
    grouped = df.groupby(['player', 'match_type'])
    
    if is_batting:
        df['career_runs'] = grouped['runs'].transform(lambda x: x.expanding().sum().shift(1))
        df['career_balls_faced'] = grouped['balls_faced'].transform(lambda x: x.expanding().sum().shift(1))
        df['career_innings'] = grouped['player'].transform(lambda x: x.expanding().count().shift(1))
    else: # is_bowling
        df['career_wickets'] = grouped['wickets'].transform(lambda x: x.expanding().sum().shift(1))
        df['career_balls_bowled'] = grouped['balls_bowled'].transform(lambda x: x.expanding().sum().shift(1))
        df['career_runs_conceded'] = grouped['runs_conceded'].transform(lambda x: x.expanding().sum().shift(1))
    
    df.fillna(0, inplace=True)
    return df

def main():
    json_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    logging.info(f"Starting parallel processing of {len(json_files)} files.")
    
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap_unordered(parse_match_data, json_files), total=len(json_files), desc="Extracting Innings Data"))
    
    batting_results = [item for sublist in (r[0] for r in results) for item in sublist]
    bowling_results = [item for sublist in (r[1] for r in results) for item in sublist]

    batting_innings = pd.DataFrame(batting_results)
    bowling_innings = pd.DataFrame(bowling_results)
    
    batting_innings['date'] = pd.to_datetime(batting_innings['date'])
    bowling_innings['date'] = pd.to_datetime(bowling_innings['date'])

    logging.info("Calculating format-specific rolling career stats...")
    batting_final = calculate_career_stats_by_format(batting_innings, is_batting=True)
    bowling_final = calculate_career_stats_by_format(bowling_innings, is_batting=False)

    db_path = f'../{DATABASE_NAME}'
    if os.path.exists(db_path): os.remove(db_path)
    conn = sqlite3.connect(db_path)
    batting_final.to_sql('batting_stats', conn, if_exists='replace', index=False)
    bowling_final.to_sql('bowling_stats', conn, if_exists='replace', index=False)
    conn.close()
    
    logging.info(f"Processing complete. Final database saved to '{db_path}'.")

if __name__ == '__main__':
    main()
