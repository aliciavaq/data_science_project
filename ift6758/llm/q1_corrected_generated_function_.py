import os
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json
from typing import Optional

# manual fix 1
#HOCKEY_DATA_DIR = os.getenv('HOCKEY_DATA_DIR')

def get_hockey_data(year: int, path: Optional[str] = None) -> pd.DataFrame:
    #if path is None:
    #   path = HOCKEY_DATA_DIR
     # manual fix 1
    if path is None:
        path = os.getenv('HOCKEY_DATA_DIR')  
    if not path:
        raise ValueError("Neither path nor HOCKEY_DATA_DIR environment variable is set")
    
    #Manual fix 2
    #season = f"{year}-{year+1}"
    season = f"{year}-{str(year + 1)[-2:]}"
    season_dir = os.path.join(path, season)
    os.makedirs(season_dir, exist_ok=True)
    
    # Start date: September 1 of year
    start_date = datetime(year, 9, 1)
    # End date: October 31 of year + 1
    end_date = datetime(year + 1, 10, 31)
    
    game_ids = set()
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            game_week = data.get('gameWeek', [])
            for day in game_week:
                for game in day.get('games', []):
                    game_id = game['id']
                    game_type = game['gameType']
                    if game_type in [2, 3] and str(game_id).startswith(str(year)):
                        game_ids.add(game_id)
        except requests.exceptions.RequestException as e:
            raise e
        # Manual fix 3: each schedule request returns 7 days (61 requests instead of 426)
        # current_date += timedelta(days=1)
        current_date += timedelta(days=7)

        # Manual fix 4: moved after each game download
        # time.sleep(1)


    # Manual fix 5: raise instead of silently returning an empty DataFrame
    if not game_ids:
        raise ValueError(f"No games found for season {season}")
    all_plays_data = []
    
    for game_id in game_ids:
        file_path = os.path.join(season_dir, f"{game_id}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise e
            
            # Validate response
            if data.get('id') != game_id or not isinstance(data.get('plays'), list):
                raise ValueError(f"Invalid response for game_id {game_id}: id mismatch or plays not a list")
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            #Manual fix 4: sleep only after an actual download
            time.sleep(1) 
        
        # Normalize plays and add game_id
        plays_df = pd.json_normalize(data['plays'])
        plays_df['game_id'] = game_id
        all_plays_data.append(plays_df)
    
    if not all_plays_data:
        return pd.DataFrame()
    
    return pd.concat(all_plays_data, ignore_index=True)