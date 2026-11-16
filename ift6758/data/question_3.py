import json
from pathlib import Path
import numpy as np
import pandas as pd

goal_line = 89
blue_line = 25


# This function extracts player names from the game data
def get_player_names(game_data):
    player_names = {}

    # Loop through roster spots to get player IDs and names
    for player in game_data.get("rosterSpots", []):
        player_id = player.get("playerId")

        first_name = player.get("firstName", {})
        last_name = player.get("lastName", {})
       
        # Checks if first_name and last_name are dictionaries and extracts the "default" value if they are. 
        if isinstance(first_name, dict):
            first_name = first_name.get("default")
        if isinstance(last_name, dict):
            last_name = last_name.get("default")

        # Store the full name if the player ID is available
        if player_id is not None:
            player_names[player_id] = " ".join(
                name for name in [first_name, last_name] if name
            )

    return player_names


# This function determines numerical strength from the situationCode 
def get_strength(
    situation_code, event_team_id, home_team_id, away_team_id
):
    if situation_code is None or event_team_id is None:
        return None, None

    # Convert the code to a string and preserve leading zeros (https://docs.python.org/3/builtins/stdtypes.html#str.zfill)
    code = str(situation_code).zfill(4)

    if len(code) != 4 or not code.isdigit():
        return None, None

    # Extract the number of skaters for each team
    away_skaters = int(code[1])
    home_skaters = int(code[2])

    # Identify the event team's number of skaters
    if event_team_id == home_team_id:
        team_skaters = home_skaters
        opponent_skaters = away_skaters
    elif event_team_id == away_team_id:
        team_skaters = away_skaters
        opponent_skaters = home_skaters
    else:
        return None, None

    # Classifying the situation based on the number of skaters
    if team_skaters == opponent_skaters:
        strength_type = "Even strength"
    elif team_skaters > opponent_skaters:
        strength_type = "Power play"
    else:
        strength_type = "Shorthanded"

    # Creating a string showing the number of skaters on each team, from the perspective of the team taking the shot or scoring the goal.
    strength = f"{team_skaters}v{opponent_skaters}"

    return strength_type, strength


# New function for milestone 2 : this function finds which net the shooting team attacks
def get_attacked_net(home_defending_side, event_team_id, home_team_id, away_team_id):
    """
    Return the x coordinate (+89 or -89) of the net attacked by the team that took the shot

    Each play has a homeTeamDefendingSide field ("left" or "right"), which is the side of the rink defended by the home team,
    where "left" means negative x.
    """
    if home_defending_side not in ("left", "right"):
        return None

    if event_team_id is None or event_team_id not in (home_team_id, away_team_id):
        return None

    # Home team defends the left net (x = -89) -> it attacks x = +89
    if home_defending_side == "left":
        home_attacked_net = goal_line
    else:
        home_attacked_net = -goal_line

    if event_team_id == home_team_id:
        return home_attacked_net

    return -home_attacked_net


# New function for milestone 2 : this function derives the empty net flag from the situationCode
def get_empty_net(
    situation_code, event_team_id, home_team_id, away_team_id
):
    """
    Return 1 if the defending team's net is empty, 0 if its goalie is in net,
    or None if it cannot be determined.

    situationCode has 4 digits:
    [away goalie in net][away skaters][home skaters][home goalie in net].
    The defending team is the team that did not take the shot, so we read
    the goalie digit of the other team.
    """
    if situation_code is None or event_team_id is None:
        return None

    code = str(situation_code).zfill(4)

    if len(code) != 4 or not code.isdigit():
        return None

    if event_team_id == home_team_id:
        goalie_digit = code[0]    # home team shoots -> away goalie defends
    elif event_team_id == away_team_id:
        goalie_digit = code[3]    # away team shoots -> home goalie defends
    else:
        return None

    # A goalie digit should only be 0 (empty net) or 1 (goalie in net)
    if goalie_digit not in ("0", "1"):
        return None

    return 1 if goalie_digit == "0" else 0


# New function for milestone 2 : fallback when homeTeamDefendingSide is missing
def fill_missing_attacked_net(df):
    """
    Fill the attacked net (net_x) when homeTeamDefendingSide is missing.

    In our raw data, homeTeamDefendingSide is null for every game of the
    2016-17, 2017-18 and 2018-19 seasons. For those shots, we infer the
    attacked net from where the team took its shots during the same period:
    teams take almost all their shots in their offensive half, so the sign
    of the median x of the team's shots in that period gives the attacked side.

    Returns a new DataFrame. The column net_x_source says where net_x comes
    from: "api" (homeTeamDefendingSide), "period_majority" (this fallback),
    or missing if it still cannot be determined.
    """

    df = df.copy()
    df["net_x"] = pd.to_numeric(df["net_x"], errors="coerce")
    df["net_x_source"] = pd.Series(
        np.where(df["net_x"].notna(), "api", None), index=df.index, dtype="object"
    )

    x = pd.to_numeric(df["x"], errors="coerce")

    # Median x of the team's shots in each (game, period, team)
    median_x = x.groupby([df["game_id"], df["period"], df["team"]]).transform("median")

    # +89 if the team shot mostly toward positive x, -89 toward negative x
    inferred_net_x = np.sign(median_x) * goal_line
    inferred_net_x = inferred_net_x.where(inferred_net_x != 0)   # median exactly 0: undetermined

    to_fill = df["net_x"].isna() & inferred_net_x.notna()
    df.loc[to_fill, "net_x"] = inferred_net_x[to_fill]
    df.loc[to_fill, "net_x_source"] = "period_majority"

    return df

# New function for milestone 2 : this function adds the distance and angle to the attacked net
def add_shot_geometry(df):
    """
    Add shot distance and angle relative to the net attacked by the shooting 
    team. The net is approximated by a single point at (+/-89, 0).

    Returns a new DataFrame with these added columns:
      - x_norm, y_norm : coordinates rotated so that the attacked net is 
        always at (+89, 0)
      - distance : distance to the attacked net, in feet
      - angle : angle in degrees, 0 = straight in front of the net, +/-90 = on 
        the goal line, beyond +/-90 = behind the net
      - def_zone : True if the shot was taken behind the shooting team's own 
        blue line (its defensive zone)

    Unknown values (missing coordinates or attacked side) stay missing.
    """
    df = df.copy()

    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")

    # +1 if the team attacks toward positive x, -1 toward negative x, NaN if unknown
    direction = np.sign(pd.to_numeric(df["net_x"], errors="coerce"))

    # Rotating the rink by 180 degrees flips both x and y, so the attacked net is always at (+89, 0)
    df["x_norm"] = x * direction
    df["y_norm"] = y * direction

    # Distance from the goal line along x: positive in front of the net, negative behind it
    dx = goal_line - df["x_norm"]

    # Hypot calculates the distance using the Pythagorean theorem
    df["distance"] = np.hypot(dx, df["y_norm"])

    # arctan2 handles shots from behind the net (dx < 0) correctly
    df["angle"] = np.degrees(np.arctan2(df["y_norm"], dx))

    # Defensive zone: behind the shooting team's own blue line (x_norm < -25).
    # Stays missing when the position is unknown instead of becoming False.
    df["def_zone"] = (
        (df["x_norm"] < -blue_line)
        .astype("boolean")
        .mask(df["x_norm"].isna())
    )

    return df

# This function creates a DataFrame containing shots on goal and goals for one season
def create_shot_dataframe(year, path):
    # Locate the season directory
    season = f"{year}-{str(year + 1)[-2:]}"
    season_path = Path(path) / season

    if not season_path.exists():
        raise FileNotFoundError(
            f"Season directory not found: {season_path}"
        )

    # Store event data
    events = []

    # Loop through all JSON files in the season directory
    for json_file in sorted(season_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as file:
            game_data = json.load(file)

        game_id = game_data.get("id")

        # Determine the game type from the game ID
        game_type_code = str(game_id)[4:6]

        if game_type_code == "02":
            game_type = "Regular season"
        elif game_type_code == "03":
            game_type = "Playoffs"
        else:
            continue

        player_names = get_player_names(game_data)

        home_team = game_data.get("homeTeam", {})
        away_team = game_data.get("awayTeam", {})
        home_team_id = home_team.get("id")
        away_team_id = away_team.get("id")

        # Loop through all events in the game
        for play in game_data.get("plays", []):
            event_type = play.get("typeDescKey")

            # Keep shots on goal and goals only
            if event_type not in ["shot-on-goal", "goal"]:
                continue

            details = play.get("details", {})
            period_info = play.get("periodDescriptor", {})

            # Get shooter ID
            shooter_id = details.get("shootingPlayerId")

            if event_type == "goal":
                shooter_id = details.get(
                    "scoringPlayerId", shooter_id
                )

            # Get goalie ID and event team ID
            goalie_id = details.get("goalieInNetId")
            team_id = details.get("eventOwnerTeamId")

            # Match the event team ID to its abbreviation
            team_name = None

            if team_id is not None and team_id == home_team_id:
                team_name = home_team.get("abbrev")
            elif team_id is not None and team_id == away_team_id:
                team_name = away_team.get("abbrev")

            # Get the situation code and calculate strength
            situation_code = play.get("situationCode")

            strength_type, strength = get_strength(
                situation_code,
                team_id,
                home_team_id,
                away_team_id
            )

            # New implementation for milestone 2 : empty net from the defending team's goalie digit
            is_empty_net = get_empty_net(
                situation_code,
                team_id,
                home_team_id,
                away_team_id
            )

            # New implementation for milestone 2 : net attacked by the shooting team (needed for distance and angle)
            home_defending_side = play.get("homeTeamDefendingSide")

            net_x = get_attacked_net(
                home_defending_side,
                team_id,
                home_team_id,
                away_team_id
            )

            # Store the event information
            events.append({
                "game_id": game_id,
                "event_id": play.get("eventId"),
                "season": season,
                "game_type": game_type,
                "game_date": game_data.get("gameDate"),          # New
                "home_team": home_team.get("abbrev"),            # New
                "away_team": away_team.get("abbrev"),            # New
                "period": period_info.get("number"),
                "period_type": period_info.get("periodType"),    # New: REG, OT or SO
                "time": play.get("timeInPeriod"),
                "team": team_name,
                "event_type": event_type,
                "is_goal": int(event_type == "goal"),            # New
                "x": details.get("xCoord"),
                "y": details.get("yCoord"),
                "zone_code": details.get("zoneCode"),            # New: O, N or D (shooting team's view)
                "home_defending_side": home_defending_side,      # New
                "net_x": net_x,                                  # New
                "shooter": player_names.get(shooter_id),
                "goalie": player_names.get(goalie_id),
                "shot_type": details.get("shotType"),
                 "is_empty_net": is_empty_net,                   # Changed
                "situation_code": situation_code,
                "strength_type": strength_type,
                "strength": strength
            })

    # Convert the event list into a DataFrame
    df = pd.DataFrame(events)

    # Nothing to compute if the season has no shots
    if df.empty:
        return df

    # Check that each event is uniquely identified
    if df.duplicated(["game_id", "event_id"]).any():
        raise ValueError("Duplicate event identifiers found.")

    # Keep unknown empty-net values as missing (NA) instead of 0
    df["is_empty_net"] = df["is_empty_net"].astype("Int64")

    # New for milestone 2 : attacked net for the seasons without homeTeamDefendingSide (2016-17 to 2018-19)
    df = fill_missing_attacked_net(df)
    
    # New for milestone 2 : distance, angle and defensive-zone flag
    df = add_shot_geometry(df)

    return df

# Path to directory containing question_3.py
if __name__ == "__main__":
    # Path to directory containing question_3.py
    DATA_DIR = Path(__file__).resolve().parent

    # Creating the DataFrame and show the first 10 rows
    df = create_shot_dataframe(2023, path=DATA_DIR)

    print(df.head(10))

# References : 
# Pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
# Python Pathlib: https://docs.python.org/3/library/pathlib.html
# Python JSON: https://docs.python.org/3/library/json.html
# Python (dict.get()) : https://docs.python.org/3/builtins/stdtypes.html#dict.get
# Pandas nullable integer (Int64): https://pandas.pydata.org/docs/user_guide/integer_na.html
# Pandas groupby transform: https://pandas.pydata.org/pandas-docs/version/2.1/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html
# Pandas nullable boolean: https://pandas.pydata.org/docs/user_guide/boolean.html
# NumPy hypot: https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
# NumPy arctan2: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
# NHL play-by-play fields (homeTeamDefendingSide, zoneCode): https://github.com/Zmalski/NHL-API-Reference