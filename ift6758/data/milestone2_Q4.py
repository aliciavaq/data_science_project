"""
Étape 2 - Question 4 : Ingénierie des caractéristiques II

On relit les JSON avec TOUS les événements (pas seulement les tirs) pour
trouver le vrai événement précédent, puis on garde les tirs et les buts et
on les fusionne avec le tableau de la question 2 sur (game_id, event_id).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOAL_LINE = 89
SHOT_TYPES = ["shot-on-goal", "goal"]

# 1) Secondes de jeu
# Cette fonction calcule le temps écoulé depuis le début du match, en secondes, pour chaque événement. Chaque période dure 1200 secondes (20 minutes).
# Si la période ou le temps est manquant, elle retourne NaN.
# Les autres caractéristiques sont déjà connues : x, y, distance, angle, shot_type, etc. 
def get_game_seconds(period, time):
    """Secondes depuis le début du match (chaque période dure 1200 s)."""
    if period is None or not isinstance(time, str):
        return np.nan
    minutes, seconds = time.split(":")
    return (int(period) - 1) * 1200 + int(minutes) * 60 + int(seconds)

# 2) Événement précédent
# Cette fonction lit tous les événements d'un match, les trie par ordre de jeu, et crée un tableau avec les informations pertinentes pour chaque événement. Elle inclut l'ID du match, l'ID de l'événement, le numéro de période, 
# le type d'événement, l'équipe associée, les coordonnées x et y, et le temps écoulé depuis le début du match.
def read_game_events(game_data):
    """Tous les événements d'un match, dans l'ordre du jeu."""
    plays = sorted(game_data.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    events = []
    for order, play in enumerate(plays):
        details = play.get("details") or {}
        period = play.get("periodDescriptor", {})
        events.append({
            "game_id": game_data["id"],
            "event_id": play.get("eventId"),
            "order": order,
            "period": period.get("number"),
            "period_type": period.get("periodType"),
            "game_seconds": get_game_seconds(period.get("number"), play.get("timeInPeriod")),
            "event_type": play.get("typeDescKey"),
            "team_id": details.get("eventOwnerTeamId"),
            "x_event": details.get("xCoord"),
            "y_event": details.get("yCoord"),
            "play": play,   # utile pour lire les pénalités (bonus)
        })
    return events

# Cette fonction ajoute des colonnes pour l'événement précédent à chaque événement d'un match. Elle trie les événements par ID de match et ordre, 
# puis groupe les événements par ID de match et période. Pour chaque événement, elle décale les colonnes pertinentes (type d'événement, coordonnées x et y, 
# temps écoulé depuis le dernier événement) d'une ligne vers le bas pour obtenir les valeurs de l'événement précédent. 
# Elle calcule également la distance entre l'événement actuel et l'événement précédent en utilisant la formule de la distance euclidienne.
def add_previous_event_columns(events):
    """Décale d'une ligne à l'intérieur de chaque (match, période)."""
    events = events.sort_values(["game_id", "order"])
    events[["x_event", "y_event"]] = events[["x_event", "y_event"]].apply(pd.to_numeric)
    groups = events.groupby(["game_id", "period"], sort=False)

    events["last_event_type"] = groups["event_type"].shift(1)
    events["last_x"] = groups["x_event"].shift(1)
    events["last_y"] = groups["y_event"].shift(1)
    events["time_since_last"] = events["game_seconds"] - groups["game_seconds"].shift(1)
    events["dist_from_last"] = np.hypot(
        events["x_event"] - events["last_x"], events["y_event"] - events["last_y"]
    )
    return events

# Cette fonction crée un DataFrame contenant les événements précédents et l'état de l'avantage numérique pour les tirs d'une saison donnée.
def create_previous_event_dataframe(year, data_dir):
    """Événement précédent + avantage numérique pour les tirs d'une saison."""
    season_path = Path(data_dir) / f"{year}-{str(year + 1)[-2:]}"
    frames = []

    for json_file in sorted(season_path.glob("*.json")):
        game_data = json.loads(json_file.read_text(encoding="utf-8"))
        game_type = str(game_data["id"])[4:6]
        if game_type not in ["02", "03"]:
            continue

        events = read_game_events(game_data)
        power_play = compute_power_play(events, game_data, game_type)
        game_df = pd.DataFrame(events).drop(columns="play").join(power_play)
        frames.append(add_previous_event_columns(game_df))

    events = pd.concat(frames, ignore_index=True)
    shots = events[events["event_type"].isin(SHOT_TYPES)]   # filtrer APRÈS le décalage
    return shots[[
        "game_id", "event_id", "game_seconds",
        "last_event_type", "last_x", "last_y", "time_since_last", "dist_from_last",
        "pp_elapsed", "shooter_skaters_pen", "opponent_skaters_pen",
    ]]

# 3) Rebond, changement d'angle, vitesse
# Cette fonction ajoute des colonnes pour le rebond, le changement d'angle et la vitesse à un DataFrame de tirs. Elle fusionne le DataFrame de tirs avec un DataFrame d'événements précédents sur les colonnes "game_id" et "event_id".
def add_feature_engineering_2(shots, previous):
    df = shots.drop(columns="game_seconds", errors="ignore").merge(
        previous, on=["game_id", "event_id"], how="left", validate="one_to_one"
    )
    print(f"{df['last_event_type'].isna().sum()} tirs sans événement précédent")

    # Rebond (reste manquant si l'événement précédent est inconnu)
    df["rebound"] = (
        df["last_event_type"].eq("shot-on-goal").astype("boolean")
        .mask(df["last_event_type"].isna())
    )

    # Angle de l'événement précédent, vers le filet attaqué par le tireur actuel
    direction = np.sign(df["net_x"])
    df["last_angle"] = np.degrees(np.arctan2(
        df["last_y"] * direction, GOAL_LINE - df["last_x"] * direction
    ))

    # Écart d'angle ramené entre -180 et 180 ; 0 si ce n'est pas un rebond
    diff = ((df["angle"] - df["last_angle"] + 180) % 360 - 180).abs()
    df["angle_change"] = diff.where(df["rebound"].fillna(False), 0.0).mask(df["rebound"].isna())

    # Vitesse ; 0 s donne une vitesse indéfinie (NaN)
    t = df["time_since_last"]
    df["speed"] = (df["dist_from_last"] / t).where(t > 0)
    print(f"{(t == 0).sum()} tirs à 0 s de l'événement précédent (vitesse manquante)")

    return df

# Cette fonction charge le tableau des caractéristiques à partir d'un cache CSV si disponible, sinon elle crée le tableau en combinant les événements précédents pour chaque saison et en ajoutant l'ingénierie des caractéristiques.
def load_feature_table(shots, data_dir, cache_path):
    """Fonction principale, avec un CSV en cache (à supprimer si le code change)."""
    if Path(cache_path).exists():
        previous = pd.read_csv(cache_path)
    else:
        years = sorted(shots["season"].str[:4].astype(int).unique())
        previous = pd.concat(
            [create_previous_event_dataframe(y, data_dir) for y in years], ignore_index=True
        )
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        previous.to_csv(cache_path, index=False)
    return add_feature_engineering_2(shots, previous)

# 4) Bonus : avantage numérique reconstruit avec les pénalités
# Cette classe PenaltyTracker suit le nombre de patineurs de chaque équipe en fonction des pénalités. Elle gère les pénalités actives et en attente, 
# calcule le nombre de patineurs sur la glace pour chaque équipe, et met à jour l'état de l'avantage numérique en fonction des événements de pénalité et de but.
class PenaltyTracker:
    """
    Nombre de patineurs de chaque équipe selon les pénalités (sans situationCode).
    Mineure : finit sur un but en avantage. Double mineure : le but termine la
    tranche en cours. Majeure : ne finit pas sur un but. Inconduite : aucun effet.
    Minimum 3 patineurs (3e pénalité en attente). Coïncidentes : s'annulent, sauf
    une mineure de chaque côté à 5 c. 5 (4 c. 4). Prolongation régulière : 3 c. 3.
    """

    def __init__(self, home_id, away_id):
        self.teams = [home_id, away_id]
        self.active = {home_id: [], away_id: []}
        self.waiting = {home_id: [], away_id: []}
        self.pp_team = None
        self.pp_start = None
        self.overtime_3v3 = False

    def other(self, team):
        return self.teams[1] if team == self.teams[0] else self.teams[0]

    def skaters(self):
        home, away = self.teams
        n_home, n_away = len(self.active[home]), len(self.active[away])
        if self.overtime_3v3:   # une pénalité donne un patineur de PLUS à l'adversaire
            diff = n_home - n_away
            return {home: 3 + max(0, min(-diff, 2)), away: 3 + max(0, min(diff, 2))}
        return {home: 5 - min(n_home, 2), away: 5 - min(n_away, 2)}

    def update_power_play(self, t):
        sk = self.skaters()
        home, away = self.teams
        if sk[home] == sk[away]:
            self.pp_team, self.pp_start = None, None
        else:
            team = home if sk[home] > sk[away] else away
            if team != self.pp_team:   # pénalités empilées : on ne remet pas à 0
                self.pp_team, self.pp_start = team, t

    def expire(self, t):
        """Retire les pénalités finies, dans l'ordre chronologique."""
        while True:
            done = [(p["end"], tm, p) for tm in self.teams for p in self.active[tm] if p["end"] <= t]
            if not done:
                return
            end, team, penalty = min(done, key=lambda d: d[0])
            self.active[team].remove(penalty)
            if self.waiting[team]:   # une pénalité en attente prend la place
                nxt = self.waiting[team].pop(0)
                self.active[team].append({"end": end + nxt["duration"], "minor": nxt["minor"]})
            self.update_power_play(end)

    def add_penalties(self, t, plays):
        new = {tm: [] for tm in self.teams}
        for play in plays:
            d = play.get("details") or {}
            if (d.get("eventOwnerTeamId") in new and d.get("duration") in [2, 4, 5]
                    and d.get("typeCode") not in ["PS", "MIS", "GMIS"]):
                new[d["eventOwnerTeamId"]].append(d["duration"])

        home, away = self.teams
        full_strength = not any(self.active[tm] or self.waiting[tm] for tm in self.teams)
        if not (full_strength and new[home] == [2] and new[away] == [2] and not self.overtime_3v3):
            for m in list(new[home]):   # pénalités coïncidentes de même durée
                if m in new[away]:
                    new[home].remove(m)
                    new[away].remove(m)

        for team in self.teams:
            for m in new[team]:
                if len(self.active[team]) < 2:
                    self.active[team].append({"end": t + m * 60, "minor": m in [2, 4]})
                else:
                    self.waiting[team].append({"duration": m * 60, "minor": m in [2, 4]})
        self.update_power_play(t)

    def goal(self, t, team):
        if team not in self.teams:
            return
        opp = self.other(team)
        minors = [p for p in self.active[opp] if p["minor"]]
        if self.skaters()[team] <= self.skaters()[opp] or not minors:
            return
        p = min(minors, key=lambda p: p["end"])
        p["end"] = t + 120 if p["end"] - t > 120 else t
        self.expire(t)
        self.update_power_play(t)


def compute_power_play(events, game_data, game_type):
    """État de l'avantage numérique au moment de chaque tir (NaN ailleurs)."""
    home_id, away_id = game_data["homeTeam"]["id"], game_data["awayTeam"]["id"]
    tracker = PenaltyTracker(home_id, away_id)
    rows = [[np.nan] * 3 for _ in events]

    penalties_by_time = {}
    for e in events:
        if e["event_type"] == "penalty":
            penalties_by_time.setdefault(e["game_seconds"], []).append(e["play"])
    done_times = set()

    for i, e in enumerate(events):
        t = e["game_seconds"]
        if pd.isna(t) or e["period_type"] == "SO":
            continue
        tracker.overtime_3v3 = game_type == "02" and e["period_type"] == "OT"
        tracker.expire(t)
        tracker.update_power_play(t)

        # Situation lue AVANT que l'événement la modifie
        if e["event_type"] in SHOT_TYPES and e["team_id"] in tracker.teams:
            sk = tracker.skaters()
            elapsed = 0.0 if tracker.pp_start is None else t - tracker.pp_start
            rows[i] = [elapsed, sk[e["team_id"]], sk[tracker.other(e["team_id"])]]

        if e["event_type"] == "penalty" and t not in done_times:
            tracker.add_penalties(t, penalties_by_time[t])
            done_times.add(t)
        if e["event_type"] == "goal":
            tracker.goal(t, e["team_id"])

    return pd.DataFrame(rows, columns=["pp_elapsed", "shooter_skaters_pen", "opponent_skaters_pen"])


def compare_with_situation_code(df):
    """Compare nos patineurs (pénalités) à ceux du situationCode (colonne strength)."""
    c = df.dropna(subset=["shooter_skaters_pen", "strength"]).copy()
    c["strength_pen"] = (c["shooter_skaters_pen"].astype(int).astype(str) + "v"
                         + c["opponent_skaters_pen"].astype(int).astype(str))
    c["agree"] = c["strength_pen"] == c["strength"]
    c["cause"] = np.select(
        [c["strength"].str.contains("6"), c["strength"].isin(["1v0", "0v1"]), c["period_type"].eq("OT")],
        ["Gardien retiré", "Tir de pénalité", "Prolongation"], default="Autre",
    )
    print(f"Concordance avec situationCode : {c['agree'].mean():.2%} ({len(c)} tirs)")
    return (c[~c["agree"]].groupby(["cause", "strength", "strength_pen"]).size()
            .sort_values(ascending=False).rename("n_tirs").reset_index())


# 5) Artefact WandB (match 2017021065, Winnipeg c. Washington)
# Cette fonction téléverse un artefact WandB contenant le DataFrame filtré pour le match Winnipeg contre Washington (ID 2017021065). 
# Elle initialise une exécution WandB, crée un artefact de type "dataset", ajoute le DataFrame filtré sous forme de table, puis enregistre l'artefact et termine l'exécution.
def upload_wpg_wsh(df, project, entity):
    import wandb
    run = wandb.init(project=project, entity=entity, job_type="dataset-upload")
    artifact = wandb.Artifact("wpg_v_wsh_2017021065", type="dataset")
    table = wandb.Table(dataframe=df[df["game_id"] == 2017021065].reset_index(drop=True))
    artifact.add(table, "wpg_v_wsh_2017021065")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pour « import ift6758 »
    from milestone2_Q2 import load_dev_set

    DATA_DIR = Path(__file__).resolve().parent
    shots = load_dev_set(DATA_DIR)
    df = load_feature_table(shots, DATA_DIR, DATA_DIR / "processed" / "previous_events.csv")

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)
    print(df[["game_id", "game_seconds", "period", "x", "y", "distance", "angle", "shot_type",
              "last_event_type", "last_x", "last_y", "time_since_last", "dist_from_last",
              "rebound", "angle_change", "speed",
              "pp_elapsed", "shooter_skaters_pen", "opponent_skaters_pen"]].head(10))

    print(compare_with_situation_code(df).head(15))

    # À lancer UNE SEULE fois (retirer le #, exécuter, remettre le #) :
    # upload_wpg_wsh(df, project="Milestone2_Question4", entity="IFT-6758B-milestone-2")