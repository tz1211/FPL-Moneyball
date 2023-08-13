import pandas as pd 
import numpy as np
import requests 
from bs4 import BeautifulSoup 
from ortools.linear_solver import pywraplp 

def scrape_standing(season, gw): 
    url = f"https://www.worldfootball.net/schedule/eng-premier-league-{season}-spieltag/{gw}/" 
    response = requests.get(url)
    if response.status_code == 200: 
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find_all("table", class_="standard_tabelle")[1]

        data = []
        columns = []

        # Extract table headers
        for th in table.find_all('th'):
            columns.append(th.text.strip())

        # Extract table rows
        for row in table.find_all('tr'):
            row_data = [td.text.strip() for td in row.find_all('td')]
            if row_data:  # Skip header row
                data.append(row_data)

        # Create a DataFrame
        df = pd.DataFrame(data)
        df.drop(1, axis=1, inplace=True)
        df.columns = columns
        df["#"] = df["#"].replace("", np.nan)
        df = df.fillna(method="ffill")
        df.rename({"#": "Standing"}, axis=1, inplace=True)
        df["GW"] = gw
        df_out = df[["GW", "Team", "Standing"]]
        
        return df_out
    else: 
        raise ValueError(f"status_code = {response.status_code}")


def get_team_standing(season, gw, id_mapping, df_teams): 
    df_standing = scrape_standing(season, gw)

    df_teams_standing = pd.DataFrame({"Team": df_standing["Team"].unique()})
    df_teams_standing.sort_values("Team", inplace=True)
    df_teams_standing["id"] = id_mapping

    df_standing_mapped = pd.merge(df_standing, df_teams_standing, on="Team", how="left")
    df_merged = pd.merge(df_teams, df_standing_mapped, on="id", how="left")
    df_standing_cleaned = df_merged[["GW", "name", "Standing"]]
    try: 
        df_standing_cleaned["Standing"].astype("int")
    except ValueError: 
        df_standing_cleaned["Standing"].astype("float")

    return df_standing_cleaned


def get_sched_strength(team, current_gw, df_fixtures, df_standing): 
    # determine next 5 gw 
    next_5 = range(current_gw+1, current_gw+6) 
    df_fixtures_next_5 = df_fixtures[df_fixtures["GW"].isin(next_5)]

    # determine opponents 
    opp_list = df_fixtures_next_5[df_fixtures_next_5["home_team"] == team].away_team.tolist()
    opp_list += df_fixtures_next_5[df_fixtures_next_5["away_team"] == team].home_team.tolist()

    # calculate avg opponent standing 
    if current_gw == 0: 
        df_standing_current = df_standing[df_standing["GW"] == 38] # use standing from the end of last season 
    else: 
        df_standing_current = df_standing[df_standing["GW"] == current_gw] # use standing from current gw (avoid data leakage)
    df_standing_opp = df_standing_current[df_standing_current["name"].isin(opp_list)] 
    avg_standing = df_standing_opp["Standing"].mean() 

    return avg_standing 


def knapsack_fpl_xi(df_players, budget, fwd_count, mid_count, def_count, gk_count):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    if not solver:
        return []

    num_players = len(df_players)
    costs = df_players["now_cost"].to_list()
    points = df_players["xP"].to_list()
    positions = df_players["position"].to_list()

    selected = [solver.IntVar(0, 1, f"player_{i}") for i in range(num_players)]

    # Budget constraint: sum(costs * selected) <= budget
    solver.Add(solver.Sum(costs[i] * selected[i] for i in range(num_players)) <= budget)

    # Position constraints
    positions_to_count = {"FWD": fwd_count, "MID": mid_count, "DEF": def_count, "GK": gk_count}
    for pos, count in positions_to_count.items():
        position_indices = [i for i, p in enumerate(positions) if p == pos]
        solver.Add(solver.Sum(selected[i] for i in position_indices) == count)

    # Objective function: maximize total points
    solver.Maximize(solver.Sum(points[i] * selected[i] for i in range(num_players)))

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        return []

    selected_indices = [i for i in range(num_players) if selected[i].solution_value() > 0.5]
    df_selected = df_players[df_players.index.isin(selected_indices)]

    return df_selected