import pandas as pd 
import numpy as np
import requests 
from bs4 import BeautifulSoup 

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