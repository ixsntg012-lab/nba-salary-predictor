"""
NBA Player Salary Predictor
Person 1's Task: Data Collection
Downloads real NBA stats + salaries from Basketball-Reference
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import io

os.makedirs("data", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.basketball-reference.com/",
}


def download_stats(season_year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_per_game.html"
    print(f"  Downloading {season_year-1}-{str(season_year)[2:]} stats...")
    try:
        session = requests.Session()
        response = session.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"  HTTP {response.status_code}")
            return None
        tables = pd.read_html(io.StringIO(response.text), attrs={"id": "per_game_stats"})
        if not tables:
            return None
        df = tables[0]
        df = df[df["Rk"] != "Rk"]
        df["Season"] = season_year
        numeric_cols = ["Age","G","GS","MP","FG","FGA","FG%","3P","3PA","3P%","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.drop_duplicates(subset=["Player","Season"], keep="first")
        df["Player"] = df["Player"].str.replace(r"\*","",regex=True).str.strip()
        print(f"  Got {len(df)} players")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_salaries(season_year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_salaries.html"
    print(f"  Downloading {season_year} salaries...")
    try:
        session = requests.Session()
        response = session.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        tables = pd.read_html(io.StringIO(response.text), attrs={"id": "salaries"})
        if not tables:
            return None
        df = tables[0]
        df.columns = ["Rank","Player","Salary"] + list(df.columns[3:])
        df = df[["Player","Salary"]].copy()
        df = df[df["Player"] != "Player"]
        df["Salary"] = df["Salary"].astype(str).str.replace(r"[\$,]","",regex=True)
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
        df = df.dropna(subset=["Salary"])
        df["Player"] = df["Player"].str.strip()
        df["Season"] = season_year
        print(f"  Got {len(df)} salaries")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None


def generate_sample_data():
    print("\n  Using sample data (scraping blocked by website).")
    print("  Sample data is fine for the project!\n")
    np.random.seed(42)
    n = 400
    positions = ["PG","SG","SF","PF","C"]
    data = {
        "Player": [f"Player_{i}" for i in range(n)],
        "Pos": np.random.choice(positions, n),
        "Age": np.random.randint(19, 38, n),
        "G": np.random.randint(20, 82, n),
        "GS": np.random.randint(0, 82, n),
        "MP": np.round(np.random.uniform(10, 36, n), 1),
        "PTS": np.round(np.random.uniform(3, 32, n), 1),
        "AST": np.round(np.random.uniform(0.5, 10, n), 1),
        "TRB": np.round(np.random.uniform(1, 13, n), 1),
        "STL": np.round(np.random.uniform(0.2, 2.5, n), 1),
        "BLK": np.round(np.random.uniform(0.1, 3, n), 1),
        "TOV": np.round(np.random.uniform(0.5, 4, n), 1),
        "FG%": np.round(np.random.uniform(0.35, 0.65, n), 3),
        "3P%": np.round(np.random.uniform(0.25, 0.45, n), 3),
        "FT%": np.round(np.random.uniform(0.55, 0.95, n), 3),
        "Season": np.random.choice([2022, 2023, 2024], n),
    }
    df = pd.DataFrame(data)
    df["Salary"] = (
        df["PTS"]*900000 + df["AST"]*500000 + df["TRB"]*350000 +
        df["STL"]*600000 + df["BLK"]*500000 +
        (df["Age"]-28).abs()*-250000 +
        np.random.normal(0, 2500000, n) + 5000000
    ).clip(lower=1000000, upper=55000000).round(-4)
    return df


def main():
    print("=" * 55)
    print("  NBA DATA COLLECTOR")
    print("=" * 55)

    seasons = [2022, 2023, 2024]
    all_stats, all_salaries = [], []

    for season in seasons:
        print(f"\nSeason {season}:")
        stats = download_stats(season)
        time.sleep(4)
        salaries = download_salaries(season)
        time.sleep(4)
        if stats is not None: all_stats.append(stats)
        if salaries is not None: all_salaries.append(salaries)

    if all_stats and all_salaries:
        print("\nMerging stats + salaries...")
        stats_df = pd.concat(all_stats, ignore_index=True)
        sal_df = pd.concat(all_salaries, ignore_index=True)
        merged = pd.merge(stats_df, sal_df[["Player","Salary","Season"]], on=["Player","Season"], how="inner")
        keep_cols = ["Player","Pos","Age","G","GS","MP","PTS","AST","TRB","STL","BLK","TOV","FG%","3P%","FT%","Season","Salary"]
        merged = merged[[c for c in keep_cols if c in merged.columns]]
        merged = merged.dropna(subset=["Salary","PTS"])
        merged.to_csv("data/nba_data.csv", index=False)
        print(f"\nReal data saved! {len(merged)} records to data/nba_data.csv")
    else:
        df = generate_sample_data()
        keep_cols = ["Player","Pos","Age","G","GS","MP","PTS","AST","TRB","STL","BLK","TOV","FG%","3P%","FT%","Season","Salary"]
        df = df[[c for c in keep_cols if c in df.columns]]
        df.to_csv("data/nba_data.csv", index=False)
        print(f"Sample data saved! {len(df)} records to data/nba_data.csv")

    print("\nNext step: cd ..  then  python models/train_models.py")


if __name__ == "__main__":
    main()