# NBA Player Salary Predictor 🏀

<div align="center">

**Predicts NBA player salaries from performance statistics using ML regression — with an interactive Streamlit dashboard**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189C1A?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### 🚀 [Live Demo → swetha-nba-salary.streamlit.app](https://swetha-nba-salary.streamlit.app)

</div>

---

## What It Does

Enter any NBA player's per-game statistics and the system predicts their market salary in real time using a trained ML regression model. The interactive Streamlit dashboard lets you adjust sliders for points, assists, rebounds, and more — and instantly see the predicted salary tier.

> **Best model:** Ridge Regression — **RMSE: $2.50M** on 2022–2024 NBA season data

---

## Results

| Model | RMSE (lower = better) |
|-------|----------------------|
| **Ridge Regression ✅** | **$2.50M** |
| Random Forest | $2.98M |
| XGBoost | $2.87M |
| SVR | $3.35M |

**Key finding:** PTS (points per game) dominates salary prediction — importance score of 0.85+ in Random Forest. Secondary factors: AST, PER, TRB.

---

## Features

| Feature | Description |
|---------|-------------|
| 4-model comparison | Ridge Regression, Random Forest, XGBoost, SVR — auto-selects best |
| Real web scraping | Data scraped from Basketball-Reference.com (2022–2024 seasons) |
| Feature engineering | PER, TS%, GS ratio, age-prime factor, pts/min |
| Interactive dashboard | Streamlit sliders for real-time prediction |
| Salary tier classification | Superstar / All-Star / Starter / Rotation / Minimum |
| Visualization | Model comparison, actual vs predicted, feature importance, salary distribution |

---

## How It Works

```
Basketball-Reference.com
        │
        ▼
Web Scraping (BeautifulSoup + requests)
Per-game stats + salaries for 2022, 2023, 2024 seasons
        │
        ▼
Feature Engineering
  Raw: PTS, AST, TRB, STL, BLK, TOV, FG%, 3P%, FT%, Age, G, MP
  Engineered: PER, TS%, GS_ratio, age_prime (|age-27|), pts_per_min
  Encoded: Position (PG=0, SG=1, SF=2, PF=3, C=4)
        │
        ▼
Train 4 Regression Models (80/20 split)
        │
        ▼
Auto-select best model by RMSE → Save model.pkl
        │
        ▼
Streamlit Dashboard → Real-time salary prediction
```

---

## Features Used

**Raw stats:** PTS, AST, TRB, STL, BLK, TOV, FG%, 3P%, FT%, Age, G, GS, MP

**Engineered features:**

| Feature | Formula | Purpose |
|---------|---------|---------|
| PER | (PTS + TRB + AST + STL + BLK - TOV) / MP | Overall efficiency |
| TS% | PTS / (2 × estimated possessions) | True shooting efficiency |
| GS_ratio | GS / G | Starter reliability |
| age_prime | \|Age - 27\| | Distance from peak age |
| pts_per_min | PTS / MP | Scoring efficiency per minute |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Collection | requests + BeautifulSoup (web scraping) |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-learn (Ridge, Random Forest, SVR), XGBoost |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Storage | Joblib |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/nba-salary-predictor.git
cd nba-salary-predictor
pip install -r requirements.txt
```

---

## Usage

```bash
# Step 1 — Scrape data
python scrape_data.py

# Step 2 — Train models
python models/train_models.py

# Step 3 — Launch dashboard
streamlit run dashboard/app.py
```

Or visit the **[Live Demo](https://swetha-nba-salary.streamlit.app)** directly!

---

## Project Structure

```
nba-salary-predictor/
├── data/
│   ├── scrape_data.py       ← Web scraping (Basketball-Reference)
│   └── nba_data.csv         ← Generated after scraping
├── models/
│   ├── train_models.py      ← Train + compare 4 models
│   ├── best_model.pkl       ← Saved best model
│   ├── scaler.pkl           ← Saved StandardScaler
│   └── features.pkl         ← Feature list
├── dashboard/
│   └── app.py               ← Streamlit dashboard
├── plots/                   ← Auto-generated visualizations
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

- Salary prediction is inherently noisy — market value depends on team needs, contract years, and negotiation
- Model trained on 3 seasons (2022–2024) — more historical data would improve generalization
- Future: add injury history, contract years remaining, team salary cap space as features
- Future: deploy on Streamlit Cloud for public access ✅ Done!

---

## Data Source

Stats and salary data scraped from [Basketball-Reference.com](https://www.basketball-reference.com) — 2022, 2023, 2024 NBA seasons.

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science | UCF

---

## License

MIT License
