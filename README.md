# 🏀 NBA Player Salary Predictor
### CAP 5937 — Final Project | Spring 2026

Predicts NBA player salaries using performance statistics scraped from Basketball-Reference.com.  
Compares 4 ML regression models and serves predictions via an interactive Streamlit dashboard.

---

## 👥 Group Members
| Name | Task |
|------|------|
| Person 1 | Data scraping (Basketball-Reference + salary data) |
| Person 2 | Feature engineering + ML model training |
| Person 3 | Streamlit dashboard + report + video |

---

## 📁 Project Structure
```
nba_salary_predictor/
├── data/
│   ├── scrape_data.py       # Person 1 — scrape stats + salaries
│   └── nba_data.csv         # generated after scraping
├── models/
│   ├── train_models.py      # Person 2 — train & compare 4 models
│   ├── best_model.pkl       # saved after training
│   ├── scaler.pkl
│   └── features.pkl
├── dashboard/
│   └── app.py               # Person 3 — Streamlit UI
├── plots/                   # auto-generated after training
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Scrape data (Person 1)
```bash
cd data
python scrape_data.py
```
> This scrapes NBA stats + salaries from Basketball-Reference for 2022–2024 seasons.  
> If scraping fails, a sample dataset is auto-generated for development.

### 3. Train models (Person 2)
```bash
python models/train_models.py
```
> Trains Ridge Regression, Random Forest, XGBoost, SVR  
> Saves best model + generates plots in `/plots`

### 4. Launch dashboard (Person 3)
```bash
streamlit run dashboard/app.py
```

---

## 🤖 ML Models Compared
| Model | Type |
|-------|------|
| Ridge Regression | Linear (baseline) |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| SVR | Support Vector Machine |

**Metric:** RMSE (Root Mean Squared Error) on 20% test split  
**Best model** is automatically saved and used in the dashboard.

---

## 📊 Features Used
- Per-game stats: PTS, AST, TRB, STL, BLK, TOV
- Shooting: FG%, 3P%, FT%
- Engineered: PER (efficiency), TS%, pts/min, GS ratio, age-prime factor
- Position encoding

---

## 📋 Evaluation
- RMSE, MAE, R² on held-out test set (2022–2024 seasons)
- Cross-validation for model selection
- Feature importance plots (Random Forest)
- Actual vs Predicted scatter plot

---

## 🔗 Links
- **Code:** [GitHub Repo Link Here]
- **Data:** Basketball-Reference.com (scraped)
- **Video:** [YouTube/Drive Link Here]
- **Dashboard Demo:** [Streamlit Share Link Here]

---

## 📚 References
- Basketball-Reference.com
- Scikit-learn documentation
- XGBoost documentation
- Streamlit documentation