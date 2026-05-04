"""
NBA Player Salary Predictor
Person 2's Task: Feature Engineering + ML Models
Trains and compares 4 ML models, saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def load_and_engineer(path="data/nba_data.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records.")

    # Convert numeric columns
    numeric_cols = ["Age", "G", "MP", "PTS", "AST", "TRB", "STL",
                    "BLK", "TOV", "FG%", "3P%", "FT%", "Salary"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Salary", "PTS", "AST", "TRB"])

    # ── Engineered Features ──
    # Player Efficiency Rating (simplified PER)
    df["PER"] = (df["PTS"] + df["TRB"] + df["AST"] + df["STL"] + df["BLK"] - df["TOV"]) / df["MP"].clip(lower=1)

    # True Shooting %
    df["TS%"] = df["PTS"] / (2 * (df.get("FGA", df["PTS"] / 0.55) + 0.44 * df.get("FTA", df["PTS"] * 0.3))).clip(lower=0.01)

    # Games Started Ratio
    df["GS_ratio"] = df.get("GS", 0) / df["G"].clip(lower=1)

    # Age Prime Factor (peak around 27)
    df["age_prime"] = np.abs(df["Age"] - 27)

    # Points per minute
    df["pts_per_min"] = df["PTS"] / df["MP"].clip(lower=1)

    # Encode position
    if "Pos" in df.columns:
        le = LabelEncoder()
        df["Pos_enc"] = le.fit_transform(df["Pos"].fillna("SF").str[:2])
    else:
        df["Pos_enc"] = 2

    df["Salary_M"] = df["Salary"] / 1_000_000  # salary in millions

    features = ["Age", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK", "TOV",
                "FG%", "3P%", "FT%", "PER", "TS%", "GS_ratio",
                "age_prime", "pts_per_min", "Pos_enc"]
    features = [f for f in features if f in df.columns]

    df = df.dropna(subset=features)
    return df, features


# ─────────────────────────────────────────────
# 2. TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(df, features):
    X = df[features].values
    y = df["Salary_M"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, random_state=42, verbosity=0),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.5),
    }

    results = {}
    print("\n" + "="*55)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("="*55)

    for name, model in models.items():
        # Ridge & SVR need scaled data
        if name in ["Ridge Regression", "SVR"]:
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2,
                         "preds": preds, "scaled": name in ["Ridge Regression", "SVR"]}
        print(f"{name:<22} {rmse:>7.2f}M {mae:>7.2f}M {r2:>8.3f}")

    print("="*55)

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n🏆 Best Model: {best_name} (RMSE = ${results[best_name]['rmse']:.2f}M)")

    # Save best model + scaler
    joblib.dump(results[best_name]["model"], "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(features, "models/features.pkl")

    with open("models/best_model_name.txt", "w") as f:
        f.write(best_name)

    return results, X_test, y_test, scaler, features, best_name


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────

def plot_results(results, X_test, y_test, df, features):
    sns.set_style("darkgrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    # ── Plot 1: RMSE Comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    colors = ["#e74c3c" if r == min(rmses) else "#3498db" for r in rmses]
    bars = ax.barh(names, rmses, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("RMSE (Salary in $M)", fontsize=12)
    ax.set_title("Model Comparison — RMSE (Lower = Better)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"${val:.2f}M", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    print("✅ Saved plots/model_comparison.png")

    # ── Plot 2: Actual vs Predicted (Best Model) ──
    best_name = min(results, key=lambda k: results[k]["rmse"])
    preds = results[best_name]["preds"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, preds, alpha=0.6, color="#2ecc71", edgecolors="white", s=60)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Salary ($M)", fontsize=12)
    ax.set_ylabel("Predicted Salary ($M)", fontsize=12)
    ax.set_title(f"{best_name} — Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("✅ Saved plots/actual_vs_predicted.png")

    # ── Plot 3: Feature Importance (Random Forest) ──
    rf_model = results["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#9b59b6", edgecolor="white")
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print("✅ Saved plots/feature_importance.png")

    # ── Plot 4: Salary Distribution ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Salary_M"], bins=30, color="#e67e22", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Salary ($M)", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)
    ax.set_title("NBA Salary Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/salary_distribution.png", dpi=150)
    plt.close()
    print("✅ Saved plots/salary_distribution.png")


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NBA SALARY PREDICTOR — Model Training")
    print("=" * 55)

    df, features = load_and_engineer("data/nba_data.csv")
    results, X_test, y_test, scaler, features, best_name = train_models(df, features)
    plot_results(results, X_test, y_test, df, features)

    print("\n✅ All models trained and saved!")
    print("📁 Saved: models/best_model.pkl, models/scaler.pkl")
    print("📊 Plots saved in plots/")
    print("\nNext step: Run streamlit run dashboard/app.py")