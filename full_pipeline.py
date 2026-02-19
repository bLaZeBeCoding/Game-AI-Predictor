# ================================
# Game AI Predictor – FULL PIPELINE
# Works in: Jenkins, CMD, IDLE, Colab
# Saves plots automatically
# ================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # IMPORTANT for Jenkins (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ================================
# FORCE OUTPUT FOLDER (WORKS EVERYWHERE)
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving plots to:", OUTPUT_DIR)

# ================================
# STEP 1: LOAD DATA
# ================================
print("\nSTEP 1: Loading dataset...")

data_path = os.path.join(BASE_DIR, "data", "high_diamond_ranked_10min.csv")
df = pd.read_csv(data_path)

print("Dataset loaded successfully")

# ================================
# STEP 2: EDA + SAVE PLOTS
# ================================
print("\nSTEP 2: Performing EDA...")

# 1 Match outcome distribution
plt.figure(figsize=(6,4))
sns.countplot(x="blueWins", data=df)
plt.title("Match Outcome Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "match_outcome.png"))
plt.close()

# 2 Correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# 3 Gold diff vs win
plt.figure(figsize=(6,4))
sns.boxplot(x="blueWins", y="blueGoldDiff", data=df)
plt.title("Gold Difference vs Match Outcome")
plt.savefig(os.path.join(OUTPUT_DIR, "gold_diff.png"))
plt.close()

# 4 Experience diff vs win
plt.figure(figsize=(6,4))
sns.boxplot(x="blueWins", y="blueExperienceDiff", data=df)
plt.title("Experience Difference vs Match Outcome")
plt.savefig(os.path.join(OUTPUT_DIR, "exp_diff.png"))
plt.close()

# 5 Kill diff vs win
plt.figure(figsize=(6,4))
sns.boxplot(x="blueWins", y="blueKills", data=df)
plt.title("Kill Difference vs Match Outcome")
plt.savefig(os.path.join(OUTPUT_DIR, "kill_diff.png"))
plt.close()

print("EDA completed. Plots saved.")

# ================================
# STEP 3: PREPROCESSING
# ================================
print("\nSTEP 3: Preprocessing data...")

X = df.drop("blueWins", axis=1)
y = df["blueWins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")

# ================================
# STEP 4: TRAIN MODELS
# ================================
print("\nSTEP 4: Training models...")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
pred_log = log_model.predict(X_test_scaled)
acc_log = accuracy_score(y_test, pred_log)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

# XGBoost
xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, pred_xgb)

# ================================
# FINAL RESULTS
# ================================
print("\n----- FINAL RESULTS -----")
print("Logistic Regression Accuracy:", acc_log)
print("Random Forest Accuracy:", acc_rf)
print("XGBoost Accuracy:", acc_xgb)

print("\nPipeline executed successfully.")
