# =========================================================
# GAME AI PREDICTOR – FULL PIPELINE (EDA + ML + PLOTS)
# Works in: Colab, IDLE, CMD, Jenkins (headless)
# =========================================================

print("STEP 0: Importing libraries...")

import os
import pandas as pd
import numpy as np

# IMPORTANT for Jenkins (no GUI)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Create output folder for plots
os.makedirs("outputs", exist_ok=True)

# =========================================================
# STEP 1 — LOAD DATASET
# =========================================================
print("STEP 1: Loading dataset...")

df = pd.read_csv("data/high_diamond_ranked_10min.csv")
print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# Drop gameId (not useful)
df.drop("gameId", axis=1, inplace=True)

# =========================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
print("STEP 2: Performing EDA...")

# Target distribution
sns.countplot(x="blueWins", data=df)
plt.title("Match Outcome Distribution")
plt.savefig("outputs/match_outcome.png")
plt.clf()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.clf()

# Gold difference plot
sns.boxplot(x="blueWins", y="blueGoldDiff", data=df)
plt.title("Gold Difference vs Match Outcome")
plt.savefig("outputs/gold_diff.png")
plt.clf()

# Experience difference plot
sns.boxplot(x="blueWins", y="blueExperienceDiff", data=df)
plt.title("Experience Difference vs Match Outcome")
plt.savefig("outputs/exp_diff.png")
plt.clf()

# Kill difference plot
sns.boxplot(x="blueWins", y="blueKills", data=df)
plt.title("Kill Difference vs Match Outcome")
plt.savefig("outputs/kill_diff.png")
plt.clf()

print("EDA completed. Plots saved.")

# =========================================================
# STEP 3 — PREPROCESSING
# =========================================================
print("STEP 3: Preprocessing data...")

X = df.drop("blueWins", axis=1)
y = df["blueWins"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")

# =========================================================
# STEP 4 — TRAIN MODELS
# =========================================================
print("STEP 4: Training models...")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# =========================================================
# FINAL RESULTS
# =========================================================
print("\n------ FINAL RESULTS ------")
print("Logistic Regression Accuracy:", acc_log)
print("Random Forest Accuracy:", acc_rf)
print("XGBoost Accuracy:", acc_xgb)

print("\nPipeline executed successfully.")
