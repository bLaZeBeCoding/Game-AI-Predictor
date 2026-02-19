# Game AI Predictor - Full Pipeline Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

print("STEP 1: Loading dataset...")
df = pd.read_csv("data/high_diamond_ranked_10min.csv")
print("Dataset loaded successfully\n")

# -----------------------------
# STEP 2: EDA
# -----------------------------
print("STEP 2: Performing EDA...")

# 1. Match outcome distribution
sns.countplot(x="blueWins", data=df)
plt.title("Match Outcome Distribution")
plt.savefig("match_outcome.png")
plt.clf()

# 2. Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("heatmap.png")
plt.clf()

# 3. Gold difference vs outcome
sns.boxplot(x="blueWins", y="blueGoldDiff", data=df)
plt.title("Gold Difference vs Match Outcome")
plt.savefig("gold_diff.png")
plt.clf()

# 4. Experience difference vs outcome
sns.boxplot(x="blueWins", y="blueExperienceDiff", data=df)
plt.title("Experience Difference vs Match Outcome")
plt.savefig("xp_diff.png")
plt.clf()

# 5. Kill difference vs outcome
df["killDiff"] = df["blueKills"] - df["redKills"]
sns.boxplot(x="blueWins", y="killDiff", data=df)
plt.title("Kill Difference vs Match Outcome")
plt.savefig("kill_diff.png")
plt.clf()

print("EDA completed. Plots saved.\n")

# -----------------------------
# STEP 3: PREPROCESSING
# -----------------------------
print("STEP 3: Preprocessing data...")

df = df.drop("gameId", axis=1)

X = df.drop("blueWins", axis=1)
y = df["blueWins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.\n")

# -----------------------------
# STEP 4: MODEL TRAINING
# -----------------------------
print("STEP 4: Training models...\n")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, y_pred_log)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print("----- FINAL RESULTS -----")
print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)
print("XGBoost Accuracy:", xgb_acc)

print("\nPipeline executed successfully.")
