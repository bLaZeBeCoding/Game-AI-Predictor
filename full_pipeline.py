# ==========================================
# GAME AI MATCH WIN PREDICTION PIPELINE
# Runs in: Jupyter, Colab, CMD, IDLE, Jenkins
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

# ==============================
# DEBUG INFO (For Jenkins marks)
# ==============================
print("========== DEBUG INFO ==========")
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output directory:", OUTPUT_DIR)
print("================================\n")


# ==============================
# STEP 1 — LOAD DATASET
# ==============================
print("STEP 1: Loading dataset...")

df = pd.read_csv("data/high_diamond_ranked_10min.csv")
print("Dataset loaded successfully\n")


# ==============================
# STEP 2 — EDA + SAVE PLOTS
# ==============================
print("STEP 2: Running EDA and saving plots...")

# Match outcome distribution
sns.countplot(x="blueWins", data=df)
plt.title("Match Outcome Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "match_outcome.png"))
plt.close()

# Correlation heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# Gold difference vs win
sns.boxplot(x="blueWins", y="blueGoldDiff", data=df)
plt.title("Gold Difference vs Win")
plt.savefig(os.path.join(OUTPUT_DIR, "gold_diff.png"))
plt.close()

# Experience difference vs win
sns.boxplot(x="blueWins", y="blueExperienceDiff", data=df)
plt.title("Experience Difference vs Win")
plt.savefig(os.path.join(OUTPUT_DIR, "exp_diff.png"))
plt.close()

# Kill difference vs win
df["killDiff"] = df["blueKills"] - df["redKills"]
sns.boxplot(x="blueWins", y="killDiff", data=df)
plt.title("Kill Difference vs Win")
plt.savefig(os.path.join(OUTPUT_DIR, "kill_diff.png"))
plt.close()

print("EDA completed. Plots saved.\n")


# ==============================
# STEP 3 — PREPROCESSING
# ==============================
print("STEP 3: Preprocessing data...")

X = df.drop(["blueWins","gameId"], axis=1)
y = df["blueWins"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.\n")


# ==============================
# STEP 4 — TRAIN MODELS
# ==============================
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
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

print("\n========== FINAL RESULTS ==========")
print("Logistic Regression Accuracy:", acc_log)
print("Random Forest Accuracy:", acc_rf)
print("XGBoost Accuracy:", acc_xgb)


# ==============================
# STEP 5 — CONFUSION MATRIX
# ==============================
print("\nSTEP 5: Generating evaluation plots...")

cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("Confusion matrix saved!")


# ==============================
# STEP 6 — FEATURE IMPORTANCE
# ==============================
feature_importance = rf_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

print("Feature importance saved!")


# ==============================
# STEP 7 — BEST MODEL SUMMARY
# ==============================
accuracies = {
    "Logistic Regression": acc_log,
    "Random Forest": acc_rf,
    "XGBoost": acc_xgb
}

best_model = max(accuracies, key=accuracies.get)

print("\nBest Model:", best_model)
print("Accuracy:", accuracies[best_model])

print("\nAll plots saved inside /outputs folder")
print("Pipeline executed successfully.")
