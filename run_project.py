print("=== Game AI Predictor Pipeline Started ===")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/high_diamond_ranked_10min.csv")
print("Dataset loaded successfully")

# Feature Engineering
df["killDiff"] = df["blueKills"] - df["redKills"]

# Features & Target
X = df.drop("blueWins", axis=1)
y = df["blueWins"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing completed")

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

print("Models trained successfully")
print("Logistic Regression Accuracy:", acc_log)
print("Random Forest Accuracy:", acc_rf)

print("=== Pipeline Finished Successfully ===")
