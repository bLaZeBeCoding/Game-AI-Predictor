from xgboost import XGBClassifier


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/high_diamond_ranked_10min.csv")

print("Dataset loaded successfully")

# Drop gameId (not useful)
df = df.drop("gameId", axis=1)

# Target variable
y = df["blueWins"]

# Features
X = df.drop("blueWins", axis=1)

print("Splitting dataset...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Logistic Regression...")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
accuracy_log = accuracy_score(y_test, y_pred_log)

print("Training Random Forest...")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", accuracy_xgb)


print("------------ RESULTS ------------")
print("Logistic Regression Accuracy:", accuracy_log)
print("Random Forest Accuracy:", accuracy_rf)
print("Pipeline executed successfully")
