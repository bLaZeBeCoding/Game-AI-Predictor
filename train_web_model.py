import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Training WEB APP model...")

# Load dataset
df = pd.read_csv("data/high_diamond_ranked_10min.csv")

# Features used in website
features = [
    "blueGoldDiff",
    "blueExperienceDiff",
    "blueKills",
    "blueDragons",
    "blueTowersDestroyed"
]

X = df[features]
y = df["blueWins"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save BOTH scaler + model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Web model saved successfully!")
