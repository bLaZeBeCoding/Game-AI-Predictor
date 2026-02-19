import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv("data/high_diamond_ranked_10min.csv")

# Target
y = df["blueWins"]

# Features used by website
X = df[[
    "blueGoldDiff",
    "blueExperienceDiff",
    "blueKills",
    "blueDragons",
    "blueTowersDestroyed"
]]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save BOTH scaler + model
joblib.dump((model, scaler), "model.pkl")

print("MODEL TRAINED & SAVED AS model.pkl")
