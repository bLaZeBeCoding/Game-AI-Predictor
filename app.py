from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Accuracy from training pipeline (same value Jenkins printed)
MODEL_ACCURACY = 0.7333
MODEL_NAME = "Logistic Regression"

# Convert 5 user inputs → 39 feature vector expected by model
def build_feature_vector(gold, xp, kills, dragons, towers):
    features = np.zeros(39)

    # IMPORTANT FEATURES discovered during EDA
    features[0] = gold          # blueGoldDiff
    features[1] = xp            # blueExperienceDiff
    features[2] = kills         # blueKills
    features[3] = dragons       # blueDragons
    features[4] = towers        # blueTowersDestroyed

    return features.reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gold = float(request.form["gold"])
    xp = float(request.form["xp"])
    kills = float(request.form["kills"])
    dragons = float(request.form["dragons"])
    towers = float(request.form["towers"])

    features = build_feature_vector(gold, xp, kills, dragons, towers)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    result = "BLUE TEAM WINS" if prediction == 1 else "RED TEAM WINS"

    return render_template(
        "index.html",
        prediction_text=result,
        prob=round(probability * 100, 2),
        accuracy=MODEL_ACCURACY,
        model_name=MODEL_NAME
    )

if __name__ == "__main__":
    app.run(debug=True)
