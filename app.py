from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load web-specific model and scaler
model = joblib.load("web_model.pkl")
scaler = joblib.load("web_scaler.pkl")

MODEL_ACCURACY = 0.7333
MODEL_NAME = "Logistic Regression"


def build_feature_vector(gold, xp, kills, dragons, towers):
    return np.array([[gold, xp, kills, dragons, towers]])


@app.route("/")
def home():
    return render_template("index.html")

from flask import jsonify

@app.route("/predict", methods=["POST"])
def predict():

    gold = float(request.form["gold"])
    xp = float(request.form["xp"])
    kills = float(request.form["kills"])
    dragons = float(request.form["dragons"])
    towers = float(request.form["towers"])

    features = build_feature_vector(gold, xp, kills, dragons, towers)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]

    result = "BLUE TEAM WINS" if prediction == 1 else "RED TEAM WINS"

    return jsonify({
        "result": result,
        "prob": round(probability * 100, 2),
        "accuracy": 73.33,
        "model_name": MODEL_NAME
    })

if __name__ == "__main__":
    app.run(debug=True)