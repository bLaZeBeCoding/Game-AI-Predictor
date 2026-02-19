from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]

    result = "Blue Team Wins" if prediction == 1 else "Blue Team Loses"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
