from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    satisfaction_level = float(request.form["satisfaction_level"])
    years = int(request.form["years_at_company"])
    features = np.array([[age, satisfaction_level, years]])
    prediction = model.predict(features)
    result = "Will Leave" if prediction[0] == 1 else "Will Stay"
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)