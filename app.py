from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

model = joblib.load("models/model.pkl")
feature_columns = joblib.load("models/features.pkl")

# Load historical dataset
data = pd.read_csv("data/crop_prices.csv")
data['date'] = pd.to_datetime(data['date'])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    crop = request.form['crop']
    market = request.form['market']
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    month = int(request.form['month'])
    year = int(request.form['year'])

    # ----- PREDICTION -----
    input_dict = {col: 0 for col in feature_columns}

    input_dict['rainfall'] = rainfall
    input_dict['temperature'] = temperature
    input_dict['month'] = month
    input_dict['year'] = year

    crop_col = f"crop_{crop}"
    market_col = f"market_{market}"

    if crop_col in input_dict:
        input_dict[crop_col] = 1
    if market_col in input_dict:
        input_dict[market_col] = 1

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    # ----- GRAPH GENERATION -----
    filtered = data[(data['crop'] == crop) & (data['market'] == market)]
    filtered = filtered.sort_values('date')

    plt.figure()
    plt.plot(filtered['date'], filtered['price'])
    plt.title(f"{crop.capitalize()} Price Trend in {market}")
    plt.xlabel("Date")
    plt.ylabel("Price (KES)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("static", exist_ok=True)
    graph_path = "static/price_trend.png"
    plt.savefig(graph_path)
    plt.close()

    return render_template("index.html",
                           prediction=round(prediction, 2),
                           graph=True)

if __name__ == "__main__":
    app.run(debug=True)
