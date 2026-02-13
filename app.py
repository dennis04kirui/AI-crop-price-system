from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a strong secret

# -----------------------------
# Load ML model and encoders
# -----------------------------
MODEL_FILE = "models/model.pkl"
FEATURES_FILE = "models/features.pkl"

model = joblib.load(MODEL_FILE)
features_data = joblib.load(FEATURES_FILE)  # tuple (encoders, date_min)
encoders, date_min = features_data

# Get lists of crops and markets for dropdowns
crops_list = encoders['crop'].classes_.tolist()
markets_list = encoders['market'].classes_.tolist()

# -----------------------------
# Database setup
# -----------------------------
DB_FILE = "users.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Home page
# -----------------------------
@app.route('/')
def home():
    return render_template("home.html")

# -----------------------------
# Register page
# -----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            flash("Registered successfully! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))
    return render_template("register.html")

# -----------------------------
# Login page
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            flash("Logged in successfully!", "success")
            return redirect(url_for('predict_page'))
        else:
            flash("Invalid credentials!", "danger")
            return redirect(url_for('login'))
    return render_template("login.html")

# -----------------------------
# Logout
# -----------------------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for('home'))

# -----------------------------
# Prediction page
# -----------------------------
@app.route('/predict_page')
def predict_page():
    if 'user' not in session:
        flash("Please login first to predict!", "warning")
        return redirect(url_for('login'))
    return render_template("predict.html", crops=crops_list, markets=markets_list)

# -----------------------------
# Prediction logic
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        flash("Please login first to predict!", "warning")
        return redirect(url_for('login'))

    crop = request.form['crop']
    market = request.form['market']
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    date_str = request.form['date']  # expects dd/mm/yyyy

    # Convert date to numeric
    date = pd.to_datetime(date_str, dayfirst=True)
    date_numeric = (date - date_min).days

    # Encode categorical values
    crop_encoded = encoders['crop'].transform([crop])[0]
    market_encoded = encoders['market'].transform([market])[0]

    # Create input dataframe
    input_data = pd.DataFrame([[date_numeric, crop_encoded, market_encoded, rainfall, temperature]],
                              columns=['date', 'crop', 'market', 'rainfall', 'temperature'])

    prediction = model.predict(input_data)[0]

    return render_template("predict.html", crops=crops_list, markets=markets_list,
                           prediction=round(prediction, 2))

# -----------------------------
# Trends page
# -----------------------------
@app.route('/trends', methods=['GET', 'POST'])
def trends():
    if 'user' not in session:
        flash("Please login first to view trends!", "warning")
        return redirect(url_for('login'))

    trend_data = None
    if request.method == 'POST':
        crop = request.form['crop']
        market = request.form['market']

        df = pd.read_csv("data/crop_prices.csv")
        df_filtered = df[(df['crop'] == crop) & (df['market'] == market)]
        df_filtered['date'] = pd.to_datetime(df_filtered['date'], dayfirst=True)
        df_filtered = df_filtered.sort_values('date')

        trend_data = {
            'dates': df_filtered['date'].dt.strftime('%d/%m/%Y').tolist(),
            'prices': df_filtered['price'].tolist(),
            'crop': crop,
            'market': market
        }

    return render_template("trends.html", crops=crops_list, markets=markets_list, trend_data=trend_data)

# -----------------------------
# Compare page
# -----------------------------
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if 'user' not in session:
        flash("Please login first to compare!", "warning")
        return redirect(url_for('login'))

    compare_data = None
    if request.method == 'POST':
        crop1 = request.form['crop1']
        crop2 = request.form['crop2']
        market = request.form['market']

        df = pd.read_csv("data/crop_prices.csv")
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df1 = df[(df['crop'] == crop1) & (df['market'] == market)].sort_values('date')
        df2 = df[(df['crop'] == crop2) & (df['market'] == market)].sort_values('date')

        compare_data = {
            'dates': df1['date'].dt.strftime('%d/%m/%Y').tolist(),
            'crop1_prices': df1['price'].tolist(),
            'crop2_prices': df2['price'].tolist(),
            'crop1': crop1,
            'crop2': crop2,
            'market': market
        }

    return render_template("compare.html", crops=crops_list, markets=markets_list, compare_data=compare_data)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
