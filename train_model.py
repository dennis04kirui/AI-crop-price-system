import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

CSV_FILE = "data/crop_prices.csv"
MODEL_FILE = "models/model.pkl"
FEATURES_FILE = "models/features.pkl"

# Load data
df = pd.read_csv(CSV_FILE)

# Encode categorical columns
encoders = {}
for col in ['crop', 'market']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Convert date to numeric
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
date_min = df['date'].min()
df['date'] = (df['date'] - date_min).dt.days

# Features and target
X = df[['date', 'crop', 'market', 'rainfall', 'temperature']]
y = df['price']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_FILE)

# Save encoders and date_min together as a tuple
joblib.dump((encoders, date_min), FEATURES_FILE)

print("Training completed and files saved.")
