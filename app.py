from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, Length, EqualTo
from wtforms.widgets import NumberInput
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-GUI backend for Flask
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.linear_model import LinearRegression

# -------------------------
# App Config
# -------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_key_change_this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# -------------------------
# User Model
# -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# -------------------------
# Forms
# -------------------------
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class PredictForm(FlaskForm):
    crop = SelectField('Crop', validators=[DataRequired()], choices=[])
    county = SelectField('County', validators=[DataRequired()], choices=[])
    year = IntegerField('Year', validators=[DataRequired()], widget=NumberInput())
    submit = SubmitField('Predict')

# -------------------------
# Load CSV Safely
# -------------------------
csv_path = os.path.join("data", "crop_prices_full.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("CSV file not found inside data folder!")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.lower()

required_cols = ['date', 'market', 'crop', 'price']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing in CSV file!")

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['year'] = df['date'].dt.year

df['county'] = df['market'].astype(str).str.title()
df['crop'] = df['crop'].astype(str).str.title()
df = df.dropna(subset=['price'])

# Dropdown values
crops = sorted(df['crop'].unique())
counties = sorted(df['county'].unique())
years = sorted(df['year'].unique())

# -------------------------
# Login Manager
# -------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------------
# Register
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'danger')
        else:
            user = User(username=form.username.data, password=form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)

# -------------------------
# Login
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
            login_user(user)
            return redirect(url_for('predict'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

# -------------------------
# Logout
# -------------------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

# -------------------------
# Predict
# -------------------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictForm()
    form.crop.choices = [(c, c) for c in crops]
    form.county.choices = [(c, c) for c in counties]

    price_prediction = None
    if form.validate_on_submit():
        try:
            crop_input = form.crop.data
            county_input = form.county.data
            year_input = int(form.year.data)

            # Filter historical data for the selected crop + county
            df_subset = df[(df['crop'] == crop_input) & (df['county'] == county_input)]

            if len(df_subset) >= 2:
                X = df_subset[['year']]
                y = df_subset['price']
                model = LinearRegression()
                model.fit(X, y)
                price_prediction = round(model.predict([[year_input]])[0], 2)
            else:
                price_prediction = "Not enough historical data"

        except Exception as e:
            print("Prediction error:", e)
            price_prediction = "Prediction failed. Check input."

    return render_template('predict.html', form=form, price=price_prediction)

# -------------------------
# Trends
# -------------------------
@app.route('/trends', methods=['GET', 'POST'])
@login_required
def trends():
    selected_crop = request.form.get('crop') or crops[0]
    selected_county = request.form.get('county') or counties[0]

    filtered = df[(df['crop'] == selected_crop) & (df['county'] == selected_county)]
    chart_data = None

    if not filtered.empty:
        yearly = filtered.groupby('year')['price'].mean().reset_index()
        plt.figure(figsize=(8,4))
        plt.plot(yearly['year'], yearly['price'], marker='o')
        plt.title(f'{selected_crop} Trend in {selected_county}')
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_data = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    return render_template('trends.html',
                           chart_data=chart_data,
                           crops=crops,
                           counties=counties,
                           selected_crop=selected_crop,
                           selected_county=selected_county)

# -------------------------
# Compare
# -------------------------
@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    crop1 = request.form.get('crop1') or crops[0]
    crop2 = request.form.get('crop2') or (crops[1] if len(crops) > 1 else crops[0])
    selected_county = request.form.get('county') or counties[0]

    filtered1 = df[(df['crop'] == crop1) & (df['county'] == selected_county)]
    filtered2 = df[(df['crop'] == crop2) & (df['county'] == selected_county)]
    chart_data = None

    if not filtered1.empty and not filtered2.empty:
        yearly1 = filtered1.groupby('year')['price'].mean().reset_index()
        yearly2 = filtered2.groupby('year')['price'].mean().reset_index()

        plt.figure(figsize=(8,4))
        plt.plot(yearly1['year'], yearly1['price'], marker='o', label=crop1)
        plt.plot(yearly2['year'], yearly2['price'], marker='o', label=crop2)
        plt.title(f'{crop1} vs {crop2} in {selected_county}')
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_data = base64.b64encode(buf.getvalue()).decode()
        plt.close()

    return render_template('compare.html',
                           chart_data=chart_data,
                           crops=crops,
                           counties=counties,
                           selected_crop1=crop1,
                           selected_crop2=crop2,
                           selected_county=selected_county)

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
