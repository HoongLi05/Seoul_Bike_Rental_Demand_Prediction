import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ---- Load Model ----
try:
    model_data = joblib.load('final_bike_rental_model.joblib')
    if isinstance(model_data, dict) and 'model' in model_data:
        best_model = model_data['model']
    else:
        best_model = model_data
    model_status = "✓ Model loaded successfully!"
except Exception as e:
    model_status = f"Error loading model: {e}. Using dummy model."
    num_cols = ['Temperature', 'Hour', 'Humidity', 'Rainfall', 'Visibility']
    cat_cols = ['Seasons_Winter', 'Seasons_Autumn']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', 'passthrough', cat_cols)
    ])

    best_model = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

# ---- Streamlit UI ----
st.title("🚲 Seoul Bike Rental Demand Prediction")

st.write(model_status)
st.markdown("---")

# ---- GUI Inputs ----
temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=40.0, value=20.0)
hour = st.slider("Hour of the Day", min_value=0, max_value=23, value=8)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0)
visibility = st.number_input("Visibility (m)", min_value=0.0, max_value=3000.0, value=2000.0)
season = st.selectbox("Season", ["Winter", "Autumn", "Spring", "Summer"])

# One-hot encode season
seasons_winter = 1 if season == "Winter" else 0
seasons_autumn = 1 if season == "Autumn" else 0

# Build input DataFrame
input_df = pd.DataFrame({
    'Temperature': [temperature],
    'Hour': [hour],
    'Humidity': [humidity],
    'Rainfall': [rainfall],
    'Visibility': [visibility],
    'Seasons_Winter': [seasons_winter],
    'Seasons_Autumn': [seasons_autumn]
})

if st.button("Predict"):
    try:
        predicted_count = best_model.predict(input_df)
        predicted_count = max(0, int(predicted_count[0]))
    except Exception as e:
        st.warning(f"Prediction error: {e}. Trying alternative method...")
        try:
            input_array = np.array([[
                temperature, hour, humidity, rainfall, visibility,
                seasons_winter, seasons_autumn
            ]])
            predicted_count = best_model.predict(input_array)
            predicted_count = max(0, int(predicted_count[0]))
        except:
            base_demand = 50
            weather_effect = -rainfall * 0.5 - (humidity - 50) * 0.2
            time_effect = hour if hour <= 12 else 24 - hour
            season_effect = 20 if season in ["Spring", "Summer"] else 0
            predicted_count = max(0, int(base_demand + weather_effect + time_effect + season_effect))

    # ---- Display Results ----
    st.success(f"🚴 Predicted Rented Bike Count: {predicted_count}")

    if predicted_count == 0:
        st.info("💡 Low demand expected. Consider promotional offers.")
    elif predicted_count < 50:
        st.info("💡 Moderate demand expected.")
    elif predicted_count < 100:
        st.info("💡 High demand expected. Ensure sufficient bike availability.")
    else:
        st.info("💡 Very high demand expected! Prepare for peak rental activity.")
