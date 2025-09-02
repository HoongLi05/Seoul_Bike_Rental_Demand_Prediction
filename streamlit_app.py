import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import time

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Seoul Bike Rental Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Load Models
# ==============================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Load clustering model and demand ranges
        kmeans, mapping, levels, ranges = joblib.load("kmeans_levels.joblib")

        # Load best prediction model
        model_data = joblib.load("final_bike_rental_model.joblib")
        if isinstance(model_data, dict) and 'model' in model_data:
            best_model = model_data['model']
        else:
            best_model = model_data

        return best_model, ranges, "✓ Model loaded successfully!", True

    except Exception as e:
        st.error(f"Error loading model: {e}. Using dummy model.")

        # Dummy fallback model (Decision Tree) if loading fails
        num_cols = ['Temperature', 'Hour', 'Humidity', 'Rainfall', 'Visibility']
        cat_cols = ['Seasons_Winter', 'Seasons_Autumn']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', 'passthrough', cat_cols)
        ])

        fallback_model = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ])

        return fallback_model, None, "⚠️ Using fallback model due to loading error", False

# Load models
best_model, ranges, model_status, model_loaded = load_models()

# ==============================
# Sidebar Inputs
# ==============================
with st.sidebar:
    st.title("🚲 Input Parameters")
    st.markdown("Adjust the parameters to predict bike rental demand.")

    st.subheader("Weather Conditions")
    temperature = st.slider("Temperature (°C)", -18.0, 40.0, 20.0, 0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 5.0, 0.0, 0.1)
    visibility = st.slider("Visibility (m)", 0.0, 2000.0, 2000.0, 10.0)

    st.subheader("Time & Season")
    hour = st.slider("Hour of the Day", 0, 23, 8)
    season = st.selectbox("Season", ["Winter", "Autumn", "Spring", "Summer"])

    if st.button("Reset to Default Values"):
        st.rerun()

# ==============================
# Main UI
# ==============================
st.title("🚲 Seoul Bike Rental Demand Prediction")
st.markdown("Predict the number of bikes that will be rented based on weather and time conditions.")

# Display model status
if model_loaded:
    st.success(model_status)
else:
    st.warning(model_status)

st.markdown("---")

# ==============================
# Prepare Input Data
# ==============================
# One-hot encode season
seasons_winter = 1 if season == "Winter" else 0
seasons_autumn = 1 if season == "Autumn" else 0

# Input DataFrame
input_df = pd.DataFrame({
    'Temperature': [temperature],
    'Hour': [hour],
    'Humidity': [humidity],
    'Rainfall': [rainfall],
    'Visibility': [visibility],
    'Seasons_Winter': [seasons_winter],
    'Seasons_Autumn': [seasons_autumn]
})

# Show input summary
with st.expander("View Input Summary"):
    st.dataframe(input_df, width="stretch")

# ==============================
# Prediction + Visualization
# ==============================
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict Demand", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            time.sleep(0.5)  # better UX

            # --- Predict ---
            try:
                predicted_count = best_model.predict(input_df)
                predicted_count = max(0, int(predicted_count[0]))
            except Exception as e:
                st.warning(f"Prediction error: {e}. Using fallback calculation...")
                base_demand = 50
                weather_effect = -rainfall * 0.5 - (humidity - 50) * 0.2
                time_effect = hour if hour <= 12 else 24 - hour
                season_effect = 20 if season in ["Spring", "Summer"] else 0
                predicted_count = max(0, int(base_demand + weather_effect + time_effect + season_effect))

            st.session_state.prediction = predicted_count

            # --- Gauge Meter ---
            # add checking for the ranges structure
            valid_ranges = False
            if ranges and hasattr(ranges, '__iter__') and len(ranges) > 0:
                # Checks if the first element is a dictionary with the correct keys
                first_range = ranges[0] if hasattr(ranges, '__getitem__') else next(iter(ranges))
                if isinstance(first_range, dict) and 'min' in first_range and 'max' in first_range:
                    valid_ranges = True
            
            if valid_ranges:
                try:
                    gauge_max = max(r['max'] for r in ranges)
                    steps = []
                    colors = ["lightgreen", "yellow", "orange"]
                    for i, r in enumerate(ranges):
                        steps.append({
                            'range': [r['min'], r['max']],
                            'color': colors[i] if i < len(colors) else 'gray'
                        })
                except Exception as e:
                    st.warning(f"Error processing ranges: {e}. Using fallback gauge.")
                    valid_ranges = False
            
            if not valid_ranges:
                gauge_max = 500
                steps = [
                    {'range': [0, 0.33*gauge_max], 'color': "lightgreen"},
                    {'range': [0.33*gauge_max, 0.66*gauge_max], 'color': "yellow"},
                    {'range': [0.66*gauge_max, gauge_max], 'color': "orange"},
                ]

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_count,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Bike Rental Demand", 'font': {'size': 24}},
                number={'font': {'size': 40}},
                gauge={
                    'axis': {'range': [None, gauge_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': steps,
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_count
                    }
                }
            ))
            gauge.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            # --- Insights ---
            st.subheader("📊 Insights")
            
            # add checking for the ranges sturcture
            label = None
            if valid_ranges:
                try:
                    for r in ranges:
                        if r['min'] <= predicted_count <= r['max']:
                            label = r.get('label', 'Unknown')
                            break
                except Exception as e:
                    st.warning(f"Error analyzing ranges: {e}")
                    label = None
            
            if label == "Low":
                st.info("Low demand expected. Consider reducing available bikes to optimize resources.")
            elif label == "Moderate":
                st.warning("Moderate demand expected. Standard operations recommended.")
            elif label == "High":
                st.error("High demand expected. Increase bike availability to meet demand.")
            else:
                # Fallback thresholds based on predicted count
                if predicted_count < 100:
                    st.info("Low demand expected. Consider reducing available bikes to optimize resources.")
                elif predicted_count < 300:
                    st.warning("Moderate demand expected. Standard operations recommended.")
                else:
                    st.error("High demand expected. Increase bike availability to meet demand.")

with col2:
    st.subheader("📈 Demand Factors")
    st.metric("Temperature", f"{temperature}°C", 
              help="Higher temperatures generally increase demand (optimal: 20-25°C)")
    st.metric("Humidity", f"{humidity}%", 
              help="High humidity may decrease demand (optimal: 40-60%)")
    st.metric("Rainfall", f"{rainfall}mm", 
              help="Rain significantly reduces bike rentals (each mm reduces demand)")
    st.metric("Visibility", f"{visibility}m", 
              help="Better visibility increases safety and demand")
    st.metric("Hour", f"{hour}:00", 
              help="Peak hours (8-9 AM, 5-6 PM) typically have higher demand")
    st.metric("Season", season, 
              help="Spring and Summer typically have 20-30% higher demand than Winter")
# ==============================
# Demand Level Indicators (Legend)
# ==============================
if 'prediction' in st.session_state:
    st.markdown("---")
    st.markdown("### 🔑 Demand Level Indicators")

    if ranges:
        cols = st.columns(len(ranges))
        for col, r in zip(cols, ranges):
            color_map = {"Low": "lightgreen", "Moderate": "yellow", "High": "orange"}
            color = color_map.get(r["label"], "gray")
            with col:
                st.markdown(
                    f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;'>"
                    f"<b>{r['label']}</b><br>{r['min']:.0f} – {r['max']:.0f}</div>",
                    unsafe_allow_html=True
                )
    else:
        cols = st.columns(3)
        for col, label in zip(cols, ["Low", "Moderate", "High"]):
            with col:
                st.markdown(
                    f"<div style='background-color: lightgray; padding: 10px; border-radius: 5px; text-align: center;'>"
                    f"<b>{label}</b><br>Range not available</div>",
                    unsafe_allow_html=True
                )

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Note: Predictions are based on historical data and patterns. Actual results may vary based on unforeseen circumstances.")
