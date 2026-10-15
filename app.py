import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fancy styling & UI layout
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        font-size: 1rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    .metric-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model using your exact pkl file name: linear_model.pkl
@st.cache_resource
def load_model():
    with open("linear_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file `linear_model.pkl` not found! Please ensure `linear_model.pkl` is in the same directory as `app.py`.")
    st.stop()

# Header Section
st.title("🏠 Residential House Price Predictor")
st.markdown("Enter property attributes below to estimate market valuation using the deployed **Linear Regression** model[cite: 2].")
st.divider()

# Two-column layout for organized input fields
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📐 Property Dimensions")
    square_footage = st.number_input("Square Footage", min_value=300, max_value=15000, value=2200, step=50, help="Total interior living space[cite: 2].")
    lot_size = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=6000, step=100, help="Total property land area[cite: 2].")
    garage_size = st.number_input("Garage Size (Cars)", min_value=0.0, max_value=6.0, value=2.0, step=1.0, help="Garage capacity in vehicles[cite: 2].")

with col2:
    st.subheader("🏢 Layout & Quality")
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=12, value=3, step=1, help="Total count of bedrooms[cite: 2].")
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5, help="Total count of bathrooms[cite: 2].")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2012, step=1, help="Original construction year[cite: 2].")
    neighborhood_quality = st.slider("Neighborhood Quality Score", min_value=1, max_value=10, value=7, step=1, help="Neighborhood rating scale (1-10)[cite: 2].")

st.divider()

# Prediction Action Button
if st.button("🔮 Calculate Estimated Property Price"):
    # Feature matrix matching the model's feature order[cite: 2]:
    # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
    features = np.array([[
        square_footage,
        num_bedrooms,
        num_bathrooms,
        year_built,
        lot_size,
        garage_size,
        neighborhood_quality
    ]])
    
    # Predict price
    prediction = model.predict(features)[0]
    
    # Result Card
    st.success("✨ Valuation Completed!")
    
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #6b7280; font-size: 1rem; margin-bottom: 0; font-weight: 600;">ESTIMATED MARKET VALUE</p>
                <h1 style="color: #10b981; font-size: 2.8rem; margin-top: 5px; margin-bottom: 5px;">${prediction:,.2f}</h1>
                <p style="color: #9ca3af; font-size: 0.85rem;">Powered by scikit-learn Linear Regression[cite: 2]</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>Developed with Streamlit • Ready for Deployment</p>", unsafe_allow_html=True)
