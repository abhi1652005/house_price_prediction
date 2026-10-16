import streamlit as st
import pickle
import numpy as np
import sys

# --- COMPATIBILITY FIX FOR PICKLE LOAD ---
try:
    import numpy._core
except ImportError:
    try:
        import numpy.core as core
        sys.modules['numpy._core'] = core
        sys.modules['numpy._core.multiarray'] = core.multiarray
    except Exception:
        pass
# ------------------------------------------

# Page Configuration
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom Styling & Modern Theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    h1 {
        color: #818cf8;
        text-align: center;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
        font-size: 16px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #34d399;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #6ee7b7;
        font-size: 32px;
        font-weight: 700;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model without changing the pkl file name
@st.cache_resource
def load_model():
    with open('linear_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model file: {e}")
    st.stop()

# Header
st.markdown("<h1>🏠 Real Estate Valuation AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict property market values instantly using machine learning</p>", unsafe_allow_html=True)

# Input Layout
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("### 📐 Dimensions")
    square_footage = st.number_input("Square Footage", min_value=300, max_value=10000, value=2000, step=50)
    lot_size = st.number_input("Lot Size (sq ft)", min_value=500, max_value=50000, value=5000, step=100)
    garage_size = st.number_input("Garage Size (Cars)", min_value=0, max_value=5, value=2, step=1)

with col2:
    st.markdown("### 🛋️ Layout & Build")
    num_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    num_bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010, step=1)

neighborhood_quality = st.slider("Neighborhood Quality (1 - 10 Scale)", min_value=1, max_value=10, value=7)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Logic matching exact feature training order
if st.button("Calculate Property Value", use_container_width=True):
    features = [
        float(square_footage),
        float(num_bedrooms),
        float(num_bathrooms),
        float(year_built),
        float(lot_size),
        float(garage_size),
        float(neighborhood_quality)
    ]
    
    try:
        prediction = model.predict([features])
        estimated_price = float(prediction[0])
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Property Value</div>
                <div class="metric-value">${estimated_price:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
