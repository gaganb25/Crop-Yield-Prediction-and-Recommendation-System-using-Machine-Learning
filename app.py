import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}

/* HERO */
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg,#052e16,#14532d,#022c22);
    box-shadow: 0px 0px 25px rgba(34,197,94,0.35);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #4ade80;
    text-align: center;
}

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(135deg,#064e3b,#14532d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(34,197,94,0.25);
    margin-bottom: 20px;
}

.metric-title {
    font-size: 20px;
    color: #d1fae5;
}

.metric-value {
    font-size: 38px;
    font-weight: bold;
    color: #4ade80;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 18px;
    border: none;
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#16a34a,#15803d);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* SECTION TITLES */
.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #4ade80;
    margin-bottom: 15px;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 30px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")
df = df.dropna()

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <div class="hero-title">
        🌱 Crop Yield Prediction and Recommendation System
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
state = st.sidebar.selectbox(
    "🌍 Select State",
    sorted(df['State_Name'].unique())
)

state_df = df[df['State_Name'] == state]

crop = st.sidebar.selectbox(
    "🌾 Select Crop",
    sorted(state_df['Crop'].unique())
)

crop_df = state_df[state_df['Crop'] == crop]

season = st.sidebar.selectbox(
    "☀️ Select Season",
    sorted(crop_df['Season'].unique())
)

area = st.sidebar.number_input(
    "📏 Enter Area",
    min_value=1.0,
    value=100.0
)

year = st.sidebar.number_input(
    "📅 Enter Year",
    min_value=1990,
    max_value=2100,
    value=2002
)

# ---------------- RECOMMENDATION ----------------
best_crop = (
    state_df.groupby('Crop')['Production']
    .mean()
    .idxmax()
)

# ---------------- TOP METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🌍 Selected State</div>
        <div class="metric-value">{state}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🌾 Selected Crop</div>
        <div class="metric-value">{crop}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🏆 Recommended Crop</div>
        <div class="metric-value">{best_crop}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FILTER DATA ----------------
filtered_df = df[
    (df['State_Name'] == state) &
    (df['Crop'] == crop) &
    (df['Season'] == season)
]

# ---------------- MODEL ----------------
if not filtered_df.empty:

    X = filtered_df[['Area', 'Crop_Year']]
    y = filtered_df['Production']

    model = RandomForestRegressor()
    model.fit(X, y)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction
    if st.button("🚀 Predict Crop Yield"):

        prediction = model.predict([[area, year]])

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌱 Predicted Production</div>
            <div class="metric-value">{prediction[0]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- GRAPH ----------------
    st.markdown(
        "<br><div class='section-title'>📊 Production Trend Analysis</div>",
        unsafe_allow_html=True
    )

    graph_df = filtered_df.groupby(
        'Crop_Year'
    )['Production'].mean().reset_index()

    graph_df = graph_df.sort_values('Crop_Year')

    st.line_chart(
        graph_df.set_index('Crop_Year')
    )

    # ---------------- INSIGHTS ----------------
    st.markdown(
        "<br><div class='section-title'>💡 AI Insights</div>",
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:
        st.info(f"🌾 Best crop in {state}: {best_crop}")

    with i2:
        st.info(f"📅 Data analyzed till year {graph_df['Crop_Year'].max()}")

    with i3:
        st.info("🤖 Model Used: Random Forest Regression")

else:
    st.warning("No data available for selected combination")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
🌱 Smart Agriculture using Machine Learning | Built with Streamlit & Random Forest
</div>
""", unsafe_allow_html=True)