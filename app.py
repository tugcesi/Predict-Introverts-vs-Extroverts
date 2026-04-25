import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🧠 Introvert vs Extrovert Predictor",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
MODEL_PATH    = Path("model.joblib")
SCALER_PATH   = Path("scaler.joblib")
FEATURES_PATH = Path("feature_columns.joblib")


@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return None, None, None
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(FEATURES_PATH),
    )


model, scaler, feature_columns = load_artifacts()

if model is None:
    st.error(
        "Model dosyaları bulunamadı. Lütfen önce `python save_model.py` çalıştırın."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input builder — encoding identical to save_model.py
# ---------------------------------------------------------------------------

def build_input_df(
    time_alone: int,
    stage_fear: str,
    social_attendance: int,
    going_outside: int,
    drained: str,
    friends_size: int,
    post_freq: int,
) -> pd.DataFrame:
    """Build a scaled input DataFrame that matches the training pipeline."""
    stage_fear_enc   = 1 if stage_fear == "Yes" else 0
    drained_enc      = 1 if drained    == "Yes" else 0

    raw = {
        "Time_spent_Alone":         float(time_alone),
        "Stage_fear":               float(stage_fear_enc),
        "Social_event_attendance":  float(social_attendance),
        "Going_outside":            float(going_outside),
        "Drained_after_socializing": float(drained_enc),
        "Friends_circle_size":      float(friends_size),
        "Post_frequency":           float(post_freq),
    }

    df_raw = pd.DataFrame([raw])[feature_columns]
    scaled = scaler.transform(df_raw)
    return pd.DataFrame(scaled, columns=feature_columns)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Kişilik Özellikleri")

time_alone        = st.sidebar.slider("⏱ Time Spent Alone (saat/gün)", 0, 11, 5)
stage_fear        = st.sidebar.selectbox("🎤 Stage Fear", ["No", "Yes"])
social_attendance = st.sidebar.slider("🎉 Social Event Attendance", 0, 10, 5)
going_outside     = st.sidebar.slider("🚶 Going Outside (gün/hafta)", 0, 7, 3)
drained           = st.sidebar.selectbox("😴 Drained After Socializing", ["No", "Yes"])
friends_size      = st.sidebar.slider("👥 Friends Circle Size", 0, 15, 7)
post_freq         = st.sidebar.slider("📱 Post Frequency", 0, 10, 5)

predict_btn = st.sidebar.button("🔍 Predict")

# ---------------------------------------------------------------------------
# Main page header
# ---------------------------------------------------------------------------
st.title("🧠 Introvert vs Extrovert Predictor")
st.markdown(
    "Bu uygulama, girilen kişilik özelliklerine göre bir kişinin "
    "**Introvert (İçe Dönük)** mi yoksa **Extrovert (Dışa Dönük)** mu "
    "olduğunu tahmin eder."
)
st.divider()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_btn:
    X_input = build_input_df(
        time_alone, stage_fear, social_attendance,
        going_outside, drained, friends_size, post_freq,
    )

    prediction = model.predict(X_input)[0]
    proba      = model.predict_proba(X_input)[0]
    prob_extrovert = float(np.clip(proba[1], 0.0, 1.0))
    prob_introvert = float(np.clip(proba[0], 0.0, 1.0))

    # -- Result banner --
    col1, col2 = st.columns([2, 1])
    with col1:
        if prediction == 1:
            st.success("🎉 Bu kişi bir **EXTROVERTtir** – Dışa dönük!")
        else:
            st.info("🤫 Bu kişi bir **INTROVERTtir** – İçe dönük!")

        st.metric("Extrovert Olasılığı", f"{prob_extrovert*100:.1f}%")
        st.metric("Introvert Olasılığı", f"{prob_introvert*100:.1f}%")

    # -- Gauge chart --
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_extrovert * 100,
            title={"text": "Extrovert Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green" if prob_extrovert >= 0.5 else "steelblue"},
                "steps": [
                    {"range": [0,  40], "color": "lightblue"},
                    {"range": [40, 60], "color": "lightyellow"},
                    {"range": [60, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%"},
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # -- Feature Importance --
    st.subheader("📊 Feature Importance")
    importances = model.get_feature_importance()
    feat_names  = feature_columns

    fig_fi = go.Figure(go.Bar(
        x=importances,
        y=feat_names,
        orientation="h",
        marker_color="steelblue",
    ))
    fig_fi.update_layout(
        xaxis_title="Importance",
        yaxis={"autorange": "reversed"},
        height=350,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # -- Input summary table --
    st.subheader("📋 Girilen Değerler")
    summary = pd.DataFrame({
        "Özellik": [
            "Time Spent Alone",
            "Stage Fear",
            "Social Event Attendance",
            "Going Outside",
            "Drained After Socializing",
            "Friends Circle Size",
            "Post Frequency",
        ],
        "Değer": [
            time_alone,
            stage_fear,
            social_attendance,
            going_outside,
            drained,
            friends_size,
            post_freq,
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.info("👈 Soldaki panelden değerleri girin ve **🔍 Predict** butonuna tıklayın.")
