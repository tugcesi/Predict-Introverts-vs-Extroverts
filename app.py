"""
app.py - Introvert vs Extrovert Streamlit App
Run: streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="🧠 Introvert vs Extrovert Predictor",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH    = Path("model.joblib")
SCALER_PATH   = Path("scaler.joblib")
FEATURES_PATH = Path("feature_columns.joblib")


@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return None, None, None
    model          = joblib.load(MODEL_PATH)
    scaler         = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

if model is None:
    st.error(
        "⚠️ Model dosyaları bulunamadı. Lütfen önce `python save_model.py` çalıştırın."
    )
    st.stop()

# ── Title ────────────────────────────────────────────────────────────
st.title("🧠 Introvert vs Extrovert Predictor")
st.markdown("Kişilik özelliklerinize göre **İçe Dönük mü, Dışa Dönük mü** olduğunuzu tahmin edin.")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("👤 Kişi Bilgileri")

time_alone      = st.sidebar.slider("Yalnız Geçirilen Zaman (Time_spent_Alone)", 0, 11, 5)
stage_fear      = st.sidebar.selectbox("Sahne Korkusu (Stage_fear)", ["No", "Yes"])
social_event    = st.sidebar.slider("Sosyal Etkinliklere Katılım (Social_event_attendance)", 0, 10, 5)
going_outside   = st.sidebar.slider("Dışarı Çıkma Sıklığı (Going_outside)", 0, 7, 3)
drained         = st.sidebar.selectbox("Sosyalleşme Sonrası Yorgunluk (Drained_after_socializing)", ["No", "Yes"])
friends_circle  = st.sidebar.slider("Arkadaş Çevresi Büyüklüğü (Friends_circle_size)", 0, 15, 7)
post_frequency  = st.sidebar.slider("Gönderi Sıklığı (Post_frequency)", 0, 10, 5)

predict_btn = st.sidebar.button("🔍 Tahmin Et", use_container_width=True)


# ── Build Input ──────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    row = {
        "Time_spent_Alone":          float(time_alone),
        "Stage_fear":                1 if stage_fear == "Yes" else 0,
        "Social_event_attendance":   float(social_event),
        "Going_outside":             float(going_outside),
        "Drained_after_socializing": 1 if drained == "Yes" else 0,
        "Friends_circle_size":       float(friends_circle),
        "Post_frequency":            float(post_frequency),
    }
    df = pd.DataFrame([row])[feature_columns].astype(float)
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_columns)


# ── Prediction ───────────────────────────────────────────────────────
if predict_btn:
    X_input = build_input_df()
    proba         = model.predict_proba(X_input)[0]
    prob_extrovert = float(np.clip(proba[1], 0.0, 1.0))
    prediction    = int(model.predict(X_input)[0])

    col1, col2 = st.columns([1, 1])

    with col1:
        if prediction == 1:
            st.success(f"🎉 Bu kişi bir **EXTROVERTtir** – Dışa Dönük!\n\nDışa Dönüklük Olasılığı: **{prob_extrovert * 100:.1f}%**")
        else:
            st.info(f"🤫 Bu kişi bir **INTROVERTtir** – İçe Dönük!\n\nDışa Dönüklük Olasılığı: **{prob_extrovert * 100:.1f}%**")

        # Girdi özet tablosu
        st.markdown("### 📋 Girilen Değerler")
        summary = pd.DataFrame({
            "Özellik": feature_columns,
            "Değer": [
                time_alone, stage_fear, social_event, going_outside,
                drained, friends_circle, post_frequency
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_extrovert * 100,
            title={"text": "Dışa Dönüklük Olasılığı (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "royalblue"},
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
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature Importance
    st.markdown("### 📊 Özellik Önemi")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": feature_columns, "importance": importances})
            .sort_values("importance", ascending=True)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker_color="steelblue",
        ))
        fig_fi.update_layout(
            title="Özellik Önemi",
            xaxis_title="Önem",
            yaxis_title="Özellik",
            height=350,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

else:
    st.info("👈 Sol panelden kişi bilgilerini girin ve **🔍 Tahmin Et** butonuna tıklayın.")

# ── About ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    ### 🧠 Introvert vs Extrovert Predictor

    Bu uygulama **Kaggle Playground Series S5E7** yarışması için geliştirilen
    makine öğrenmesi modelini kullanarak bir kişinin **içe dönük (Introvert)**
    mi yoksa **dışa dönük (Extrovert)** mi olduğunu tahmin eder.

    **Kullanılan Teknikler:**
    - KNN Imputation (eksik değer doldurma)
    - StandardScaler (özellik ölçeklendirme)
    - CatBoost Classifier

    **Model Doğruluğu:** ~%96

    **Veri Kaynağı:** [Kaggle – Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7)
    """)