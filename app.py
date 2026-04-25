"""
app.py - Introvert vs Extrovert Predictor (Streamlit App)
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

MODEL_PATH = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")
FEATURES_PATH = Path("feature_columns.joblib")

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not all(p.exists() for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

if model is None:
    st.error("Model dosyaları bulunamadı. Lütfen önce `python save_model.py` çalıştırın.")
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("📋 Kişilik Özellikleri")

time_alone = st.sidebar.slider("Time Spent Alone (hours/day)", 0, 11, 5)
stage_fear = st.sidebar.selectbox("Stage Fear", ["No", "Yes"])
social_event = st.sidebar.slider("Social Event Attendance (events/month)", 0, 10, 5)
going_outside = st.sidebar.slider("Going Outside (times/week)", 0, 7, 3)
drained = st.sidebar.selectbox("Drained After Socializing", ["No", "Yes"])
friends_circle = st.sidebar.slider("Friends Circle Size", 0, 15, 7)
post_freq = st.sidebar.slider("Post Frequency (posts/week)", 0, 10, 5)

predict_btn = st.sidebar.button("🔍 Predict")

# ── Build input dataframe ─────────────────────────────────────────────────────
def build_input_df():
    stage_fear_enc = 1 if stage_fear == "Yes" else 0
    drained_enc = 1 if drained == "Yes" else 0
    data = {
        "Time_spent_Alone": [time_alone],
        "Stage_fear": [stage_fear_enc],
        "Social_event_attendance": [social_event],
        "Going_outside": [going_outside],
        "Drained_after_socializing": [drained_enc],
        "Friends_circle_size": [friends_circle],
        "Post_frequency": [post_freq],
    }
    df = pd.DataFrame(data)
    df = df[feature_columns]
    df_scaled = scaler.transform(df.astype(float))
    return pd.DataFrame(df_scaled, columns=feature_columns)

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🧠 Introvert vs Extrovert Predictor")
st.markdown(
    "Bu uygulama, girdiğiniz kişilik özelliklerine göre bir kişinin **içe dönük (Introvert)** "
    "mi yoksa **dışa dönük (Extrovert)** mi olduğunu tahmin eder."
)

if predict_btn:
    input_df = build_input_df()
    prediction = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]
    extrovert_prob = float(proba[1]) * 100

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Tahmin Sonucu")
        if prediction == 1:
            st.success(
                "<div style='font-size:1.4rem; font-weight:bold; color:#155724;'>"
                "🎉 Bu kişi bir EXTROVERTtir!</div>",
                icon="🎉",
            )
        else:
            st.info(
                "<div style='font-size:1.4rem; font-weight:bold; color:#0c5460;'>"
                "🤫 Bu kişi bir INTROVERTtir!</div>",
                icon="🤫",
            )

        st.markdown(
            f"**Extrovert Olasılığı:** `{extrovert_prob:.1f}%`  \n"
            f"**Introvert Olasılığı:** `{100 - extrovert_prob:.1f}%`"
        )

    with col2:
        # Gauge chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=extrovert_prob,
                title={"text": "Extrovert Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40], "color": "lightblue"},
                        {"range": [40, 60], "color": "lightyellow"},
                        {"range": [60, 100], "color": "lightgreen"},
                    ],
                    "bar": {"color": "darkblue"},
                },
            )
        )
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # Feature importance chart
    st.subheader("📊 Feature Importance")
    try:
        importances = model.get_feature_importance()
        fi_df = pd.DataFrame(
            {"Feature": feature_columns, "Importance": importances}
        ).sort_values("Importance", ascending=True)

        fig_bar = go.Figure(
            go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker_color="steelblue",
            )
        )
        fig_bar.update_layout(
            title="Feature Importance (CatBoost)",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=350,
            margin=dict(t=50, b=40, l=160, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        st.info("Feature importance bilgisi bu model için mevcut değil.")

    st.divider()

    # Input summary table
    st.subheader("📋 Girdi Özeti")
    summary = pd.DataFrame(
        {
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
                social_event,
                going_outside,
                drained,
                friends_circle,
                post_freq,
            ],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.info("👈 Sol panelden değerleri ayarlayın ve **🔍 Predict** butonuna tıklayın.")

# ── About expander ────────────────────────────────────────────────────────────
with st.expander("ℹ️ Hakkında"):
    st.markdown(
        """
        ## 🧠 Introvert vs Extrovert Predictor

        Bu uygulama, **Kaggle Playground Series Season 5 Episode 7** yarışması için geliştirilen
        makine öğrenmesi modelini kullanmaktadır.

        ### 🎯 Görev
        Bireylerin davranışsal özelliklerine dayanarak **içe dönük (Introvert)** veya
        **dışa dönük (Extrovert)** olduklarını tahmin etmek.

        ### 📈 Model Performansı
        - **Doğruluk (Accuracy):** ~%96

        ### 🛠️ Kullanılan Teknikler
        - **KNN Imputation** — eksik değerlerin doldurulması
        - **StandardScaler** — özellik ölçeklendirme
        - **CatBoost Classifier** — gradient boosting tabanlı sınıflandırma

        ### 📋 Özellikler
        | Özellik | Açıklama |
        |---|---|
        | Time_spent_Alone | Günlük yalnız geçirilen süre (0-11 saat) |
        | Stage_fear | Sahne korkusu (Evet/Hayır) |
        | Social_event_attendance | Aylık sosyal etkinlik katılımı (0-10) |
        | Going_outside | Haftalık dışarı çıkma sıklığı (0-7) |
        | Drained_after_socializing | Sosyalleşme sonrası yorgunluk (Evet/Hayır) |
        | Friends_circle_size | Arkadaş çevresi büyüklüğü (0-15) |
        | Post_frequency | Haftalık gönderi sıklığı (0-10) |
        """
    )
