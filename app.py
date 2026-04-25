import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

MODEL_PATH = Path("model.joblib")
SCALER_PATH = Path("scaler.joblib")
FEATURES_PATH = Path("feature_columns.joblib")

st.set_page_config(
    page_title="Introvert vs Extrovert Predictor",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols


model, scaler, feature_cols = load_artifacts()

if model is None:
    st.error(
        "Model files not found. Please run `python save_model.py` first to generate "
        "`model.joblib`, `scaler.joblib`, and `feature_columns.joblib`."
    )
    st.stop()

st.title("🧠 Introvert vs Extrovert Predictor")
st.markdown("Predict whether a person is an **Introvert** or **Extrovert** based on behavioral traits.")

with st.sidebar:
    st.header("🔧 Input Features")

    time_alone = st.slider("Time Spent Alone (hours/day)", 0, 11, 5)
    stage_fear = st.selectbox("Stage Fear", ["No", "Yes"])
    social_events = st.slider("Social Event Attendance (per month)", 0, 10, 5)
    going_outside = st.slider("Going Outside (days/week)", 0, 7, 3)
    drained = st.selectbox("Drained After Socializing", ["No", "Yes"])
    friends = st.slider("Friends Circle Size", 0, 15, 7)
    post_freq = st.slider("Post Frequency (per week)", 0, 10, 5)

    predict_btn = st.button("🔍 Predict", use_container_width=True)


def build_input_df():
    stage_fear_enc = 1 if stage_fear == "Yes" else 0
    drained_enc = 1 if drained == "Yes" else 0
    data = {
        "Time_spent_Alone": [time_alone],
        "Stage_fear": [stage_fear_enc],
        "Social_event_attendance": [social_events],
        "Going_outside": [going_outside],
        "Drained_after_socializing": [drained_enc],
        "Friends_circle_size": [friends],
        "Post_frequency": [post_freq],
    }
    input_df = pd.DataFrame(data)[feature_cols]
    scaled = scaler.transform(input_df)
    return pd.DataFrame(scaled, columns=feature_cols)


def show_about():
    with st.expander("ℹ️ About This App"):
        st.markdown(
            """
            **Introvert vs Extrovert Predictor** uses a CatBoost classifier trained on the
            [Kaggle Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7) dataset.

            **Techniques used:**
            - KNN Imputation for missing values
            - Standard Scaling for feature normalization
            - CatBoost Classifier for prediction

            **Model performance:** ~96% accuracy on the test split.

            Run `python save_model.py` to retrain the model with the latest `train.csv`.
            """
        )


if predict_btn:
    input_scaled = build_input_df()
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    extrovert_prob = proba[1]
    introvert_prob = proba[0]
    label = "Extrovert" if prediction == 1 else "Introvert"

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Prediction Result")
        if label == "Extrovert":
            st.success(f"### 🌟 {label}", icon="✅")
            st.markdown("This person shows **extroverted** behavioral patterns.")
        else:
            st.info(f"### 🔵 {label}", icon="ℹ️")
            st.markdown("This person shows **introverted** behavioral patterns.")

        st.metric("Extrovert Probability", f"{extrovert_prob:.1%}")
        st.metric("Introvert Probability", f"{introvert_prob:.1%}")

    with col2:
        st.subheader("📊 Extrovert Probability Gauge")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=extrovert_prob * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ecc71" if label == "Extrovert" else "#3498db"},
                    "steps": [
                        {"range": [0, 40], "color": "#d6eaf8"},
                        {"range": [40, 60], "color": "#d5f5e3"},
                        {"range": [60, 100], "color": "#a9dfbf"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                title={"text": "Extrovert Score"},
            )
        )
        gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("📈 Feature Importance")
        try:
            importances = model.get_feature_importance()
            fi_df = pd.DataFrame(
                {"Feature": feature_cols, "Importance": importances}
            ).sort_values("Importance", ascending=True)
            bar_fig = go.Figure(
                go.Bar(
                    x=fi_df["Importance"],
                    y=fi_df["Feature"],
                    orientation="h",
                    marker_color="#5dade2",
                )
            )
            bar_fig.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="",
                height=350,
                margin=dict(t=20, b=20, l=10, r=10),
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        except Exception:
            st.info("Feature importance not available for this model.")

    with col4:
        st.subheader("📋 Input Summary")
        summary = pd.DataFrame(
            {
                "Feature": [
                    "Time Spent Alone",
                    "Stage Fear",
                    "Social Event Attendance",
                    "Going Outside",
                    "Drained After Socializing",
                    "Friends Circle Size",
                    "Post Frequency",
                ],
                "Value": [
                    time_alone,
                    stage_fear,
                    social_events,
                    going_outside,
                    drained,
                    friends,
                    post_freq,
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    show_about()
else:
    st.info("👈 Adjust the sliders and dropdowns in the sidebar, then click **Predict**.")
    show_about()
