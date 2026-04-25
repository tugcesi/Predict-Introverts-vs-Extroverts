# 🧠 Introvert vs Extrovert Predictor

A machine learning web application that predicts whether a person is an **Introvert** or **Extrovert** based on their behavioral traits. Built for the **Kaggle Playground Series Season 5 Episode 7** challenge.

---

## 🎯 Overview

This project trains a **CatBoost** classifier on personality survey data and serves predictions through an interactive **Streamlit** web application. The model achieves approximately **96% accuracy** on the validation set.

---

## ✨ Features

- 🔍 **Real-time prediction** — adjustable sliders and dropdowns for instant results
- 📊 **Plotly gauge chart** — visual extrovert probability meter (0–100%)
- 📈 **Feature importance** — horizontal bar chart showing which features drive predictions
- 📋 **Input summary table** — review all entered values at a glance
- 🛡️ **Robust preprocessing** — KNN imputation and StandardScaler applied consistently between training and inference

---

## 🛠️ Techniques

| Technique | Purpose |
|---|---|
| **KNN Imputation** | Fill missing values using 5 nearest neighbours |
| **StandardScaler** | Normalise features before training |
| **CatBoost Classifier** | Gradient-boosted decision trees for high accuracy |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model

```bash
python save_model.py
```

This produces three artefact files:
- `model.joblib` — trained CatBoost model
- `scaler.joblib` — fitted StandardScaler
- `feature_columns.joblib` — ordered feature column list

### 2. Launch the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📁 Project Structure

| File | Description |
|---|---|
| `save_model.py` | Trains the model and saves artefacts |
| `app.py` | Streamlit prediction app |
| `requirements.txt` | Python dependencies |
| `train.csv` | Training dataset |
| `test.csv` | Test dataset |
| `predict-the-introverts-from-the-extroverts-0-96.ipynb` | Exploratory notebook |

---

## 📊 Dataset Features

| Feature | Type | Range | Description |
|---|---|---|---|
| `Time_spent_Alone` | Numeric | 0–11 | Hours spent alone per day |
| `Stage_fear` | Categorical | Yes/No | Presence of stage fright |
| `Social_event_attendance` | Numeric | 0–10 | Social events attended per month |
| `Going_outside` | Numeric | 0–7 | Times going outside per week |
| `Drained_after_socializing` | Categorical | Yes/No | Feeling drained after social interaction |
| `Friends_circle_size` | Numeric | 0–15 | Number of close friends |
| `Post_frequency` | Numeric | 0–10 | Social media posts per week |
| `Personality` | Target | Introvert/Extrovert | Personality classification |

---

## 📈 Model Performance

- **Accuracy:** ~96%
- **Challenge:** Kaggle Playground Series S5E7 — *Predict the Introverts from the Extroverts*

---

## 📝 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
