# 🧠 Introvert vs Extrovert Predictor

A machine-learning web application that predicts whether a person is an **Introvert** or **Extrovert** based on their behavioral traits.  
Built for the **Kaggle Playground Series S5E7** challenge, achieving **~96% accuracy** with CatBoost.

---

## 🎯 About the Project

This project is part of the Kaggle Playground Series Season 5, Episode 7 competition.  
The goal is to classify individuals as Introverts or Extroverts using features such as time spent alone, social event attendance, stage fear, and posting frequency.

**Best model: CatBoostClassifier — ~96% test accuracy**

---

## 🛠 Techniques Used

| Step | Method |
|------|--------|
| Missing value imputation | `KNNImputer(n_neighbors=5)` |
| Feature scaling | `StandardScaler` |
| Classification | `CatBoostClassifier(verbose=0, random_state=42)` |

---

## 📁 Project Structure

```
Predict-Introverts-vs-Extroverts/
├── predict-the-introverts-from-the-extroverts-0-96.ipynb  # EDA & model comparison notebook
├── train.csv                   # Training data (18 524 rows)
├── test.csv                    # Test data
├── save_model.py               # Train & save model artifacts
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and save the model

```bash
python save_model.py
```

This produces three artifact files:
- `model.joblib` — trained CatBoost model
- `scaler.joblib` — fitted StandardScaler
- `feature_columns.joblib` — ordered feature list

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥 App Features

- **Sidebar inputs** — sliders and dropdowns for all 7 features
- **Prediction banner** — clear Introvert / Extrovert result
- **Probability metrics** — confidence scores for both classes
- **Gauge chart** — visual probability indicator (Plotly)
- **Feature importance** — horizontal bar chart (CatBoost native)
- **Input summary table** — review the values you entered

---

## 📊 Dataset

`train.csv` — 18 524 rows, 9 columns:

| Column | Type | Range |
|--------|------|-------|
| `Time_spent_Alone` | numerical | 0–11 |
| `Stage_fear` | categorical | Yes / No |
| `Social_event_attendance` | numerical | 0–10 |
| `Going_outside` | numerical | 0–7 |
| `Drained_after_socializing` | categorical | Yes / No |
| `Friends_circle_size` | numerical | 0–15 |
| `Post_frequency` | numerical | 0–10 |
| `Personality` | **target** | Introvert / Extrovert |

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
