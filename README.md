# 🧠 Introvert vs Extrovert Predictor

A Streamlit web application that predicts whether a person is an **Introvert** or **Extrovert** based on behavioral traits, built on the [Kaggle Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7) dataset.

**Model Accuracy: ~96%**

---

## 📁 Project Structure

```
├── train.csv                                      # Training dataset
├── test.csv                                       # Test dataset
├── predict-the-introverts-from-the-extroverts-0-96.ipynb  # Exploration notebook
├── save_model.py                                  # Model training & artifact saving
├── app.py                                         # Streamlit application
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/tugcesi/Predict-Introverts-vs-Extroverts.git
cd Predict-Introverts-vs-Extroverts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python save_model.py
```

This will generate three artifact files:
- `model.joblib` — trained CatBoost classifier
- `scaler.joblib` — fitted StandardScaler
- `feature_columns.joblib` — ordered list of feature column names

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Dataset Columns

| Column | Type | Description |
|---|---|---|
| `Time_spent_Alone` | Numeric (0–11) | Hours per day spent alone |
| `Stage_fear` | Yes / No | Whether the person has stage fear |
| `Social_event_attendance` | Numeric (0–10) | Social events attended per month |
| `Going_outside` | Numeric (0–7) | Days per week going outside |
| `Drained_after_socializing` | Yes / No | Feels drained after social interaction |
| `Friends_circle_size` | Numeric (0–15) | Number of close friends |
| `Post_frequency` | Numeric (0–10) | Social media posts per week |
| `Personality` | Introvert / Extrovert | **Target variable** |

---

## 🔬 Techniques Used

- **KNN Imputation** (`KNNImputer(n_neighbors=5)`) — handles missing values in all feature columns
- **Standard Scaling** (`StandardScaler`) — normalizes features before model training
- **CatBoost Classifier** (`CatBoostClassifier(verbose=0, random_state=42)`) — gradient boosting model achieving ~96% accuracy

---

## 🖥️ App Features

- Sidebar sliders and dropdowns for all 7 input features
- **Prediction result** displayed as a green (Extrovert) or blue (Introvert) banner
- **Plotly gauge chart** showing extrovert probability score
- **Feature importance** horizontal bar chart
- **Input summary** table
- About expander with model details
