import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

# Feature columns (order matters)
FEATURE_COLS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

# 1. Read data
if not Path("train.csv").exists():
    raise FileNotFoundError(
        "train.csv not found. Please ensure the training data is in the current directory."
    )
df = pd.read_csv("train.csv")

# 2. Encode categorical features
df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0}).astype("Int64")
df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0}).astype("Int64")

# 3. Fill missing values with KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])

# 4. Encode target
df["Personality"] = df["Personality"].map({"Introvert": 0, "Extrovert": 1})

X = df[FEATURE_COLS].astype(float)
y = df["Personality"]

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

# 6. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# 7. Train CatBoost
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Introvert", "Extrovert"]))

# 9. Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(FEATURE_COLS, "feature_columns.joblib")
print("Saved: model.joblib, scaler.joblib, feature_columns.joblib")
