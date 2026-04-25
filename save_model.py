import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import joblib

FEATURE_COLS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

df = pd.read_csv("train.csv")

df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})

imputer = KNNImputer(n_neighbors=5)
df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])

df["Personality"] = df["Personality"].map({"Introvert": 0, "Extrovert": 1})

X = df[FEATURE_COLS]
y = df["Personality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Introvert", "Extrovert"]))

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(FEATURE_COLS, "feature_columns.joblib")

print("Saved: model.joblib, scaler.joblib, feature_columns.joblib")
