"""
save_model.py - Introvert vs Extrovert Model Training
Run: python save_model.py
Output: model.joblib, scaler.joblib, feature_columns.joblib
"""
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]
TARGET = "Personality"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Encode binary categoricals
    df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
    df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
    # KNN impute
    imputer = KNNImputer(n_neighbors=5)
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    return df

def main():
    print("Loading data...")
    df = pd.read_csv("train.csv")
    df = preprocess(df)
    df[TARGET] = df[TARGET].map({"Introvert": 0, "Extrovert": 1})
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train
    print("Training CatBoost...")
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Introvert", "Extrovert"]))
    # Save
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(FEATURE_COLS, "feature_columns.joblib")
    print("Saved: model.joblib, scaler.joblib, feature_columns.joblib")

if __name__ == "__main__":
    main()
