"""
save_model.py - Introvert vs Extrovert Model Training
Run: python save_model.py
Output: model.joblib, scaler.joblib, feature_columns.joblib
"""
import joblib
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

BINARY_MAP = {"Yes": 1, "No": 0}
TARGET_MAP = {"Introvert": 0, "Extrovert": 1}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Stage_fear"] = df["Stage_fear"].map(BINARY_MAP)
    df["Drained_after_socializing"] = df["Drained_after_socializing"].map(BINARY_MAP)
    imputer = KNNImputer(n_neighbors=5)
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    return df

def main():
    print("Loading data...")
    df = pd.read_csv("train.csv")
    print(f"Shape: {df.shape}")

    df = preprocess(df)
    df[TARGET] = df[TARGET].map(TARGET_MAP)

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training CatBoost...")
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Introvert", "Extrovert"]))

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(FEATURE_COLS, "feature_columns.joblib")
    print("\n✅ Saved: model.joblib, scaler.joblib, feature_columns.joblib")

if __name__ == "__main__":
    main()