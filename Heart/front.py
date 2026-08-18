import os
import joblib
import pandas as pd

def load_artifacts(base_dir: str):
    try:
        model = joblib.load(os.path.join(base_dir, "heart_disease_model.pkl"))
        scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
        feature_cols = joblib.load(os.path.join(base_dir, "feature_columns.pkl"))
        return model, scaler, feature_cols
    except Exception as e:
        print("Error loading artifacts:", e)
        return None, None, None


def prepare_data(csv_path: str, feature_cols):
    df = pd.read_csv(csv_path)
    # replace zeros for RestingBP and Cholesterol with means (same as training)
    rest = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
    df["RestingBP"] = df["RestingBP"].replace(0, rest)
    ch = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
    df["Cholesterol"] = df["Cholesterol"].replace(0, ch)

    df_processed = pd.get_dummies(df, drop_first=True)
    # ensure all training feature columns exist in the same order
    X = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        if col in df_processed.columns:
            X[col] = df_processed[col]
        else:
            X[col] = 0
    X = X.astype(float)
    return X


def demo_predict(base_dir: str, idx: int = 0):
    model, scaler, feature_cols = load_artifacts(base_dir)
    if model is None:
        print("Model artifacts not found. Run `main.py` to train and save them first.")
        return

    csv_path = os.path.join(base_dir, "heart.csv")
    if not os.path.exists(csv_path):
        print("heart.csv not found in", base_dir)
        return

    X = prepare_data(csv_path, feature_cols)
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print("Error scaling features:", e)
        return

    pred = model.predict(X_scaled)
    print(f"Prediction for row {idx}:", pred[idx])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        print(f"Probabilities for row {idx}:", proba[idx])


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Running demo prediction using saved model artifacts in:", base_dir)
    demo_predict(base_dir, idx=0)

import joblib
import pandas as pd
import streamlit as st

# Load saved models and artifacts
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Heart Disease Prediction")
st.markdown(
    "Provide the following details to predict the possibility of heart"
    " disease."
)

# User inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["M", "F"])
chestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST", "LVH"])
maxHR = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input(
    "Oldpeak (ST depression induced by exercise relative to rest)",
    0.0,
    6.0,
    1.0,
)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # 1. Initialize input dictionary with numeric fields
    raw_input = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": cholesterol,
        "FastingBS": fastingBS,
        "MaxHR": maxHR,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chestPainType}": 1,
        f"RestingECG_{ecg}": 1,
        f"ExerciseAngina_{exerciseAngina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([raw_input])

    # 2. Fill missing feature columns with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # 3. Ensure exact column ordering matching the trained model
    input_df = input_df[feature_columns]

    # 4. Corrected typo: transform (was transfrom)
    scaler_input = scaler.transform(input_df)

    # 5. Predict outcome
    prediction = model.predict(scaler_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")