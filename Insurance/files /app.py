import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰", layout="centered")

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "insurance.csv")


# ----------------------------------------------------------------------------
# Train the model once (mirrors the original insurance.py pipeline) and cache it
# ----------------------------------------------------------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv(csv_path)
    df = data.copy()

    # Encode categoricals the same way the original script does
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df.rename(columns={"sex": "IsFemale", "smoker": "IsSmoker"}, inplace=True)

    df = pd.get_dummies(df, columns=["region"], drop_first=True)
    # ensure region dummy columns are integer (leave numeric columns as floats)
    for col in df.columns:
        if col.startswith("region_"):
            df[col] = df[col].astype(int)

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    df = pd.get_dummies(df, columns=["bmi_category"], drop_first=True)
    # ensure BMI category dummy columns are integer
    for col in df.columns:
        if col.startswith("bmi_category_"):
            df[col] = df[col].astype(int)

    # Fit the scaler on the raw (unscaled) numeric columns, keep the fitted
    # scaler around so we can transform a single new row the same way later
    columns_to_scale = ["age", "bmi", "children"]
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Final feature set used by the original script
    feature_cols = [
        "age", "IsFemale", "bmi", "children", "IsSmoker",
        "region_southeast", "bmi_category_Obese", "region_northwest",
    ]
    # Make sure every expected dummy column exists even if a category
    # happened to be missing from the training data
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    final_df = df[feature_cols + ["charges"]]

    X = final_df.drop("charges", axis=1)
    y = final_df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape[0], X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return model, scaler, feature_cols, r2, adj_r2


model, scaler, feature_cols, r2, adj_r2 = train_model()


# ----------------------------------------------------------------------------
# Helper: turn one raw user input into the exact feature row the model expects
# ----------------------------------------------------------------------------
def build_feature_row(age, sex, bmi, children, smoker, region):
    is_female = 1 if sex == "Female" else 0
    is_smoker = 1 if smoker == "Yes" else 0

    if bmi <= 18.5:
        bmi_category = "Underweight"
    elif bmi <= 24.9:
        bmi_category = "Normal"
    elif bmi <= 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    row = {
        "age": age,
        "IsFemale": is_female,
        "bmi": bmi,
        "children": children,
        "IsSmoker": is_smoker,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "bmi_category_Obese": 1 if bmi_category == "Obese" else 0,
    }

    row_df = pd.DataFrame([row])

    # Scale age / bmi / children with the SAME scaler fitted during training
    row_df[["age", "bmi", "children"]] = scaler.transform(row_df[["age", "bmi", "children"]])

    return row_df[feature_cols], bmi_category


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("💰 Insurance Charge Predictor")
st.write(
    "Estimate annual medical insurance charges from a linear regression model "
    "trained on the classic `insurance.csv` dataset."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex = st.radio("Sex", options=["Male", "Female"], horizontal=True)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.radio("Smoker", options=["Yes", "No"], horizontal=True)
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict charges", use_container_width=True)

if submitted:
    features, bmi_category = build_feature_row(age, sex, bmi, children, smoker, region)
    prediction = model.predict(features)[0]
    prediction = max(prediction, 0)  # charges can't be negative

    st.success(f"### Estimated annual charges: **${prediction:,.2f}**")
    st.caption(f"BMI category detected: **{bmi_category}**")

    with st.expander("Model details"):
        st.write(f"Test R²: **{r2:.4f}**")
        st.write(f"Adjusted R²: **{adj_r2:.4f}**")
        st.dataframe(
            pd.DataFrame(
                {"Feature": feature_cols, "Coefficient": model.coef_}
            ).sort_values("Coefficient", key=abs, ascending=False),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption(
    "Model: Linear Regression on age, sex, BMI, children, smoker status, "
    "region, and BMI category — trained on load and cached."
)
