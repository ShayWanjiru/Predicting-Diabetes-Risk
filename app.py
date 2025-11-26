# app.py  — Streamlit Cloud friendly with SHAP explanations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import shap

st.set_page_config(
    page_title="Diabetes risk prediction (Pima dataset model)",
    layout="wide"
)

st.title("Diabetes risk prediction (Pima dataset model)")
st.write(
    "Prototype risk prediction model trained on the Pima Indians Diabetes dataset "
    "and re-trained directly inside this app for Streamlit Cloud deployment."
)

# ---------- 1. LOAD DATA ----------

@st.cache_data
def load_data():
    # assumes diabetes.csv is in the same folder as app.py
    df = pd.read_csv("diabetes.csv")
    return df

data = load_data()

feature_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
target_col = "Outcome"

X = data[feature_cols].copy()
y = data[target_col].copy()

# ---------- 2. PREPROCESSING + MODEL ----------

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # zeros treated as missing for selected variables
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    def zero_to_nan(df_in):
        df = df_in.copy()
        for col in zero_as_missing:
            df.loc[df[col] == 0, col] = np.nan
        return df

    # small helper transformer for zero->NaN
    class ZeroToNaNTransformer:
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return zero_to_nan(pd.DataFrame(X, columns=feature_cols)).values

    preprocessor = Pipeline(
        steps=[
            ("zero_to_nan", ZeroToNaNTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    pipe.fit(X_train, y_train)

    # test ROC-AUC
    proba_test = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba_test)

    return pipe, X_train, y_train, X_test, y_test, roc_auc


model, X_train, y_train, X_test, y_test, roc_auc = train_model(X, y)

st.write(f"**Internal test ROC–AUC of trained model:** `{roc_auc:.3f}`")

# ---------- 3. SIDEBAR INPUTS ----------

st.sidebar.header("Patient input")

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
Glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
BloodPressure = st.sidebar.number_input("BloodPressure (mmHg)", min_value=0, max_value=200, value=70, step=1)
SkinThickness = st.sidebar.number_input("SkinThickness (mm)", min_value=0, max_value=100, value=20, step=1)
Insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=0, max_value=1000, value=80, step=1)
BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1, format="%.1f")
DiabetesPedigreeFunction = st.sidebar.number_input(
    "DiabetesPedigreeFunction",
    min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f"
)
Age = st.sidebar.number_input("Age (years)", min_value=10, max_value=120, value=30, step=1)

threshold = st.sidebar.slider(
    "Decision threshold for class 1 (diabetes)",
    min_value=0.1, max_value=0.9, value=0.50, step=0.01,
)
st.sidebar.caption(
    "Lower the threshold (e.g. 0.3) to prioritise recall in screening "
    "contexts where missing high-risk individuals is costly."
)

input_dict = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age,
}
df_in = pd.DataFrame([input_dict])

st.subheader("Input summary")
st.table(df_in.T.rename(columns={0: "value"}))

# ---------- 4. PREDICTION ----------

if st.button("Predict risk"):
    proba = model.predict_proba(df_in)[:, 1][0]
    pred_class = int(proba >= threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Predicted class (0 = no diabetes, 1 = diabetes)",
            value=pred_class,
        )
    with col2:
        st.metric(
            "Predicted probability of diabetes",
            value=f"{proba:.3f}",
        )

    st.caption(
        f"Decision threshold currently set to **{threshold:.2f}**. "
        "Lower thresholds increase sensitivity (recall) but also increase false positives."
    )

# ---------- 5. SHAP EXPLANATIONS ----------

st.markdown("---")
st.subheader("Model explanations (SHAP)")

show_shap = st.checkbox("Show SHAP explanations", value=False)

@st.cache_resource
def get_shap_explainer(trained_model, background_df):
    """
    Build a TreeExplainer on a small background sample to keep it light
    for Streamlit Cloud.
    """
    preproc = trained_model.named_steps["preprocessor"]
    clf = trained_model.named_steps["classifier"]

    # sample up to 200 rows for background
    bg = background_df.sample(
        n=min(200, len(background_df)), random_state=42
    )

    X_bg_proc = preproc.transform(bg)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_bg_proc)
    return explainer, shap_values, bg

if show_shap:
    explainer, shap_vals_bg, bg_df = get_shap_explainer(model, X_train)

    # 5a. Global summary plot
    st.markdown("**Global feature importance (SHAP summary plot)**")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    # for binary RF, shap_values is a list; use index 1 (positive class)
    sv_bg = shap_vals_bg[1] if isinstance(shap_vals_bg, list) else shap_vals_bg
    shap.summary_plot(sv_bg, bg_df, show=False, plot_type="dot")
    st.pyplot(fig1)

    # 5b. Local explanation for current input
    st.markdown("**Local explanation for the current patient**")

    preproc = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    X_input_proc = preproc.transform(df_in)
    shap_vals_input = explainer.shap_values(X_input_proc)
    sv_input = shap_vals_input[1][0] if isinstance(shap_vals_input, list) else shap_vals_input[0]

    contrib = pd.Series(sv_input, index=feature_cols).sort_values(
        key=lambda x: x.abs(), ascending=False
    )

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    contrib.plot(kind="barh", ax=ax2)
    ax2.invert_yaxis()
    ax2.set_xlabel("SHAP value (impact on model output)")
    ax2.set_title("Top feature contributions for this prediction")
    st.pyplot(fig2)

    st.caption(
        "Bars to the right (positive SHAP values) push the prediction towards diabetes; "
        "bars to the left (negative values) push it towards no diabetes."
    )
