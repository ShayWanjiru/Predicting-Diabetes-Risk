# app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Optional: SHAP (wrapped in try so the app still runs if shap fails to import)
try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes risk prediction (Pima dataset model)",
    layout="wide",
)

st.title("Diabetes risk prediction (Pima dataset model)")
st.write(
    "Prototype risk prediction model trained on the Pima Indians Diabetes "
    "dataset (`diabetes.csv`) and re-trained directly inside this app for "
    "Streamlit Cloud deployment."
)

DATA_PATH = "diabetes.csv"  # file should be in the repo root


# ---------------------------------------------------------------------
# 1. Load data and train model (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_data_and_train():
    if not os.path.exists(DATA_PATH):
        st.error(f"Could not find `{DATA_PATH}` in the app directory.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Expect Pima layout
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

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessor (all numeric -> standardize)
    numeric_features = feature_cols
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    # Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    pipe.fit(X_train, y_train)

    # Evaluate on test set at default threshold 0.5
    y_proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "roc_auc": roc_auc_score(y_test, y_proba_test),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    return pipe, X_train, X_test, y_train, y_test, metrics, (fpr, tpr)


model, X_train, X_test, y_train, y_test, metrics, roc_points = load_data_and_train()


# ---------------------------------------------------------------------
# 2. Sidebar: patient inputs and threshold
# ---------------------------------------------------------------------
st.sidebar.header("Patient input")

def sidebar_inputs():
    Pregnancies = st.sidebar.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1
    )
    Glucose = st.sidebar.number_input(
        "Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1
    )
    BloodPressure = st.sidebar.number_input(
        "BloodPressure (mmHg)", min_value=0, max_value=200, value=70, step=1
    )
    SkinThickness = st.sidebar.number_input(
        "SkinThickness (mm)", min_value=0, max_value=100, value=20, step=1
    )
    Insulin = st.sidebar.number_input(
        "Insulin (µU/mL)", min_value=0, max_value=1000, value=80, step=1
    )
    BMI = st.sidebar.number_input(
        "BMI (kg/m²)",
        min_value=10.0,
        max_value=70.0,
        value=30.0,
        step=0.1,
        format="%.1f",
    )
    DiabetesPedigreeFunction = st.sidebar.number_input(
        "DiabetesPedigreeFunction",
        min_value=0.0,
        max_value=5.0,
        value=0.50,
        step=0.01,
        format="%.2f",
    )
    Age = st.sidebar.number_input(
        "Age (years)", min_value=10, max_value=120, value=30, step=1
    )

    thresh = st.sidebar.slider(
        "Decision threshold for class 1 (diabetes)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )
    st.sidebar.caption(
        "Lower the threshold (e.g., 0.3) to prioritize recall in screening "
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

    return pd.DataFrame([input_dict]), thresh


df_in, decision_threshold = sidebar_inputs()

# ---------------------------------------------------------------------
# 3. Main panel – input summary and internal metrics
# ---------------------------------------------------------------------
st.subheader("Input summary")
st.table(df_in.T.rename(columns={0: "value"}))

st.markdown("**Internal test ROC–AUC of trained model:** "
            f"`{metrics['roc_auc']:.3f}`")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col_m2.metric("Precision", f"{metrics['precision']:.3f}")
col_m3.metric("Recall", f"{metrics['recall']:.3f}")
col_m4.metric("F1-score", f"{metrics['f1']:.3f}")

# ROC curve
with st.expander("Show ROC curve"):
    fpr, tpr = roc_points
    fig_roc = plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"RandomForest (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig_roc)


# ---------------------------------------------------------------------
# 4. Single-patient prediction
# ---------------------------------------------------------------------
if st.button("Predict risk"):
    try:
        proba = model.predict_proba(df_in)[:, 1]
        pred_class = (proba >= decision_threshold).astype(int)[0]
        risk = float(proba[0])
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.stop()

    st.success("Prediction complete.")
    st.metric("Predicted class (0 = no diabetes, 1 = diabetes)", pred_class)
    st.metric("Predicted probability of diabetes", f"{risk:.3f}")
    st.caption(
        f"Current decision threshold = {decision_threshold:.2f}. "
        "In screening scenarios, recall is often prioritized, so slightly "
        "lower thresholds may be appropriate."
    )


# ---------------------------------------------------------------------
# 5. SHAP explanations (no Streamlit caching here)
# ---------------------------------------------------------------------
def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Build a SHAP TreeExplainer on a small background sample.
    No Streamlit caching to avoid UnhashableParamError.
    """
    clf = model
    preproc = None

    if hasattr(model, "named_steps"):
        preproc = model.named_steps.get("preprocessor")
        clf = model.named_steps.get("classifier") or clf

    # Apply preprocessing to background data if needed
    if preproc is not None:
        X_bg_proc = preproc.transform(X_background)
    else:
        X_bg_proc = X_background.values

    explainer = shap.TreeExplainer(clf)
    shap_vals_bg = explainer.shap_values(X_bg_proc)

    feature_names = list(X_background.columns)
    return explainer, shap_vals_bg, X_bg_proc, feature_names


st.subheader("Model explanations (SHAP)")
show_shap = st.checkbox("Show SHAP explanations", value=False)

if show_shap:
    if not HAS_SHAP:
        st.warning(
            "SHAP is not installed or could not be imported in this environment."
        )
    else:
        with st.spinner("Computing SHAP explanations…"):
            # Background sample for global SHAP
            bg_sample = X_train.sample(
                n=min(200, len(X_train)), random_state=42
            )

            explainer, shap_vals_bg, X_bg_proc, feat_names = get_shap_explainer(
                model, bg_sample
            )

            # Local explanation: current patient
            if hasattr(model, "named_steps"):
                preproc = model.named_steps.get("preprocessor")
                if preproc is not None:
                    x_current_proc = preproc.transform(df_in)
                else:
                    x_current_proc = df_in.values
            else:
                x_current_proc = df_in.values

            shap_vals_current = explainer.shap_values(x_current_proc)

        # Global summary
        st.markdown("**Global feature importance (SHAP summary plot)**")
        fig_shap_global = plt.figure(figsize=(8, 4))
        sv_bg = shap_vals_bg[1] if isinstance(shap_vals_bg, list) else shap_vals_bg
        shap.summary_plot(
            sv_bg,
            X_bg_proc,
            feature_names=feat_names,
            show=False,
        )
        st.pyplot(fig_shap_global)

        # Local explanation as bar chart
        st.markdown("**Local explanation for this patient (top contributions)**")
        fig_shap_local = plt.figure(figsize=(6, 4))
        sv_local = (
            shap_vals_current[1][0]
            if isinstance(shap_vals_current, list)
            else shap_vals_current[0]
        )
        contrib = pd.Series(sv_local, index=feat_names).sort_values(
            key=lambda x: x.abs(), ascending=True
        )
        contrib.tail(8).plot(kind="barh")
        plt.xlabel("SHAP value (impact on model output)")
        plt.tight_layout()
        st.pyplot(fig_shap_local)


# ---------------------------------------------------------------------
# 6. Batch CSV scoring
# ---------------------------------------------------------------------
st.sidebar.header("Batch / CSV scoring")

uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: "
    "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, "
    "BMI, DiabetesPedigreeFunction, Age",
    type=["csv"],
)

if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    st.subheader("Batch prediction preview")
    st.dataframe(batch_df.head())

    if st.sidebar.button("Run batch predict"):
        try:
            batch_proba = model.predict_proba(batch_df)[:, 1]
            batch_df["pred_prob"] = batch_proba
            batch_df["pred_class"] = (
                batch_df["pred_prob"] >= decision_threshold
            ).astype(int)
            st.write(batch_df.head())

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                csv_bytes,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
