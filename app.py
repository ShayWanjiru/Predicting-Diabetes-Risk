# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

# SHAP is optional but highly recommended
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ----------------- Streamlit page config ----------------- #

st.set_page_config(
    page_title="Diabetes risk prediction (Pima dataset model)",
    layout="centered"
)

st.title("Diabetes risk prediction (Pima dataset model)")
st.caption(
    "Prototype risk prediction model trained on the Pima Indians Diabetes dataset "
    "and re-trained directly inside this app for Streamlit Cloud deployment."
)


# ----------------- Data & model training ----------------- #

DATA_PATH = "diabetes.csv"  # make sure this file is in the repo root
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

ZERO_INFLATED = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data file not found at: {path}")
        st.stop()
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_pipeline_and_shap():
    """Train preprocessing + RandomForest pipeline and (optionally) SHAP explainer."""
    df = load_data()

    # Basic sanity: keep only expected columns
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from diabetes.csv: {missing_cols}")
        st.stop()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Replace biologically implausible zeros with NaN for selected features
    X[ZERO_INFLATED] = X[ZERO_INFLATED].replace(0, np.nan)

    # Define numeric preprocessor (median imputation + standardization)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURES)
        ]
    )

    # RandomForest classifier (you can tweak hyperparameters to match your notebook)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # Train/test split for quick internal evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # Compute ROC-AUC on test set to show in the UI
    if hasattr(pipeline, "predict_proba"):
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba_test)
    else:
        test_auc = None

    # Prepare SHAP explainer on processed training data (optional)
    explainer = None
    shap_background = None

    if HAS_SHAP:
        try:
            preproc = pipeline.named_steps["preprocessor"]
            clf_rf = pipeline.named_steps["classifier"]
            X_train_proc = preproc.transform(X_train)

            explainer = shap.TreeExplainer(clf_rf)
            shap_background = (X_train_proc, X_train)  # processed + raw for later use
        except Exception:
            explainer = None
            shap_background = None

    return pipeline, test_auc, explainer, shap_background


# Train once and cache
model, test_auc, shap_explainer, shap_background = train_pipeline_and_shap()


# ----------------- Sidebar: user inputs ----------------- #

st.sidebar.header("Patient input")

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
Glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
BloodPressure = st.sidebar.number_input("BloodPressure (mmHg)", min_value=0, max_value=200, value=70, step=1)
SkinThickness = st.sidebar.number_input("SkinThickness (mm)", min_value=0, max_value=100, value=20, step=1)
Insulin = st.sidebar.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=80, step=1)
BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1, format="%.1f")
DiabetesPedigreeFunction = st.sidebar.number_input(
    "DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f"
)
Age = st.sidebar.number_input("Age (years)", min_value=10, max_value=120, value=30, step=1)

threshold = st.sidebar.slider(
    "Decision threshold for class 1 (diabetes)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)
st.sidebar.caption(
    "Lower the threshold (e.g., 0.3) to prioritize **recall** in screening contexts "
    "where missing high-risk individuals is costly."
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

if test_auc is not None:
    st.markdown(f"**Internal test ROC-AUC of trained model:** `{test_auc:.3f}`")


# ----------------- Prediction ----------------- #

if st.button("Predict risk"):
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_in)[0, 1]
        else:
            # Fallback if no predict_proba
            raw = model.decision_function(df_in)
            prob = float((raw - raw.min()) / (raw.max() - raw.min()))
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.stop()

    pred_class = int(prob >= threshold)

    st.metric("Predicted class (0 = no diabetes, 1 = diabetes)", pred_class)
    st.metric("Predicted probability of diabetes", f"{prob:.3f}")

    st.caption(
        f"Prediction computed using a RandomForestClassifier with a decision threshold of {threshold:.2f}."
    )

    # ------------- Optional SHAP explanation ------------- #

    show_shap = st.checkbox("Show SHAP explanation (tree-based model only)", value=False)

    if show_shap:
        if not HAS_SHAP:
            st.warning(
                "SHAP is not available in this environment. "
                "Ensure `shap` is listed in `requirements.txt`."
            )
        elif shap_explainer is None or shap_background is None:
            st.warning(
                "SHAP explainer could not be initialized for this model. "
                "This may happen if the preprocessing step is incompatible."
            )
        else:
            try:
                preproc = model.named_steps["preprocessor"]
                X_proc_single = preproc.transform(df_in)

                shap_vals_single = shap_explainer.shap_values(X_proc_single)
                # For binary classification with TreeExplainer, shap_values is a list [class0, class1]
                if isinstance(shap_vals_single, list):
                    shap_vals_single = shap_vals_single[1]

                sv = shap_vals_single[0]  # 1D array of contributions
                contrib = pd.Series(sv, index=FEATURES).sort_values(key=lambda x: x.abs(), ascending=False)

                st.subheader("SHAP feature contribution (this patient)")
                fig, ax = plt.subplots(figsize=(6, 4))
                contrib.plot(kind="barh", ax=ax)
                ax.invert_yaxis()
                ax.set_xlabel("SHAP value (impact on model output)")
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
