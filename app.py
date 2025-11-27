# app.py
# Diabetes risk prediction – Streamlit Cloud friendly
# ---------------------------------------------------
# Trains the preprocessing + RandomForest pipeline directly from diabetes.csv
# and uses SHAP for global + local explanations.

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

# ---- Optional: SHAP ----
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# -------------------------------------------------
# Streamlit configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes risk prediction (Pima dataset model)",
    layout="centered"
)

st.title("Diabetes risk prediction (Pima dataset model)")
st.caption(
    "Prototype risk prediction model trained on the Pima Indians Diabetes dataset "
    "directly inside this Streamlit app."
)


# -------------------------------------------------
# Constants
# -------------------------------------------------
DATA_PATH = "diabetes.csv"  # must be in the repo root

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


# -------------------------------------------------
# Helper for robust SHAP handling
# -------------------------------------------------
def normalize_shap_values(shap_output):
    """
    Convert whatever SHAP returns (list / array / Explanation / 3D)
    into a 2D numpy array of shape (n_samples, n_features) for the
    positive class.
    """
    # New API: Explanation object
    if HAS_SHAP and hasattr(shap_output, "values"):
        values = shap_output.values
    else:
        values = shap_output

    # List case (e.g., [array(class0), array(class1)])
    if isinstance(values, list):
        # use positive class (1) if available, otherwise first
        values = np.array(values[1 if len(values) > 1 else 0])

    values = np.array(values)

    # 3D case: (n_samples, n_features, n_outputs)
    if values.ndim == 3:
        out_idx = 1 if values.shape[2] > 1 else 0
        values = values[:, :, out_idx]

    # 1D case: (n_features,) -> make (1, n_features)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values  # guaranteed 2D


# -------------------------------------------------
# Data + model training
# -------------------------------------------------
@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data file not found at: {path}")
        st.stop()
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_pipeline_and_shap():
    df = load_data()

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from diabetes.csv: {missing_cols}")
        st.stop()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Treat biologically impossible zeros as missing
    X[ZERO_INFLATED] = X[ZERO_INFLATED].replace(0, np.nan)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURES)]
    )

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    # ROC-AUC on test set
    if hasattr(pipeline, "predict_proba"):
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba_test)
    else:
        test_auc = None

    shap_explainer = None
    X_bg_proc = None
    X_bg_raw = None
    shap_bg_vals = None

    if HAS_SHAP:
        try:
            preproc = pipeline.named_steps["preprocessor"]
            clf_rf = pipeline.named_steps["classifier"]

            # background subset for SHAP (max 200 samples)
            X_bg_raw = X_train.iloc[:200].copy()
            X_bg_proc = preproc.transform(X_bg_raw)

            shap_explainer = shap.TreeExplainer(clf_rf)
            shap_raw = shap_explainer.shap_values(X_bg_proc)
            shap_bg_vals = normalize_shap_values(shap_raw)
        except Exception:
            shap_explainer = None
            X_bg_proc = None
            X_bg_raw = None
            shap_bg_vals = None

    return pipeline, test_auc, shap_explainer, X_bg_proc, X_bg_raw, shap_bg_vals


# Train model + SHAP stuff once (cached)
model, test_auc, shap_explainer, X_bg_proc, X_bg_raw, shap_bg_vals = train_pipeline_and_shap()


# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------
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
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)
st.sidebar.caption(
    "Lower the threshold (e.g., 0.30) to prioritize **recall** in screening contexts."
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
    st.markdown(f"**Internal test ROC–AUC of trained model:** `{test_auc:.3f}`")


# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict risk"):
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(df_in)[0, 1])
        else:
            raw = model.decision_function(df_in)
            prob = float((raw - raw.min()) / (raw.max() - raw.min()))
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        st.stop()

    pred_class = int(prob >= threshold)

    st.metric("Predicted class (0 = no diabetes, 1 = diabetes)", pred_class)
    st.metric("Predicted probability of diabetes", f"{prob:.3f}")
    st.caption(
        f"Prediction computed using a RandomForestClassifier with decision threshold `{threshold:.2f}`."
    )

    # -------------------------------------------------
    # SHAP explanations (global + local)
    # -------------------------------------------------
    show_shap = st.checkbox("Show SHAP explanations", value=False)

    if show_shap:
        if not HAS_SHAP:
            st.warning(
                "SHAP is not available in this environment. "
                "Ensure `shap` is listed in `requirements.txt`."
            )
        elif shap_explainer is None or X_bg_proc is None or X_bg_raw is None or shap_bg_vals is None:
            st.warning(
                "SHAP explainer could not be initialized for this model. "
                "This may happen if the preprocessing step is incompatible."
            )
        else:
            # ---------- Global summary ----------
            st.subheader("Global feature importance (SHAP summary plot)")

            try:
                # Let SHAP create its own figure, then grab it
                shap.summary_plot(
                    shap_bg_vals,
                    X_bg_raw[FEATURES],
                    show=False,
                    plot_type="bar"
                )
                fig_global = plt.gcf()
                st.pyplot(fig_global)
                plt.clf()
            except Exception as e:
                st.warning(f"Global SHAP summary failed: {e}")

            # ---------- Local explanation ----------
            st.subheader("Local explanation for this patient (top contributions)")

            try:
                preproc = model.named_steps["preprocessor"]
                X_proc_single = preproc.transform(df_in)

                shap_raw_single = shap_explainer.shap_values(X_proc_single)
                shap_vals_single = normalize_shap_values(shap_raw_single)  # (1, n_features)
                sv_local = shap_vals_single[0]  # 1D vector

                feat_names = FEATURES
                contrib = pd.Series(sv_local, index=feat_names).sort_values(
                    key=lambda x: x.abs(), ascending=False
                )

                fig_local, ax_local = plt.subplots(figsize=(6, 4))
                contrib.plot(kind="barh", ax=ax_local)
                ax_local.invert_yaxis()
                ax_local.set_xlabel("SHAP value (impact on model output)")
                st.pyplot(fig_local)
                plt.close(fig_local)
            except Exception as e:
                st.warning(f"Local SHAP explanation failed: {e}")
