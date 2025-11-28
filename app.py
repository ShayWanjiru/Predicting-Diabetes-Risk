# app.py
# Diabetes risk prediction – Streamlit Cloud
# ---------------------------------------------------
# Trains the preprocessing + RandomForest pipeline directly from diabetes.csv
# and uses SHAP for global + local explanations, plus EDA & model performance.

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
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)

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

# Threshold used in your report for recall-oriented screening
TUNED_THRESHOLD = 0.287


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
    """
    Train the RF pipeline on diabetes.csv and prepare SHAP artifacts.
    Returns:
        pipeline, test_auc, shap_explainer, X_bg_proc, X_bg_raw, shap_bg_vals
    """
    df = load_data()

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from diabetes.csv: {missing_cols}")
        st.stop()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Treat biologically impossible zeros as missing BEFORE building the pipeline
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

            # Background subset for SHAP (max 200 samples)
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
# Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Model Performance", "EDA Snapshot"]
)


# -------------------------------------------------
# PAGE 1: Single Prediction
# -------------------------------------------------
if page == "Single Prediction":
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
        st.markdown(f"**Internal test ROC–AUC of trained Random Forest model:** `{test_auc:.3f}`")

    # ---------------- Prediction ----------------
    if st.button("Predict risk"):
        # Make prediction with consistent zero→NaN handling
        df_in_proc = df_in.copy()
        df_in_proc[ZERO_INFLATED] = df_in_proc[ZERO_INFLATED].replace(0, np.nan)

        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(df_in_proc)[0, 1])
            else:
                raw = model.decision_function(df_in_proc)
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

        # ---------------- SHAP explanations ----------------
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
                    fig_global, ax_global = plt.subplots(figsize=(7, 4))
                    shap.summary_plot(
                        shap_bg_vals,
                        X_bg_raw[FEATURES],
                        show=False,
                        plot_type="bar"
                    )
                    st.pyplot(fig_global)
                    plt.close(fig_global)
                except Exception as e:
                    st.warning(f"Global SHAP summary failed: {e}")

                # ---------- Local explanation ----------
                st.subheader("Local explanation for this patient (top contributions)")

                try:
                    preproc = model.named_steps["preprocessor"]
                    X_proc_single = preproc.transform(df_in_proc)

                    shap_raw_single = shap_explainer.shap_values(X_proc_single)

                    # Robust handling of SHAP values
                    if isinstance(shap_raw_single, list):
                        if len(shap_raw_single) == 2:
                            sv_local = shap_raw_single[1][0]  # First sample, class 1
                        else:
                            sv_local = shap_raw_single[0][0]
                    else:
                        shap_vals_single = normalize_shap_values(shap_raw_single)
                        if shap_vals_single.ndim == 2:
                            sv_local = shap_vals_single[0]
                        else:
                            sv_local = shap_vals_single

                    # Ensure we have the right number of features
                    if len(sv_local) != len(FEATURES):
                        # Truncate or pad if needed
                        sv_local = sv_local[: len(FEATURES)]

                    feat_names = FEATURES

                    contrib = pd.Series(sv_local, index=feat_names).sort_values(
                        key=lambda x: x.abs(), ascending=False
                    )

                    fig_local, ax_local = plt.subplots(figsize=(8, 5))
                    bars = ax_local.barh(range(len(contrib)), contrib.values)

                    # Color bars based on positive/negative impact
                    for i, bar in enumerate(bars):
                        if contrib.values[i] > 0:
                            bar.set_color("red")   # Increases risk
                        else:
                            bar.set_color("blue")  # Decreases risk

                    ax_local.set_yticks(range(len(contrib)))
                    ax_local.set_yticklabels(contrib.index)
                    ax_local.invert_yaxis()
                    ax_local.set_xlabel("SHAP value (impact on model output)")
                    ax_local.set_title("Feature contributions to prediction")
                    ax_local.axvline(x=0, color="black", linestyle="-", alpha=0.3)

                    st.pyplot(fig_local)
                    plt.close(fig_local)

                    # Detailed table
                    st.subheader("Detailed feature contributions")
                    contrib_df = pd.DataFrame(
                        {
                            "Feature": contrib.index,
                            "SHAP Value": [f"{x:.4f}" for x in contrib.values],
                            "Impact": [
                                "Increases risk" if x > 0 else "Decreases risk"
                                for x in contrib.values
                            ],
                            "Magnitude": [f"{abs(x):.4f}" for x in contrib.values],
                        }
                    ).reset_index(drop=True)
                    st.table(contrib_df)

                except Exception as e:
                    st.error(f"Local SHAP explanation failed: {str(e)}")


# -------------------------------------------------
# PAGE 2: Model Performance (test set)
# -------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model performance on held-out test set")

    df = load_data()
    X_all = df[FEATURES].copy()
    y_all = df[TARGET].astype(int)

    # Apply the same zero→NaN rule used in training
    X_all[ZERO_INFLATED] = X_all[ZERO_INFLATED].replace(0, np.nan)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Default threshold 0.50
    y_proba = model.predict_proba(X_test2)[:, 1]
    y_pred_050 = (y_proba >= 0.50).astype(int)

    acc_050 = accuracy_score(y_test2, y_pred_050)
    prec_050 = precision_score(y_test2, y_pred_050)
    rec_050 = recall_score(y_test2, y_pred_050)
    f1_050 = f1_score(y_test2, y_pred_050)
    auc_050 = roc_auc_score(y_test2, y_proba)

    # Tuned threshold (e.g. 0.287 from your analysis)
    y_pred_tuned = (y_proba >= TUNED_THRESHOLD).astype(int)
    acc_t = accuracy_score(y_test2, y_pred_tuned)
    prec_t = precision_score(y_test2, y_pred_tuned)
    rec_t = recall_score(y_test2, y_pred_tuned)
    f1_t = f1_score(y_test2, y_pred_tuned)

    # Metrics table
    metrics_df = pd.DataFrame(
        {
            "Threshold": ["0.50 (default)", f"{TUNED_THRESHOLD:.3f} (tuned)"],
            "Accuracy": [acc_050, acc_t],
            "Precision": [prec_050, prec_t],
            "Recall": [rec_050, rec_t],
            "F1-score": [f1_050, f1_t],
            "ROC-AUC": [auc_050, auc_050],  # AUC unaffected by threshold
        }
    )

    st.markdown("**Test-set metrics (Random Forest)**")
    st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

    # Confusion matrix for default threshold
    st.markdown("**Confusion matrix (threshold = 0.50)**")
    cm_050 = confusion_matrix(y_test2, y_pred_050)

    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    im = ax_cm.imshow(cm_050, cmap="Blues")

    for i in range(cm_050.shape[0]):
        for j in range(cm_050.shape[1]):
            ax_cm.text(j, i, cm_050[i, j], ha="center", va="center", color="black")

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["No diabetes", "Diabetes"])
    ax_cm.set_yticklabels(["No diabetes", "Diabetes"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Confusion matrix for tuned threshold
    st.markdown(f"**Confusion matrix (threshold = {TUNED_THRESHOLD:.3f})**")
    cm_t = confusion_matrix(y_test2, y_pred_tuned)

    fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
    im2 = ax_cm2.imshow(cm_t, cmap="Blues")

    for i in range(cm_t.shape[0]):
        for j in range(cm_t.shape[1]):
            ax_cm2.text(j, i, cm_t[i, j], ha="center", va="center", color="black")

    ax_cm2.set_xticks([0, 1])
    ax_cm2.set_yticks([0, 1])
    ax_cm2.set_xticklabels(["No diabetes", "Diabetes"])
    ax_cm2.set_yticklabels(["No diabetes", "Diabetes"])
    ax_cm2.set_xlabel("Predicted")
    ax_cm2.set_ylabel("Actual")
    st.pyplot(fig_cm2)
    plt.close(fig_cm2)

    # ROC curve
    st.markdown("**ROC curve (Random Forest)**")
    fpr, tpr, _ = roc_curve(y_test2, y_proba)

    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_050:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    st.caption(
        "Threshold = 0.50 balances precision and recall; the tuned threshold "
        f"{TUNED_THRESHOLD:.3f} substantially increases recall, making the model more "
        "suitable for high-sensitivity screening scenarios."
    )


# -------------------------------------------------
# PAGE 3: EDA Snapshot
# -------------------------------------------------
elif page == "EDA Snapshot":
    st.subheader("Exploratory data analysis snapshot")

    df = load_data()

    # 1. Target distribution
    st.markdown("**Class balance (Outcome)**")
    counts = df[TARGET].value_counts().sort_index()
    counts.index = ["No diabetes (0)", "Diabetes (1)"]
    st.bar_chart(counts)

    st.caption(
        "The target variable is moderately imbalanced (~65% no diabetes, ~35% diabetes). "
        "This motivated using metrics like Recall, F1-score, and ROC-AUC instead of "
        "accuracy alone."
    )

    # 2. Key feature histograms: Glucose and BMI
    st.markdown("**Key feature distributions**")
    fig_hist, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].hist(df["Glucose"], bins=20)
    axes[0].set_title("Glucose")
    axes[0].set_xlabel("Glucose (mg/dL)")

    axes[1].hist(df["BMI"], bins=20)
    axes[1].set_title("BMI")
    axes[1].set_xlabel("BMI (kg/m²)")

    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.caption(
        "Glucose and BMI show clear variation between individuals and are known clinical "
        "risk factors, which is reflected later in the model's feature importance."
    )

    # 3. Simple correlation view for main predictors
    st.markdown("**Correlation with Outcome (top predictors)**")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)
    corr_df = corr.to_frame(name="Correlation_with_Outcome")
    st.table(corr_df)

    st.caption(
        "Glucose, BMI, Age, and Pregnancies have the strongest positive correlations "
        "with diabetes Outcome. These EDA findings guided the decision to use tree-based "
        "models that can capture non-linear relationships and interactions among these "
        "features."
    )
