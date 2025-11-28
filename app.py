# app.py
# Diabetes risk prediction – Streamlit Cloud
# -----------------------------------------
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Optional: SHAP (guarded import)
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
    layout="wide",
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

# Biologically impossible zeros that indicate missingness
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
    if HAS_SHAP and hasattr(shap_output, "values"):
        values = shap_output.values
    else:
        values = shap_output

    if isinstance(values, list):
        # list for classes -> use class 1 if available
        values = np.array(values[1 if len(values) > 1 else 0])

    values = np.array(values)

    # 3D case: (n_samples, n_features, n_outputs)
    if values.ndim == 3:
        out_idx = 1 if values.shape[2] > 1 else 0
        values = values[:, :, out_idx]

    # 1D case: (n_features,) -> (1, n_features)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    return values


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

    # Test predictions
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    auc = roc_auc_score(y_test, y_proba_test)

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "Value": [acc, prec, rec, f1, auc],
        }
    )

    # SHAP background
    shap_explainer = None
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
            X_bg_raw = None
            shap_bg_vals = None

    return (
        pipeline,
        metrics_df,
        shap_explainer,
        X_bg_raw,
        shap_bg_vals,
        X_test,
        y_test,
        y_pred_test,
        y_proba_test,
    )


# Train model + SHAP stuff once (cached)
(
    model,
    metrics_df,
    shap_explainer,
    X_bg_raw,
    shap_bg_vals,
    X_test,
    y_test,
    y_pred_test,
    y_proba_test,
) = train_pipeline_and_shap()


# -------------------------------------------------
# Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Model Performance", "EDA Snapshot"],
)


# -------------------------------------------------
# PAGE 1 – SINGLE PREDICTION
# -------------------------------------------------
if page == "Single Prediction":
    st.subheader("Single patient prediction")

    st.sidebar.header("Patient input")

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
        "Insulin (µU/mL)", min_value=0, max_value=900, value=80, step=1
    )
    BMI = st.sidebar.number_input(
        "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1, format="%.1f"
    )
    DiabetesPedigreeFunction = st.sidebar.number_input(
        "DiabetesPedigreeFunction",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.01,
        format="%.2f",
    )
    Age = st.sidebar.number_input(
        "Age (years)", min_value=10, max_value=120, value=30, step=1
    )

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

    st.markdown("#### Input summary")
    st.table(df_in.T.rename(columns={0: "value"}))

    if st.button("Predict risk"):
        # Treat zeros as missing for zero-inflated features
        df_in_model = df_in.copy()
        df_in_model[ZERO_INFLATED] = df_in_model[ZERO_INFLATED].replace(0, np.nan)

        try:
            prob = float(model.predict_proba(df_in_model)[0, 1])
        except Exception as e:
            st.error(f"Model inference failed: {e}")
            st.stop()

        pred_class = int(prob >= threshold)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted class (0 = no diabetes, 1 = diabetes)", pred_class)
        with col2:
            st.metric("Predicted probability of diabetes", f"{prob:.3f}")

        st.caption(
            f"Prediction computed using a RandomForestClassifier with decision threshold `{threshold:.2f}`."
        )

        # SHAP explanations
        show_shap = st.checkbox("Show SHAP explanations", value=False)

        if show_shap:
            if not HAS_SHAP:
                st.warning(
                    "SHAP is not available in this environment. "
                    "Ensure `shap` is listed in `requirements.txt`."
                )
            elif (
                shap_explainer is None
                or X_bg_raw is None
                or shap_bg_vals is None
            ):
                st.warning(
                    "SHAP explainer could not be initialized for this model. "
                    "This may happen if the preprocessing step is incompatible."
                )
            else:
                # ---------- Global SHAP summary ----------
                st.subheader("Global feature importance (SHAP summary plot)")
                try:
                    shap.summary_plot(
                        shap_bg_vals,
                        X_bg_raw[FEATURES],
                        show=False,
                        plot_type="bar",
                    )
                    fig_global = plt.gcf()
                    st.pyplot(fig_global)
                    plt.close(fig_global)
                except Exception as e:
                    st.warning(f"Global SHAP summary failed: {e}")

                # ---------- Local SHAP explanation ----------
                st.subheader("Local explanation for this patient (top contributions)")
                try:
                    preproc = model.named_steps["preprocessor"]
                    X_proc_single = preproc.transform(df_in_model)

                    shap_raw_single = shap_explainer.shap_values(X_proc_single)

                    if isinstance(shap_raw_single, list):
                        # binary classification -> class 1
                        sv_local = (
                            shap_raw_single[1][0]
                            if len(shap_raw_single) > 1
                            else shap_raw_single[0][0]
                        )
                    else:
                        shap_vals_single = normalize_shap_values(shap_raw_single)
                        sv_local = shap_vals_single[0]

                    sv_local = np.array(sv_local).flatten()
                    if len(sv_local) > len(FEATURES):
                        sv_local = sv_local[: len(FEATURES)]

                    contrib = pd.Series(sv_local, index=FEATURES).sort_values(
                        key=lambda x: x.abs(), ascending=False
                    )

                    fig_local, ax_local = plt.subplots(figsize=(8, 5))
                    bars = ax_local.barh(range(len(contrib)), contrib.values)

                    for i, bar in enumerate(bars):
                        bar.set_color(
                            "red" if contrib.values[i] > 0 else "blue"
                        )

                    ax_local.set_yticks(range(len(contrib)))
                    ax_local.set_yticklabels(contrib.index)
                    ax_local.invert_yaxis()
                    ax_local.set_xlabel("SHAP value (impact on model output)")
                    ax_local.set_title("Feature contributions to prediction")
                    ax_local.axvline(x=0, color="black", linestyle="-", alpha=0.3)

                    st.pyplot(fig_local)
                    plt.close(fig_local)

                    st.subheader("Detailed feature contributions")
                    contrib_df = pd.DataFrame(
                        {
                            "Feature": contrib.index,
                            "SHAP Value": [f"{v:.4f}" for v in contrib.values],
                            "Impact": [
                                "Increases risk" if v > 0 else "Decreases risk"
                                for v in contrib.values
                            ],
                        }
                    ).reset_index(drop=True)
                    st.table(contrib_df)
                except Exception as e:
                    st.error(f"Local SHAP explanation failed: {str(e)}")


# -------------------------------------------------
# PAGE 2 – MODEL PERFORMANCE
# -------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model performance (Random Forest)")

    st.markdown("#### Test-set metrics")
    # simplest & safest: no styling errors
    st.dataframe(metrics_df, use_container_width=True)

    # ROC curve
    st.markdown("#### ROC curve (test set)")
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"RandomForest (AUC = {roc_auc_score(y_test, y_proba_test):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # Confusion matrix
    st.markdown("#### Confusion matrix (test set)")
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticklabels(["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax_cm.set_title("Confusion Matrix")
    fig_cm.colorbar(im, ax=ax_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)


# -------------------------------------------------
# PAGE 3 – EDA SNAPSHOT
# -------------------------------------------------
else:  # "EDA Snapshot"
    st.subheader("Exploratory data analysis snapshot")

    df = load_data()

    # Class balance
    st.markdown("#### Class balance (Outcome)")
    class_counts = df[TARGET].value_counts().sort_index()
    fig_cb, ax_cb = plt.subplots(figsize=(4, 4))
    ax_cb.bar(["No diabetes (0)", "Diabetes (1)"], class_counts.values)
    ax_cb.set_ylabel("Count")
    st.pyplot(fig_cb)
    plt.close(fig_cb)
    st.caption(
        "The target variable is moderately imbalanced (~65% no diabetes, ~35% diabetes). "
        "This motivated using metrics like Recall, F1-score, and ROC-AUC instead of accuracy alone."
    )

    # Key feature distributions
    st.markdown("#### Key feature distributions")
    fig_hist, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(df["Glucose"], bins=20)
    axes[0].set_title("Glucose")
    axes[0].set_xlabel("mg/dL")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["BMI"], bins=20)
    axes[1].set_title("BMI")
    axes[1].set_xlabel("kg/m²")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)
    st.caption(
        "Histograms show realistic ranges but also reveal data issues such as zero-inflated measurements, "
        "which are handled in the preprocessing pipeline."
    )

    # Correlation with outcome
    st.markdown("#### Correlation with diabetes outcome")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].sort_values(ascending=False)
    fig_corr, ax_corr = plt.subplots(figsize=(4, 4))
    ax_corr.barh(corr.index, corr.values)
    ax_corr.invert_yaxis()
    ax_corr.set_xlabel("Correlation with Outcome")
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    st.caption(
        "Glucose, BMI, Age, and Pregnancies show the strongest positive correlations with diabetes, "
        "supporting their role as key predictors in the model."
    )
