# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Optional: shap import inside try because shap can be heavy
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


st.set_page_config(page_title="Diabetes Risk — Pima demo", layout="centered")

st.title("Diabetes risk prediction (Pima dataset model)")
st.write("Uses `pima_final_pipeline.pkl`. Make sure this file is in the same folder as `app.py`.")

MODEL_PATH = "pima_final_pipeline.pkl"


@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()
    model = joblib.load(path)
    return model


@st.cache_data
def create_empty_df():
    cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age"
    ]
    return pd.DataFrame([dict(zip(cols, [0,120,70,20,80,30.0,0.5,30]))])


# load model
model = load_model()

# Sidebar for inputs
st.sidebar.header("Patient input")

Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
Glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
BloodPressure = st.sidebar.number_input("BloodPressure (mmHg)", min_value=0, max_value=200, value=70, step=1)
SkinThickness = st.sidebar.number_input("SkinThickness (mm)", min_value=0, max_value=100, value=20, step=1)
Insulin = st.sidebar.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80, step=1)
BMI = st.sidebar.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1, format="%.1f")
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f")
Age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30, step=1)

input_dict = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age
}

df_in = pd.DataFrame([input_dict])

st.subheader("Input summary")
st.table(df_in.T.rename(columns={0: "value"}))


# Predict
if st.button("Predict risk"):

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_in)[:, 1]
        else:
            try:
                raw = model.decision_function(df_in)
                probs = (raw - raw.min()) / (raw.max() - raw.min())
            except Exception:
                probs = np.array([0.0])

        pred = (probs >= 0.5).astype(int)

    except Exception as e:
        st.error("Model inference failed: " + str(e))
        raise

    risk = float(probs[0])
    cls = int(pred[0])

    st.metric("Predicted class (0 = no diabetes, 1 = diabetes)", cls)
    st.metric("Predicted probability of diabetes", f"{risk:.3f}")

    st.caption("Default threshold = 0.5. Consider using a lower threshold to prioritize recall in screening contexts.")


    # SHAP explanations
    show_shap = st.checkbox("Show SHAP explanation (only supported for tree models)", value=False)

    if show_shap:
        if not HAS_SHAP:
            st.warning("SHAP not installed or failed to import in environment. Install shap to enable explanations.")
        else:
            try:
                if hasattr(model, "named_steps"):
                    clf = model.named_steps.get("classifier") or model.named_steps.get("clf")
                    preproc = model.named_steps.get("preprocessor")
                else:
                    clf = model
                    preproc = None

                if preproc is not None:
                    X_proc = preproc.transform(df_in)
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_proc)

                    shap.initjs()
                    fig = plt.figure(figsize=(6,3))

                    try:
                        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                    except Exception:
                        sv = shap_vals

                    feat_names = df_in.columns.tolist()
                    contrib = pd.Series(sv[0], index=feat_names).sort_values(key=lambda x: x.abs(), ascending=False)

                    contrib[:10].plot(kind="barh")
                    plt.title("Top SHAP contributions (this record)")
                    plt.gca().invert_yaxis()
                    st.pyplot(fig)

                else:
                    st.warning("Preprocessor not found inside the pipeline; SHAP explanation requires the preprocessing step.")

            except Exception as e:
                st.error("SHAP explanation failed: " + str(e))


# Batch upload for CSV predictions
st.sidebar.header("Batch / CSV scoring")

uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age",
    type=["csv"]
)

if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(batch_df.head())

    if st.sidebar.button("Run batch predict"):
        try:
            if hasattr(model, "predict_proba"):
                batch_probs = model.predict_proba(batch_df)[:, 1]
            else:
                raw = model.decision_function(batch_df)
                batch_probs = (raw - raw.min()) / (raw.max() - raw.min())

            batch_df["pred_prob"] = batch_probs
            batch_df["pred_class"] = (batch_df["pred_prob"] >= 0.5).astype(int)

            st.write(batch_df.head())

            st.download_button(
                "Download results CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv"
            )

        except Exception as e:
            st.error("Batch prediction failed: " + str(e))
