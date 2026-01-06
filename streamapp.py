import joblib
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#load models
dt_model = joblib.load("models/decision_tree.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

#load metrics
with open("artifacts/metrics_dt.json") as f:
    metrics_dt = json.load(f)

with open("artifacts/metrics_xgb.json") as f:
    metrics_xgb = json.load(f)

#conf matrixes
conf_dt = np.load("artifacts/confusion_dt.npy")
conf_xgb = np.load("artifacts/confusion_xgb.npy")

#load class names
with open("artifacts/class_names.json") as f:
    class_names = json.load(f)
# load SHAP artifacts
shap_dt = np.load("artifacts/shap_dt_values.npy", allow_pickle=True)
shap_xgb = np.load("artifacts/shap_xgb_values.npy", allow_pickle=True)

with open("artifacts/shap_feature_names.json") as f:
    shap_feature_names = json.load(f)

sample_idx = np.load("artifacts/shap_sample_idx.npy")

#SHAP helper
def plot_shap_bar(shap_values, feature_names, title):
    shap_values = np.array(shap_values)

    """
    Supported SHAP shapes:
    - (n_samples, n_features)                     -> binary / regression
    - (n_samples, n_features, n_classes)          -> multiclass (your case)
    - (n_classes, n_features)                     -> already aggregated
    - (n_features,)                               -> already final
    """

    # Multiclass with samples: (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        shap_abs_mean = np.mean(np.abs(shap_values), axis=(0, 2))

    # Binary or regression: (n_samples, n_features)
    elif shap_values.ndim == 2 and shap_values.shape[1] == len(feature_names):
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0)

    # Already aggregated per class: (n_classes, n_features)
    elif shap_values.ndim == 2 and shap_values.shape[0] != len(feature_names):
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0)

    # Already final: (n_features,)
    elif shap_values.ndim == 1 and shap_values.shape[0] == len(feature_names):
        shap_abs_mean = np.abs(shap_values)

    else:
        st.error(f"Unexpected SHAP shape: {shap_values.shape}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_names, shap_abs_mean)
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP value|")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)


st.title("Tomato Leaf Disease Model Evaluation Dashboard")

model_choice = st.selectbox(
    "Pilih Model",
    ["Decision Tree", "XGBoost"]
)

if model_choice == "Decision Tree":
    st.subheader("Decision Tree Metrics")
    st.json(metrics_dt)

    fig, ax = plt.subplots()
    sns.heatmap(conf_dt, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Decision Tree Explainability (SHAP)")
    plot_shap_bar(
        shap_dt,
        shap_feature_names,
        "Global Feature Importance - Decision Tree"
    )

else:
    st.subheader("XGBoost Metrics")
    st.json(metrics_xgb)

    fig, ax = plt.subplots()
    sns.heatmap(conf_xgb, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Greens", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("XGBoost Explainability (SHAP)")
    plot_shap_bar(
        shap_xgb,
        shap_feature_names,
        "Global Feature Importance - XGBoost"
    )
