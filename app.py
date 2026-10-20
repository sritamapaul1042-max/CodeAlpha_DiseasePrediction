import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

st.set_page_config(page_title="Disease Prediction Demo", page_icon="🩺")
st.title("🩺 Breast Cancer Classification Demo")
st.warning("Educational machine-learning demonstration only — not a medical diagnostic tool.")

@st.cache_resource
def train_model():
    ds = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.data, ds.target, test_size=0.2, random_state=42, stratify=ds.target
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return ds, model, X_test, y_test, pred

ds, model, X_test, y_test, pred = train_model()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, pred):.2%}")
col2.metric("Precision", f"{precision_score(y_test, pred):.2%}")
col3.metric("Recall", f"{recall_score(y_test, pred):.2%}")
col4.metric("F1-score", f"{f1_score(y_test, pred):.2%}")
st.caption(f"Dataset: {len(ds.data)} records, {ds.data.shape[1]} numeric features; 80% train / 20% test.")

st.subheader("Explore a held-out example")
idx = st.slider("Choose a test record", 0, len(X_test)-1, 0)
row = X_test.iloc[[idx]]
prediction = int(model.predict(row)[0])
actual = int(y_test.iloc[idx])
labels = ds.target_names
st.write("Model prediction:", f"**{labels[prediction]}**")
st.write("Dataset label:", f"**{labels[actual]}**")
st.write("Prediction matches dataset label:", prediction == actual)
st.dataframe(row.T.rename(columns={row.index[0]: "Feature value"}), use_container_width=True)

with st.expander("Feature importance (top 10)"):
    imp = pd.Series(model.feature_importances_, index=ds.feature_names).sort_values().tail(10)
    st.bar_chart(imp)
with st.expander("Classification report"):
    st.dataframe(pd.DataFrame(classification_report(y_test, pred, target_names=labels, output_dict=True, zero_division=0)).transpose())
with st.expander("Confusion matrix"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labels, ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
