import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Predictor")

# Load model files (same Colab directory)
rf_model = pickle.load(open("disease_model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
top_30 = pickle.load(open("top_30_symptoms.pkl", "rb"))

st.subheader("Select Symptoms")
selected_symptoms = st.multiselect("Symptoms", top_30)

if st.button("Predict Disease 🩺"):
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    for s in selected_symptoms:
        input_df[s] = 1

    pred = rf_model.predict(input_df)[0]
    st.success(f"Predicted Disease: {le.classes_[pred]}")
