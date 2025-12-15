import streamlit as st
import pandas as pd
import pickle

# --- Page config ---
st.set_page_config(page_title="Disease Predictor", layout="centered")

# --- Load model files ---
rf_model = pickle.load(open("disease_model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
top_30 = pickle.load(open("top_30_symptoms.pkl", "rb"))

# --- Initialize session state ---
if 'page' not in st.session_state:
    st.session_state['page'] = 1  # default to page 1 (user info)

# --- Page 1: User Info ---
if st.session_state['page'] == 1:
    st.title("🩺 Welcome to Disease Predictor")
    st.subheader("Please enter your details first:")

    with st.form("user_info_form"):
        name = st.text_input("Name")
        dob = st.date_input("Date of Birth")
        age = st.number_input("Age", min_value=1, max_value=120)
        weight = st.number_input("Weight (kg)", min_value=1)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        submitted = st.form_submit_button("Next")

    if submitted:
        st.session_state['name'] = name
        st.session_state['dob'] = dob
        st.session_state['age'] = age
        st.session_state['weight'] = weight
        st.session_state['sex'] = sex
        st.session_state['page'] = 2  # go to prediction page
        st.experimental_rerun()  # safe now

# --- Page 2: Symptom selection & prediction ---
if st.session_state['page'] == 2:
    st.title(f"Hello {st.session_state['name']}! 🩺")
    st.subheader("Let's predict your disease based on your symptoms")

    # --- User info summary ---
    st.markdown("**Your Info:**")
    st.info(f"**Name:** {st.session_state['name']}\n\n"
            f"**Age:** {st.session_state['age']}\n\n"
            f"**Sex:** {st.session_state['sex']}\n\n"
            f"**Weight:** {st.session_state['weight']} kg")

    # --- Symptoms selection ---
    selected_symptoms = st.multiselect("Select your symptoms:", top_30)

    if st.button("Predict Disease 🩺"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom!")
        else:
            # Input preparation
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            for s in selected_symptoms:
                input_df[s] = 1

            # Prediction
            pred = rf_model.predict(input_df)[0]
            disease = le.classes_[pred]

            # --- Display selected symptoms ---
            st.markdown("**Your Selected Symptoms:**")
            st.info(", ".join(selected_symptoms))

            # --- Prediction Result ---
            st.markdown("**Predicted Disease:**")
            st.success(disease)

            # --- Doctor visit suggestion ---
            serious_diseases = ['Cancer', 'Heart Disease', 'Diabetes']  # adjust per your model
            if disease in serious_diseases:
                st.warning("⚠️ This seems serious. You should visit a doctor immediately!")
            else:
                st.info("✅ This seems mild. You can monitor symptoms and consult a doctor if needed.")

    # --- Start Over Button ---
    if st.button("Start Over"):
        for key in ['name', 'dob', 'age', 'weight', 'sex', 'page']:
            st.session_state.pop(key, None)
        st.experimental_rerun()

