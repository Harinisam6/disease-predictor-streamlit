import streamlit as st
import pandas as pd
import pickle
import datetime

# PAGE CONFIGURATION
st.set_page_config(page_title="Disease Predictor", layout="centered")

# LOADING ALL THE PKL FILES
rf_model = pickle.load(open("disease_model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
top_30 = pickle.load(open("top_30_symptoms.pkl", "rb"))

# INITIALIZATION FOR THE SESSION
if 'page' not in st.session_state:
    st.session_state['page'] = 1

# PAGE 1 : USER INFO
if st.session_state['page'] == 1:
    st.title("🩺 Welcome to Disease Predictor")
    st.subheader("Please enter your details first:")

    with st.form("USER INFORMATION FORM"):
        name = st.text_input("Name")

        dob = st.date_input(
            "Date of Birth",
            min_value=datetime.date(1925, 1, 1),
            max_value=datetime.date(2025, 12, 31),
            value=datetime.date(2000, 1, 1)
        )

        age = st.number_input("Age", min_value=1, max_value=120)
        height = st.number_input("Height (cm)", min_value=50, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=1)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])

        submitted = st.form_submit_button("Next")

    if submitted:
        # BMI CALCULATION
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)

        # BMI CATEGORY
        if bmi < 18.5:
            bmi_status = "YOU'RE UNDERWEIGHT"
        elif 18.5 <= bmi < 25:
            bmi_status = "YOUR WEIGHT IS NORMAL"
        elif 25 <= bmi < 30:
            bmi_status = "YOU'RE OVERWIGHT"
        else:
            bmi_status = "YOU'RE OBESE"

        # SAVE TO SESSION
        st.session_state['name'] = name
        st.session_state['dob'] = dob
        st.session_state['age'] = age
        st.session_state['height'] = height
        st.session_state['weight'] = weight
        st.session_state['bmi'] = bmi
        st.session_state['bmi_status'] = bmi_status
        st.session_state['sex'] = sex
        st.session_state['page'] = 2

        st.rerun()

# PAGE 2 : SYMPTOMS & PREDICTION
if st.session_state['page'] == 2:
    st.title(f"Hello {st.session_state['name']}! 🩺")
    st.subheader("LET'S PREDICT YOUR CONDITION BASED ON YOUR SYMPTOMS")

    # USER INFORMATION DISPLAY
    st.markdown("**YOUR HEALTH PROFILE:**")
    st.info(
        f"**Name:** {st.session_state['name']}\n\n"
        f"**Age:** {st.session_state['age']}\n\n"
        f"**Sex:** {st.session_state['sex']}\n\n"
        f"**Height:** {st.session_state['height']} cm\n\n"
        f"**Weight:** {st.session_state['weight']} kg\n\n"
        f"**BMI:** {st.session_state['bmi']}\n\n"
        f"**Weight Status:** {st.session_state['bmi_status']}"
    )

    # SYMPTOMS SELECTION
    selected_symptoms = st.multiselect("Select your symptoms:", top_30)

    if st.button("Predict Disease 🩺"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom!")
        else:
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            for s in selected_symptoms:
                input_df[s] = 1

            # PREDICTION
            pred = rf_model.predict(input_df)[0]
            disease = le.classes_[pred]

            # DISPLAY SYMPTOMS
            st.markdown("**Your Selected Symptoms:**")
            st.info(", ".join(selected_symptoms))

            # DISPLAY RESULT
            st.markdown("**Predicted Disease:**")
            st.success(disease)

            # DOCTOR SUGGESTION
            serious_diseases = ['Cancer', 'Heart Disease', 'Diabetes']
            if disease in serious_diseases:
                st.warning("This seems serious.You should visit a doctor immediately!")
            else:
                st.info("This seems mild.You can monitor symptoms and consult a doctor if needed.")

    # START AGAIN BUTTON
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            st.session_state.pop(key)
        st.rerun()

