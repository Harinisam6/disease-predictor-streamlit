import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIGURATION
st.set_page_config(page_title="Disease Predictor", layout="centered")

# LOADING ALL THE PKL FILES
rf_model = pickle.load(open("disease_model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
top_30 = pickle.load(open("top_30_symptoms.pkl", "rb"))

# INITIALIZATION FOR THE SESSION
if 'page' not in st.session_state:
    st.session_state['page'] = 1  # default to page 1 (user info)

# CODE FOR PAGE 1
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
        st.session_state['page'] = 2  #AFTER COMPLETION GOES TO NEXT PAGE
        st.rerun()  # updated method

# SYMTOMS SELECTION IS DONE
if st.session_state['page'] == 2:
    st.title(f"Hello {st.session_state['name']}! 🩺")
    st.subheader("Let's predict your disease based on your symptoms")

    # USER INFORMATION DISPLAYED FROM PAGE 1
    st.markdown("**Your Info:**")
    st.info(f"**Name:** {st.session_state['name']}\n\n"
            f"**Age:** {st.session_state['age']}\n\n"
            f"**Sex:** {st.session_state['sex']}\n\n"
            f"**Weight:** {st.session_state['weight']} kg")

    # SELECTING THE SYMPTOMS
    selected_symptoms = st.multiselect("Select your symptoms:", top_30)

    if st.button("Predict Disease 🩺"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom!")
        else:
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            for s in selected_symptoms:
                input_df[s] = 1
                
            # PREDICTION OF DISEASE IS DONE
            pred = rf_model.predict(input_df)[0]
            disease = le.classes_[pred]

            # DISPLAYING ALL SYMPTOMS
            st.markdown("**Your Selected Symptoms:**")
            st.info(", ".join(selected_symptoms))

            # PREDICTED DISEASE
            st.markdown("**Predicted Disease:**")
            st.success(disease)

            # TELL WHETHER TO VISIT DOCTOR OR NOT
            serious_diseases = ['Cancer', 'Heart Disease', 'Diabetes']
            if disease in serious_diseases:
                st.warning("This seems serious. You should visit a doctor immediately!")
            else:
                st.info("This seems mild. You can monitor symptoms and consult a doctor if needed.")

    # START AGAIN BUTTON
    if st.button("Start Over"):
        for key in ['name', 'dob', 'age', 'weight', 'sex', 'page']:
            st.session_state.pop(key, None)
        st.rerun()  

