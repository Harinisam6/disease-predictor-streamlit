import streamlit as st
import pandas as pd
import pickle

# --- CONFIG ---
st.set_page_config(page_title="Clinical Decision Support System", layout="centered")

# --- LOAD MODEL ---
rf_model = pickle.load(open("disease_model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# --- CATEGORIZE SYMPTOMS ---
def auto_categorize_symptoms(symptoms):
    categories = {
        "Fever Related": ["fever", "chill", "sweat", "temperature"],
        "Respiratory": ["cough", "breath", "chest", "throat", "nose", "sputum"],
        "Gastrointestinal": ["abdominal", "stomach", "nausea", "vomit", "diarr", "appetite", "constipation"],
        "Neurological": ["headache", "dizz", "seizure", "confusion", "memory", "unconscious"],
        "Skin": ["skin", "rash", "itch", "yellow", "bruise", "ulcer"],
        "Musculoskeletal": ["joint", "muscle", "pain", "cramp", "weakness", "stiff"],
        "Urinary / Renal": ["urine", "bladder", "kidney", "burning_micturition"],
        "Cardiovascular": ["heart", "palpitation", "pressure", "pulse"],
        "Eye / ENT": ["eye", "ear", "vision", "hearing", "nasal"],
        "General": ["fatigue", "weight", "loss", "gain", "malaise"]
    }

    categorized = {cat: [] for cat in categories}
    categorized["Other"] = []

    for symptom in symptoms:
        placed = False
        for cat, keywords in categories.items():
            if any(k in symptom for k in keywords):
                categorized[cat].append(symptom)
                placed = True
                break
        if not placed:
            categorized["Other"].append(symptom)

    return categorized

SYMPTOM_TREE = auto_categorize_symptoms(feature_columns)

# --- SESSION INIT ---
if "page" not in st.session_state:
    st.session_state.page = 1
if "main_symptom" not in st.session_state:
    st.session_state.main_symptom = None

# --- PAGE 1 ---
if st.session_state.page == 1:
    st.title("🩺 Clinical Decision Support System")
    st.subheader("Enter your basic information")

    with st.form("user_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        height = st.number_input("Height (cm)", min_value=50, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=1)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        submit = st.form_submit_button("Next")

    if submit:
        bmi = round(weight / ((height/100)**2), 2)
        if bmi < 18.5: bmi_status = "Underweight"
        elif bmi < 25: bmi_status = "Normal"
        elif bmi < 30: bmi_status = "Overweight"
        else: bmi_status = "Obese"

        st.session_state.update({
            "name": name,
            "age": age,
            "height": height,
            "weight": weight,
            "sex": sex,
            "bmi": bmi,
            "bmi_status": bmi_status,
            "page": 2
        })

# --- PAGE 2 ---
if st.session_state.page == 2:
    st.title(f"Hello {st.session_state.name}")
    st.subheader("Symptom-Based Risk Assessment")
    st.info(f"Age: {st.session_state.age} | Sex: {st.session_state.sex} | BMI: {st.session_state.bmi} ({st.session_state.bmi_status})")

    # Step 1: Choose main symptom
    st.subheader("Step 1: Select Main Symptom")
    main_symptom = st.selectbox("Choose the main symptom", feature_columns)
    st.session_state.main_symptom = main_symptom

    # Step 2: Filter relevant symptoms dynamically
    st.subheader("Step 2: Select other relevant symptoms & severity")
    selected_symptoms = {}
    relevant_symptoms = [s for s in feature_columns if st.session_state.main_symptom.split()[0] in s]  # simple filter

    for symptom in relevant_symptoms:
        col1, col2 = st.columns([3, 2])
        with col1:
            checked = st.checkbox(symptom.replace("_"," ").title(), key=symptom)
        with col2:
            severity = st.selectbox("Severity", ["Mild","Moderate","Severe"], key=f"{symptom}_sev")
        if checked:
            selected_symptoms[symptom] = {"Mild":1, "Moderate":2, "Severe":3}[severity]

    # Step 3: Predict using model
    if st.button("Predict Condition"):
        if not selected_symptoms:
            st.warning("Select at least one symptom.")
        else:
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            for symptom, weight in selected_symptoms.items():
                if symptom in input_df.columns:
                    input_df[symptom] = weight
            pred = rf_model.predict(input_df)[0]
            disease = le.classes_[pred]
            st.subheader("Prediction Result")
            st.success(disease)

    if st.button("Start Over"):
        st.session_state.clear()
        st.session_state.page = 1
