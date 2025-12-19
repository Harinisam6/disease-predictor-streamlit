import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
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
default_keys = ["page", "main_symptom_selected", "sub_symptoms", "name", "age", "height", "weight", "sex", "bmi", "bmi_status"]
for key in default_keys:
    if key not in st.session_state:
        if key == "page":
            st.session_state[key] = 1
        elif key == "sub_symptoms":
            st.session_state[key] = {}
        else:
            st.session_state[key] = None

# --- PAGE 1: Basic Info ---
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
        bmi = round(weight / ((height / 100) ** 2), 2)
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
        st.experimental_rerun()

# --- PAGE 2: Symptom Selection & Prediction ---
if st.session_state.page == 2:
    st.title(f"Hello {st.session_state.name}")
    st.subheader("Symptom-Based Risk Assessment")
    st.info(f"Age: {st.session_state.age} | Sex: {st.session_state.sex} | BMI: {st.session_state.bmi} ({st.session_state.bmi_status})")

    # Step 1: Main Symptom
    if st.session_state.main_symptom_selected is None:
        st.subheader("Step 1: Select Main Symptom")
        main_symptom = st.radio("Choose one main symptom:", list(SYMPTOM_TREE.keys()))
        if main_symptom:
            st.session_state.main_symptom_selected = main_symptom
            st.experimental_rerun()

    # Step 2: Sub-Symptoms
    if st.session_state.main_symptom_selected:
        sub_symptoms = SYMPTOM_TREE[st.session_state.main_symptom_selected]
        if sub_symptoms:
            st.subheader(f"Step 2: Select sub-symptoms for '{st.session_state.main_symptom_selected}'")
            for symptom in sub_symptoms:
                col1, col2 = st.columns([3, 2])
                with col1:
                    checked = st.checkbox(symptom.replace("_", " ").title(), key=symptom)
                with col2:
                    severity = st.selectbox(
                        "Severity", 
                        ["Mild", "Moderate", "Severe"], 
                        key=f"{symptom}_sev", 
                        disabled=not st.session_state.sub_symptoms.get(symptom)
                    )
                if checked:
                    st.session_state.sub_symptoms[symptom] = {"Mild": 1, "Moderate": 2, "Severe": 3}[severity]
                elif symptom in st.session_state.sub_symptoms:
                    del st.session_state.sub_symptoms[symptom]

    # Step 3: Prediction
    if st.button("Predict Condition"):
        if not st.session_state.sub_symptoms:
            st.warning("Select at least one sub-symptom.")
        else:
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

            # Map main symptom to feature columns more robustly
            main_sym = st.session_state.main_symptom_selected
            for col in feature_columns:
                if main_sym.lower() in col.lower():
                    input_df[col] = 3  # main symptom = severe

            for symptom, weight in st.session_state.sub_symptoms.items():
                if symptom in input_df.columns:
                    input_df[symptom] = weight

            pred = rf_model.predict(input_df)[0]
            disease = le.classes_[pred]
            st.subheader("Prediction Result")
            st.success(disease)

# --- Safe Start Over ---
if "start_over" not in st.session_state:
    st.session_state.start_over = False

if st.button("Start Over"):
    # Reset all session variables
    st.session_state.page = 1
    st.session_state.main_symptom_selected = None
    st.session_state.sub_symptoms = {}
    st.session_state.name = None
    st.session_state.age = None
    st.session_state.height = None
    st.session_state.weight = None
    st.session_state.sex = None
    st.session_state.bmi = None
    st.session_state.bmi_status = None
    # Set rerun flag instead of calling rerun immediately
    st.session_state.start_over = True

# Trigger rerun safely on next run
if st.session_state.start_over:
    st.session_state.start_over = False
    st.experimental_rerun()
