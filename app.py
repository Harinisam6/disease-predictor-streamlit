import streamlit as st
import pandas as pd
import pickle
import numpy as np

# PAGE CONFIGURATION
st.set_page_config(page_title="DISEASE PREDICTION SYSTEM", layout="centered")

# CUSTOM CSS FOR LARGE TEXT & VERTICAL LAYOUT
st.markdown("""
    <style>
    .stRadio [data-testid="stWidgetLabel"] p {
        font-size: 24px !important;
        font-weight: bold;
        color: #1E3A8A;}
    .stRadio label {
        font-size: 20px !important;
        padding: 10px 0px;}
    .stCheckbox label {
        font-size: 18px !important;}
    .stButton button {
        height: 3em;
        width: 100%;
        font-size: 18px !important;}
    </style>
    """, unsafe_allow_html=True)

# LOAD MODELS (XGBOOST)
try:
    model = pickle.load(open("disease_model.pkl", "rb"))
    feature_columns = pickle.load(open("features.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except:
    st.error("Error: Ensure 'disease_model.pkl', 'features.pkl', and 'label_encoder.pkl' are in the folder.")
    st.stop()

# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = 1
if "sub_symptoms" not in st.session_state:
    st.session_state.sub_symptoms = {}

# CATEGORIZE SYMPTOMS
def auto_categorize_symptoms(symptoms):
    categories = {
        "Fever Related": ["fever","chill","sweat",],
        "Respiratory": ["cough","breath","chest","throat","nose","sputum"],
        "Gastrointestinal": ["abdominal","stomach","nausea","vomit","diarr","appetite","constipation"],
        "Neurological": ["headache","dizz","seizure","confusion","memory","unconscious"],
        "Skin": ["skin","rash","itch","yellow","bruise","ulcer"],
        "Musculoskeletal": ["joint","muscle","pain","cramp","weakness","stiff"],
        "Urinary / Renal": ["urine","bladder","kidney","burning_micturition"],
        "Cardiovascular": ["heart","palpitation","pressure","pulse"],
        "Eye / ENT": ["eye","ear","vision","hearing","nasal"],
        "General": ["fatigue","weight","loss","gain","malaise"]
    }
    categorized = {cat: [] for cat in categories}
    categorized["Other"] = []
    for s in symptoms:
        placed = False
        for cat, keywords in categories.items():
            if any(k in s.lower() for k in keywords):
                categorized[cat].append(s)
                placed = True
                break
        if not placed:
            categorized["Other"].append(s)
    return categorized

SYMPTOM_TREE = auto_categorize_symptoms(feature_columns)

# PAGE 1: USER INFORMATION
if st.session_state.page == 1:
    st.title("🩺 PATIENT REGISTRATION")
    with st.form("user_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=1, value=70)
        sex = st.selectbox("Sex", ["Male","Female","Other"])
        submit = st.form_submit_button("Next")
    
    if submit:
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.session_state.update({
            "name": name,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "page": 2
        })
        st.rerun()

# PAGE 2: SYMPTOMS & PREDICTION
elif st.session_state.page == 2:
    st.title(f"Patient: {st.session_state['name']}")
    st.info(f"Age: {st.session_state['age']} | Sex: {st.session_state['sex']} | BMI: {st.session_state['bmi']}")

    if st.button("⬅ Back to Registration"):
        st.session_state.page = 1
        st.rerun()

    st.markdown("---")
    
    # MAJOR SYMPTOMS SELECTION
    st.markdown("SELECT PRIMARY SYMPTOM CATEGORY")
    main_category = st.radio(
        "Choose the category that best describes the main issue:",
        options=list(SYMPTOM_TREE.keys()),
        index=0,
        label_visibility="collapsed"
    )

    # SUB SYMPTOMS SELECTION
    st.markdown(f"SELECT SPECIFIC SYMPTOMS AND SEVERITY — {main_category}")
    subs = SYMPTOM_TREE[main_category]
    
    for s in subs:
        col1, col2 = st.columns([3,2])
        with col1:
            is_checked = st.checkbox(s.replace("_"," ").title(), key=f"check_{s}")
        with col2:
            sev = st.selectbox("Severity", ["Mild","Moderate","Severe"], key=f"sev_{s}")
        if is_checked:
            st.session_state.sub_symptoms[s] = {"Mild":1,"Moderate":2,"Severe":3}[sev]
        else:
            st.session_state.sub_symptoms.pop(s, None)

    st.markdown("---")

    # PREDICTION LOGIC (XGBOOST)
    if st.button("Generate Diagnostic Prediction"):
        if not st.session_state.sub_symptoms:
            st.error("Please select at least one specific symptom below the category.")
        else:
            # BUILD INPUT ROW
            input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
            for s, weight in st.session_state.sub_symptoms.items():
                if s in input_data.columns:
                    input_data[s] = weight
            
            # PROBABILITY-BASED PREDICTION
            probs = model.predict_proba(input_data)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]

            st.subheader("ANALYSIS REPORT")
            st.success("BASED UPON THE SELECTED SYMPTOMS, THE MOST LIKELY CONDITIONS ARE:")

            for idx in top3_idx:
                st.write(f"**{le.classes_[idx]}** — {probs[idx]*100:.2f}%")
