import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clinical Decision Support System", layout="centered")

# --- CUSTOM CSS FOR TEXT SIZE ---
st.markdown("""
    <style>
    /* Increase size of the main symptom radio button labels */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .stRadio label {
        font-size: 20px !important;
        font-weight: bold;
    }
    /* Style the sub-symptom checkboxes */
    .stCheckbox label {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
# Using try-except in case files are missing during testing
try:
    rf_model = pickle.load(open("disease_model.pkl", "rb"))
    feature_columns = pickle.load(open("features.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# --- SESSION STATE INIT ---
if "page" not in st.session_state:
    st.session_state.page = 1
if "sub_symptoms" not in st.session_state:
    st.session_state.sub_symptoms = {}
if "main_symptom_selected" not in st.session_state:
    st.session_state.main_symptom_selected = None

# --- CATEGORIZE SYMPTOMS ---
def auto_categorize_symptoms(symptoms):
    categories = {
        "Fever Related": ["fever","chill","sweat","temperature"],
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

# --- PAGE 1: USER INFO ---
if st.session_state.page == 1:
    st.title("🩺 Clinical Decision Support System")
    st.subheader("Enter your basic information")
    
    with st.form("user_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        height = st.number_input("Height (cm)", min_value=50, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=1)
        sex = st.selectbox("Sex", ["Male","Female","Other"])
        submit = st.form_submit_button("Next")
    
    if submit:
        # Calculate BMI
        bmi = round(weight/((height/100)**2),2)
        if bmi < 18.5: bmi_status = "Underweight"
        elif bmi < 25: bmi_status = "Normal"
        elif bmi < 30: bmi_status = "Overweight"
        else: bmi_status = "Obese"
        
        # Save to session state
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.bmi = bmi
        st.session_state.bmi_status = bmi_status
        st.session_state.page = 2
        st.rerun() # <--- CRITICAL: This clears Page 1 and refreshes to show Page 2

# --- PAGE 2: SYMPTOMS & PREDICTION ---
elif st.session_state.page == 2:
    st.title(f"Hello {st.session_state['name']}")
    st.subheader("Symptom-Based Risk Assessment")
    st.info(f"Age: {st.session_state['age']} | Sex: {st.session_state['sex']} | BMI: {st.session_state['bmi']} ({st.session_state['bmi_status']})")

    if st.button("⬅️ Start Over"):
        st.session_state.page = 1
        st.session_state.main_symptom_selected = None
        st.session_state.sub_symptoms = {}
        st.rerun()

    st.divider()

    # Step 1: Main symptom
    st.markdown("### **Step 1: Select Main Symptom Category**")
    main_symptom = st.radio(
        "Which area is bothering you most?", 
        options=list(SYMPTOM_TREE.keys()),
        horizontal=True # Optional: makes it easier to read on wide screens
    )
    st.session_state.main_symptom_selected = main_symptom

    # Step 2: Sub-symptoms
    if st.session_state.main_symptom_selected:
        st.markdown(f"### **Step 2: Specific Symptoms for {st.session_state.main_symptom_selected}**")
        available_subs = SYMPTOM_TREE[st.session_state.main_symptom_selected]
        
        if not available_subs:
            st.write("No specific sub-symptoms listed for this category.")
        else:
            for symptom in available_subs:
                col1, col2 = st.columns([3,2])
                with col1:
                    # Use a key to keep track of checkboxes
                    checked = st.checkbox(symptom.replace("_"," ").title(), key=f"chk_{symptom}")
                with col2:
                    severity = st.selectbox("Severity", ["Mild","Moderate","Severe"], key=f"sev_{symptom}")
                
                if checked:
                    st.session_state.sub_symptoms[symptom] = {"Mild":1,"Moderate":2,"Severe":3}[severity]
                else:
                    st.session_state.sub_symptoms.pop(symptom, None)

        st.divider()
        
        # Step 3: Prediction
        if st.button("🔍 Predict Condition", use_container_width=True):
            if not st.session_state.sub_symptoms:
                st.error("Please select at least one specific symptom.")
            else:
                # Prepare data
                input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
                for s, w in st.session_state.sub_symptoms.items():
                    if s in input_df.columns:
                        input_df[s] = w
                
                # Make prediction
                pred = rf_model.predict(input_df)[0]
                result = le.classes_[pred]
                
                st.balloons()
                st.subheader("Probable Condition:")
                st.success(f"**{result}**")
