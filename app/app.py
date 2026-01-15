import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Career Intelligence System",
    layout="wide"
)

st.title("🎓 Career Intelligence System")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎓 Placement Prediction",
    "💰 Salary Prediction",
    "🧭 Career Path Recommendation",
    "🚀 Startup Risk Analyzer",
    "🧠 Skill Gap Analyzer"
])

# ======================================================
# 🎓 Placement Prediction
# ========================================================

with tab1:

    # ---------- LOAD MODEL ----------
    MODEL_PATH = os.path.join("models", "placement_model.pkl")

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    placement_model = load_model()

    st.subheader("🎓 Placement Prediction")

    # ---------- INPUT FORM ----------
    st.markdown("### 👤 Student Profile")

    with st.form("placement_form"):
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        prev_sem_result = st.slider("Previous Semester Result", 0.0, 10.0, 7.0)
        academic_performance = st.slider("Academic Performance (1–10)", 1, 10, 7)

        internship_experience = st.selectbox(
            "Internship Experience", ["Yes", "No"]
        )

        communication_skills = st.slider("Communication Skills (1–10)", 1, 10, 6)
        projects_completed = st.number_input("Projects Completed", 0, 20, 2)
        extra_curricular_score = st.slider("Extra Curricular Score (1–10)", 1, 10, 5)

        submit = st.form_submit_button("Predict Placement")

    # ---------- FEATURE ENGINEERING + PREDICTION ----------
    if submit:

        # ---- SAME FEATURE ENGINEERING AS TRAINING ----

        # CGPA bucket
        if cgpa < 6:
            cgpa_bucket = "low"
        elif cgpa < 7.5:
            cgpa_bucket = "medium"
        else:
            cgpa_bucket = "high"

        # Internship flag
        has_internship = 1 if internship_experience == "Yes" else 0

        # Academic score
        academic_score = (prev_sem_result * 0.4) + (academic_performance * 0.6)

        # Skill score
        skill_score = (
            communication_skills +
            projects_completed +
            extra_curricular_score
        )

        # ---- FINAL INPUT (MATCH TRAINING FEATURES) ----
        input_df = pd.DataFrame([{
            # ---------- RAW FEATURES (EXPECTED BY PIPELINE) ----------
            "iq": 100,
            "cgpa": cgpa,
            "prev_sem_result": prev_sem_result,
            "academic_performance": academic_performance,
            "internship_experience": internship_experience,
            "communication_skills": communication_skills,
            "projects_completed": projects_completed,
            "extra_curricular_score": extra_curricular_score,

            # ---------- ENGINEERED FEATURES (ALREADY CREATED IN TRAINING DATA) ----------
            "cgpa_bucket": cgpa_bucket,
            "has_internship": has_internship,
            "academic_score": academic_score,
            "skill_score": skill_score
        }])


        # ---- PREDICTION ----
        prediction = placement_model.predict(input_df)[0]
        probability = placement_model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Student is likely to be **Placed**")
        else:
            st.error("❌ Student is likely to be **Not Placed**")

        st.info(f"📈 Placement Probability: **{probability * 100:.2f}%**")


# ==========================================================
# 💰 SALARY PREDICTION
# =========================================================
with tab2:

    st.subheader("💰 Salary Prediction")

    # ---------- LOAD SALARY MODEL ----------
    SALARY_MODEL_PATH = os.path.join("models", "salary_model.pkl")

    @st.cache_resource
    def load_salary_model():
        return joblib.load(SALARY_MODEL_PATH)

    salary_model = load_salary_model()

    st.markdown("### 👤 Job Profile")

    with st.form("salary_form"):
        years_of_experience = st.slider("Years of Experience", 0, 10, 1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        degree = st.selectbox("Degree", ["Bachelor", "Masters", "PhD"])
        stream = st.selectbox("Stream", ["CS", "IT", "ECE", "Mechanical", "Other"])

        submit_salary = st.form_submit_button("Predict Salary")

    # ---------- FEATURE ENGINEERING + PREDICTION ----------
    if submit_salary:

        # Create input exactly as training schema
        salary_input_df = pd.DataFrame([{
            # ---------- RAW / REQUIRED COLUMNS ----------
            "id": 0,
            "age": 22,
            "gpa": 8.0,
            "years_of_experience": years_of_experience,

            # ---------- ENGINEERED FEATURES ----------
            "is_fresher": 1 if years_of_experience == 0 else 0,
            "exp_salary_interaction": years_of_experience * 60000,
            "log_salary": 11.0,  # dummy, pipeline expects it

            # ---------- CATEGORICAL FEATURES ----------
            "gender": gender,
            "degree": degree,
            "stream": stream,
            "college_name": "Unknown",
            "placement_status": "Placed",
            "name": "Unknown"
        }])


        predicted_salary = salary_model.predict(salary_input_df)[0]

        st.markdown("### 📊 Salary Prediction Result")
        st.success(f"💵 Expected Salary: ₹ {predicted_salary:,.0f} per year")



# ===============================
# 🧭 Career Path Recommendation
# ===============================
with tab3:

    CAREER_MODEL_PATH = os.path.join("models", "career_path_model.pkl")
    CAREER_ENCODER_PATH = os.path.join("models", "career_path_encoder.pkl")
    SALARY_LEVEL_ENCODER_PATH = os.path.join("models", "salary_level_encoder.pkl")

    @st.cache_resource
    def load_career_models():
        career_model = joblib.load(CAREER_MODEL_PATH)
        career_encoder = joblib.load(CAREER_ENCODER_PATH)
        salary_level_encoder = joblib.load(SALARY_LEVEL_ENCODER_PATH)
        return career_model, career_encoder, salary_level_encoder

    career_model, career_encoder, salary_level_encoder = load_career_models()

    st.subheader("🧭 Career Path Recommendation")
    # st.caption("AI-based guidance for Job, Higher Studies, or Startup")

    st.markdown("### 👤 Career Profile")

    with st.form("career_path_form"):

        placed = st.selectbox(
            "Are you currently placed?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        salary = st.number_input(
            "Current / Expected Salary (₹ per year)",
            min_value=0,
            step=5000,
            value=60000
        )

        salary_level = st.selectbox(
            "Salary Level",
            ["Low", "Medium", "High"]
        )

        submit_career = st.form_submit_button("Recommend Career Path")

    if submit_career:

        # Encode salary level
        salary_level_enc = salary_level_encoder.transform([salary_level])[0]

        # Create input dataframe (same order as training)
        input_df = pd.DataFrame([{
            "placed": placed,
            "salary": salary,
            "salary_level_enc": salary_level_enc
        }])

        # Predict
        pred_class = career_model.predict(input_df)[0]
        pred_label = career_encoder.inverse_transform([pred_class])[0]

        st.markdown("---")
        st.subheader("🎯 Recommended Career Path")

        if pred_label == "Job":
            st.success("💼 **Recommended Path: JOB**")
            st.caption("Stable income and industry exposure suggested.")

        elif pred_label == "Higher Studies":
            st.info("🎓 **Recommended Path: HIGHER STUDIES**")
            st.caption("Skill & qualification enhancement advised.")

        else:
            st.warning("🚀 **Recommended Path: STARTUP**")
            st.caption("Entrepreneurial potential detected.")

# ==========================================
# 🚀 Startup Risk Analyzer
# ==========================================
with tab4:
    # ===== Startup Risk Model Loading =====

    STARTUP_MODEL_PATH = "models/startup_risk_model.pkl"
    RISK_ENCODER_PATH = "models/startup_risk_encoder.pkl"
    STARTUP_PREPROCESSOR_PATH = "models/startup_preprocessor.pkl"

    @st.cache_resource
    def load_startup_model():
        model = joblib.load(STARTUP_MODEL_PATH)
        risk_encoder = joblib.load(RISK_ENCODER_PATH)
        preprocessor = joblib.load(STARTUP_PREPROCESSOR_PATH)
        return model, risk_encoder, preprocessor

    startup_model, risk_encoder, startup_preprocessor = load_startup_model()

    st.subheader("🚀 Startup Risk Analyzer")
    # st.caption("AI-based analysis to estimate startup survival & risk level")

    st.markdown("### 🏭 Startup Profile")

    industry = st.selectbox(
        "Startup Industry",
        ["Technology", "E-commerce", "FinTech", "Healthcare", "Education", "Other"]
    )

    city = st.selectbox(
        "Startup City",
        ["Bangalore", "Mumbai", "Delhi", "Gurgaon", "Pune", "Other"]
    )

    investment_type = st.selectbox(
        "Funding Type",
        ["Seed", "Series A", "Series B", "Private Equity", "Debt", "Other"]
    )

    funding_amount = st.number_input(
        "Total Funding Amount (USD)",
        min_value=0,
        step=100000
    )

    analyze = st.button("Analyze Startup Risk")

    if analyze:
        # Create input dataframe (same feature names as training)
        input_df = pd.DataFrame([{
            "industry_reduced": industry.lower(),
            "city_reduced": city.lower(),
            "investment_type_grouped": investment_type.lower().replace(" ", "_"),
            "log_funding": np.log1p(funding_amount)
        }])

        # Preprocess
        input_enc = startup_preprocessor.transform(input_df)

        # Predict
        pred_class = startup_model.predict(input_enc)[0]
        pred_label = risk_encoder.inverse_transform([pred_class])[0]

        st.markdown("---")
        st.subheader("📊 Startup Risk Assessment")

        if pred_label == "Low Risk":
            st.success("🟢 **Low Risk Startup**")
            st.caption("Strong ecosystem & funding profile detected.")

        elif pred_label == "Medium Risk":
            st.warning("🟡 **Medium Risk Startup**")
            st.caption("Moderate risk — execution & scaling matter.")

        else:
            st.error("🔴 **High Risk Startup**")
            st.caption("High uncertainty — funding & market risk present.")


# ==========================================================
# 🧠 Skill Gap Analyzer
# =============================================================

with tab5:
    st.subheader("🧠 Skill Gap Analyzer")
    # st.caption(
    #     "AI-based analysis to compare required job skills with your current skills "
    #     "and suggest a personalized upskilling path."
    # )
    st.markdown("### 👤 Skill Profile")


    JOB_SKILL_MAP = {
        "Data Scientist": {
            "python", "numpy", "pandas", "sql", "machine learning",
            "statistics", "scikit-learn", "data visualization"
        },
        "Software Engineer": {
            "data structures", "algorithms", "java", "python",
            "git", "oop", "system design"
        },
        "Web Developer": {
            "html", "css", "javascript", "react", "node", "git", "api"
        },
        "AI / ML Engineer": {
            "python", "deep learning", "tensorflow", "pytorch",
            "machine learning", "nlp", "computer vision"
        },
        "Business Analyst": {
            "excel", "sql", "power bi", "tableau",
            "data analysis", "communication"
        },
        "Other": set()
    }


    with st.form("skill_gap_form"):

        job_role = st.selectbox(
            "🎯 Target Job Role",
            [
                "Data Scientist",
                "Software Engineer",
                "Web Developer",
                "AI / ML Engineer",
                "Business Analyst",
                "Other"
            ]
        )

        candidate_skills = st.text_area(
            "🛠️ Your Current Skills (comma-separated)",
            placeholder="Python, SQL, Machine Learning, Pandas, Git"
        )

        analyze_skill_gap = st.form_submit_button("Analyze Skill Gap")


    if analyze_skill_gap:

        if not candidate_skills.strip():
            st.warning("⚠️ Please enter your skills to analyze the gap.")
        else:
            # Normalize candidate skills
            candidate_skill_set = {
                s.strip().lower() for s in candidate_skills.split(",")
            }

            # Required skills for selected role
            required_skills = JOB_SKILL_MAP.get(job_role, set())

            if not required_skills:
                st.info("ℹ️ No predefined skill map for this role yet.")
            else:
                # Skill gap
                missing_skills = required_skills - candidate_skill_set
                matched_skills = required_skills & candidate_skill_set

                readiness_score = (
                    len(matched_skills) / len(required_skills)
                ) * 100

                st.markdown("---")
                st.subheader("📊 Skill Gap Analysis Result")

                # Readiness status
                if readiness_score >= 70:
                    st.success(f"✅ **Ready for {job_role}**")
                elif readiness_score >= 40:
                    st.warning(f"⚠️ **Partially Ready for {job_role}**")
                else:
                    st.error(f"❌ **Needs Upskilling for {job_role}**")

                st.info(f"🎯 **Readiness Score:** {readiness_score:.2f}%")

                # Show details
                st.markdown("#### ✅ Matched Skills")
                st.write(sorted(matched_skills) if matched_skills else "None")

                st.markdown("#### ❌ Missing Skills")
                st.write(sorted(missing_skills) if missing_skills else "None")



