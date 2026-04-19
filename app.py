import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page Config 

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Models 

@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

clf_model = load_model("best_model_classification.pkl")
reg_model = load_model("best_model_regression.pkl")

# Sidebar

with st.sidebar:
    st.title("🎓 Student Predictor")
    st.markdown("Aplikasi prediksi penempatan kerja dan estimasi gaji mahasiswa.")
    st.divider()

    page = st.radio(
        "Mode Prediksi",
        [" Classification", " Regression"],
        help="Classification → Placed / Not Placed\nRegression → Estimasi Salary (LPA)",
    )
    st.divider()

    with st.expander("ℹ Info Model"):
        if "Classification" in page:
            st.markdown("**Model:** MLP Classifier  \n**Target:** placement_status  \n**Metrik:** Accuracy 89% | AUC 0.899")
        else:
            st.markdown("**Model:** Linear Regression  \n**Target:** salary_lpa  \n**Catatan:** Hanya untuk mahasiswa Placed")

# Helper: Input Form 

def render_input_form(show_salary_note: bool = False) -> pd.DataFrame:
    if show_salary_note:
        st.info(" Model regresi dilatih hanya pada mahasiswa **Placed**. Input ini mengestimasi salary jika mahasiswa tersebut diterima.", icon="ℹ️")

    st.subheader(" Data Mahasiswa")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Akademik**")
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.01)
        backlogs = st.number_input("Backlogs", 0, 5, 0)
        internships = st.number_input("Internships Completed", 0, 4, 1)
        certifications = st.number_input("Certifications Count", 0, 9, 2)

    with col2:
        st.markdown("**Skill & Aktivitas**")
        coding = st.select_slider("Coding Skill Rating", [1, 2, 3, 4, 5], value=3)
        communication = st.select_slider("Communication Skill Rating", [1, 2, 3, 4, 5], value=3)
        aptitude = st.select_slider("Aptitude Skill Rating", [1, 2, 3, 4, 5], value=4)
        extracurricular = st.selectbox(
            "Extracurricular Involvement",
            ["Low", "Medium", "High", "unknown"],
        )

    with col3:
        st.markdown("**Gaya Hidup & Demografi**")
        study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 4.0, 0.1)
        sleep_hours = st.slider("Sleep Hours", 4.0, 9.0, 7.0, 0.1)
        stress_level = st.slider("Stress Level", 1, 10, 5)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "CE", "ME"])
        family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

    data = {
        "cgpa": cgpa,
        "study_hours_per_day": study_hours,
        "sleep_hours": sleep_hours,
        "coding_skill_rating": coding,
        "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude,
        "certifications_count": certifications,
        "stress_level": stress_level,
        "backlogs": backlogs,
        "internships_completed": internships,
        "family_income_level": family_income,
        "city_tier": city_tier,
        "extracurricular_involvement": extracurricular,
        "gender": gender,
        "internet_access": internet_access,
        "part_time_job": part_time_job,
        "branch": branch,
    }
    return pd.DataFrame([data])

#  Helper: Charts

def gauge_chart(value: float, title: str, suffix: str = "%", max_val: float = 100) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": suffix, "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar": {"color": "#4CAF50" if value >= max_val * 0.5 else "#F44336"},
            "steps": [
                {"range": [0, max_val * 0.4], "color": "#FFCDD2"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#FFF9C4"},
                {"range": [max_val * 0.7, max_val], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": max_val * 0.5,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def prob_bar_chart(placed_prob: float) -> go.Figure:
    not_placed_prob = 1 - placed_prob
    fig = go.Figure(go.Bar(
        x=[placed_prob * 100, not_placed_prob * 100],
        y=["Placed", "Not Placed"],
        orientation="h",
        marker_color=["#4CAF50", "#F44336"],
        text=[f"{placed_prob*100:.1f}%", f"{not_placed_prob*100:.1f}%"],
        textposition="inside",
    ))
    fig.update_layout(
        title="Probabilitas Prediksi",
        xaxis_title="Probabilitas (%)",
        xaxis={"range": [0, 100]},
        height=200,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def salary_bar_chart(predicted: float) -> go.Figure:
    avg_salary = 15.56
    min_salary = 10.0
    max_salary = 20.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Min (10 LPA)", "Rata-rata (15.56 LPA)", "Prediksi Anda", "Max (20 LPA)"],
        y=[min_salary, avg_salary, predicted, max_salary],
        marker_color=["#90CAF9", "#FFD54F", "#4CAF50", "#EF9A9A"],
        text=[f"{min_salary:.2f}", f"{avg_salary:.2f}", f"{predicted:.2f}", f"{max_salary:.2f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Salary Prediction vs Benchmark (LPA)",
        yaxis_title="Salary (LPA)",
        yaxis={"range": [0, 22]},
        height=320,
        margin=dict(t=50, b=20, l=10, r=10),
    )
    return fig

# Page: Classification 

def page_classification():
    st.title("🏷️ Placement Classification")
    st.markdown("Prediksi apakah seorang mahasiswa akan **Placed** atau **Not Placed**.")
    st.divider()

    with st.form("clf_form"):
        input_df = render_input_form()
        submitted = st.form_submit_button(" Prediksi", width="stretch", type="primary")

    if submitted:
        prob = clf_model.predict_proba(input_df)[0]
        placed_prob = prob[1]
        prediction = "Placed" if placed_prob >= 0.5 else "Not Placed"

        st.divider()
        st.subheader(" Hasil Prediksi")

        col_res, col_gauge, col_bar = st.columns([1, 1.5, 1.5])

        with col_res:
            if prediction == "Placed":
                st.success("### PLACED")
                st.markdown("Mahasiswa ini diprediksi **berhasil** mendapatkan pekerjaan.")
            else:
                st.error("###  NOT PLACED")
                st.markdown("Mahasiswa ini diprediksi **belum** mendapatkan pekerjaan.")

            st.metric("Probabilitas Placed", f"{placed_prob*100:.1f}%")
            st.metric("Probabilitas Not Placed", f"{(1-placed_prob)*100:.1f}%")

        with col_gauge:
            st.plotly_chart(
                gauge_chart(placed_prob * 100, "Placed Probability", suffix="%"),
                width="stretch",
            )

        with col_bar:
            st.plotly_chart(prob_bar_chart(placed_prob), width="stretch")

        with st.expander(" Lihat Input Data"):
            display_df = input_df.T.rename(columns={0: "Nilai"}).astype(str)
            st.dataframe(display_df, width="stretch")

# Page: Regression 

def page_regression():
    st.title("💰 Salary Regression")
    st.markdown("Estimasi **salary (LPA)** berdasarkan profil mahasiswa yang telah Placed.")
    st.divider()

    with st.form("reg_form"):
        input_df = render_input_form(show_salary_note=True)
        submitted = st.form_submit_button(" Estimasi Salary", width="stretch", type="primary")

    if submitted:
        predicted_salary = float(reg_model.predict(input_df)[0])
        predicted_salary = max(0.0, min(20.0, predicted_salary))

        st.divider()
        st.subheader(" Hasil Estimasi")

        col_res, col_gauge = st.columns([1, 2])

        with col_res:
            st.metric(
                label="💰 Estimasi Salary",
                value=f"{predicted_salary:.2f} LPA",
                delta=f"{predicted_salary - 15.56:+.2f} vs rata-rata",
            )

            if predicted_salary >= 18:
                st.success(" **Sangat Tinggi** Top earner!")
            elif predicted_salary >= 15:
                st.info("**Di atas rata-rata**  Bagus!")
            elif predicted_salary >= 12:
                st.warning("**Rata-rata**  Masih bisa ditingkatkan.")
            else:
                st.error(" **Di bawah rata-rata** —Perlu peningkatan skill.")

        with col_gauge:
            st.plotly_chart(
                gauge_chart(predicted_salary, "Predicted Salary", suffix=" LPA", max_val=20),
                width="stretch",
            )

        st.plotly_chart(salary_bar_chart(predicted_salary), width="stretch")

        with st.expander(" Lihat Input Data"):
            display_df = input_df.T.rename(columns={0: "Nilai"}).astype(str)
            st.dataframe(display_df, width="stretch")

# Router

if "Classification" in page:
    page_classification()
else:
    page_regression()
