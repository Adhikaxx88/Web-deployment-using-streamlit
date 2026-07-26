# Student Placement Predictor

A Streamlit web app that predicts student job placement status and estimates salary, using pre-trained machine learning models.

## Features

The app has two prediction modes selectable from the sidebar:

1. **Classification** — Predicts whether a student will be Placed or Not Placed, complete with probability, a gauge chart, and a comparison bar chart.
2. **Regression** — Estimates salary (LPA) for a student assumed to be Placed, compared against benchmark values (min, average, max).

Student input data covers three categories:
- **Academic**: CGPA, number of backlogs, internships, certifications
- **Skills & Activity**: coding, communication, and aptitude ratings, extracurricular involvement
- **Lifestyle & Demographics**: study/sleep hours, stress level, gender, branch, family income level, city tier, internet access, part-time job

## Model

**Classification**
- Model: MLP Classifier
- Target: `placement_status`
- Metrics: Accuracy 89%, AUC 0.899

**Regression**
- Model: Linear Regression
- Target: `salary_lpa`
- Note: Trained only on students who were Placed

Models are stored in `.pkl` format:
- `best_model_classification.pkl`
- `best_model_regression.pkl`

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Plotly (gauge chart & bar chart visualizations)

## Project Structure

- `app.py` — Main Streamlit app
- `best_model_classification.pkl` — Classification model (MLP)
- `best_model_regression.pkl` — Regression model (Linear Regression)
- `requirements.txt` — Dependencies

## How to Run

1. Clone this repository:

   git clone https://github.com/Adhikaxx88/Web-deployment-using-streamlit.git

   cd Web-deployment-using-streamlit

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run app.py

4. Open your browser at `http://localhost:8501`

## How to Use

1. Choose a prediction mode in the sidebar (Classification or Regression)
2. Fill in the student data form
3. Click Predict/Estimate Salary
4. View the prediction result along with the probability/estimate visualization

## License

No license specified yet.
