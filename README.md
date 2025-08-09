# Salary Prediction using Traditional ML Techniques

This is a runnable project (Task-04) prepared for the AI Internship. It uses a small synthetic dataset and a RandomForest pipeline as a demo so you can open the project in VS Code and run it immediately.

## How to run locally (VS Code)

1. Create a virtual environment (recommended):
   - `python -m venv .venv`
   - Activate it: Windows: `.venv\Scripts\activate`  macOS/Linux: `source .venv/bin/activate`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the Streamlit URL printed in terminal (usually http://localhost:8501).

## Project structure

```
salary_prediction_project/
├─ notebooks/
│  └─ salary_prediction_task04.ipynb
├─ streamlit_app.py
├─ models/
│  └─ best_salary_model.joblib  (pretrained demo pipeline)
├─ data/
│  └─ sample_data.csv
├─ requirements.txt
├─ README.md
```

Notes:
- `models/best_salary_model.joblib` is a demo model trained on synthetic data so you can test the app immediately.
- Replace `data/sample_data.csv` with the real Kaggle dataset if you want to re-train.
- The notebook contains the full EDA + training pipeline steps shown in the internship task.
