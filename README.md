# 💼 Salary Prediction Using Traditional Machine Learning

## 🚀 Project Overview
This project was completed as part of the *AI Internship (Task-04)* and demonstrates how to build an *end-to-end salary prediction system* using traditional machine learning techniques.  
By combining a *scikit-learn preprocessing pipeline* with *ensemble models*, the system estimates an employee’s salary based on demographic and skill-related attributes.

---

## 🎯 Objective
To design, train, and deploy a robust machine learning model that predicts salaries from structured data, while ensuring:
- Data preprocessing is automated
- The model is reproducible
- The application is deployable for real-world use cases

---

## 📂 Dataset Used
(Demo dataset provided — replace with real HR dataset for production)

*Features:*
- Experience — Total years of work experience
- Age — Candidate’s age
- Education — Educational qualification
- Skill_Level — Skill proficiency category

*Target:*
- Salary — Annual salary amount

---

## 🔍 What Was Delivered
- ✅ Full preprocessing pipeline (handling numeric & categorical data)
- ✅ Missing value imputation, scaling, and one-hot encoding
- ✅ Model training using Random Forest Regressor
- ✅ Model persistence with *joblib*
- ✅ Evaluation using MAE, RMSE, and R²
- ✅ Interactive *Streamlit app* for live salary predictions
- ✅ Deployment guide for *Render*

---

## 🧹 Data Preprocessing Steps
- Median imputation for numerical columns
- Most frequent imputation for categorical columns
- Scaling numeric features (StandardScaler)
- One-hot encoding categorical features
- ColumnTransformer for clean integration

---

## 🤖 Machine Learning Model
*Algorithm:* Random Forest Regressor  
*Reason:* Non-linear handling, feature interaction capture, robust performance with minimal tuning.

*Evaluation Metrics:*
| Metric | Value (Demo Data) |
|--------|-------------------|
| MAE    | ~2500             |
| RMSE   | ~3200             |
| R²     | ~0.92              |

---

## 🔍 Key Insights
- 📈 Experience and skill level are strong predictors of salary.
- 🎓 Higher education levels typically yield higher salaries.
- 👥 Age can correlate with salary, but experience is more important.

---

## 🎯 Recommendations
- ✅ Integrate into HR dashboards for instant candidate salary estimation.
- ✅ Expand features to include industry, location, and company size.
- ✅ Retrain quarterly to adapt to market salary trends.

---

## 💻 Streamlit Web Application

### 🚀 Features:
- Simple and clean UI for entering candidate details
- Real-time salary prediction
- Fully integrated preprocessing + prediction pipeline

### ▶ To Run the App Locally:
bash
pip install -r requirements.txt
streamlit run streamlit_app.py


---

## 🌐 Deployment on Render — Quick Steps
1. Push code to GitHub (ensure requirements.txt and streamlit_app.py are in root).
2. On Render: *New → Web Service* → Connect Repo.
3. Set *Build Command*:
bash
pip install -r requirements.txt

4. Set *Start Command*:
bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0

5. Deploy and access your public link.

---

## 📌 Tech Stack
- *Python* — Core programming
- *pandas, numpy* — Data handling
- *scikit-learn* — Preprocessing & modeling
- *joblib* — Model saving/loading
- *Streamlit* — Web interface
- *Matplotlib, Seaborn* — Visualization
