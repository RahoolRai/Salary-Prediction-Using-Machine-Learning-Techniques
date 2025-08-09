import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import sklearn
from packaging import version

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

# Create synthetic dataset
np.random.seed(42)
df = pd.DataFrame({
    "Experience": np.random.randint(0, 20, 100),
    "Age": np.random.randint(20, 60, 100),
    "Education": np.random.choice(["High School","Bachelors","Masters","PhD"], 100),
    "Skill_Level": np.random.choice(["Beginner","Intermediate","Advanced","Expert"], 100),
    "Salary": np.random.randint(20000, 120000, 100)
})

# Features & target
X = df.drop(columns=["Salary"])
y = df["Salary"]

# Preprocessing
num = ["Experience", "Age"]
cat = ["Education", "Skill_Level"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Handle OneHotEncoder version changes
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe)
])

pre = ColumnTransformer([
    ("num", num_pipe, num),
    ("cat", cat_pipe, cat)
])

# Model
pipe = Pipeline([
    ("preprocessor", pre),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and save only the full pipeline
pipe.fit(X, y)
joblib.dump(pipe, "models/best_salary_model.joblib")

print("✅ Saved: models/best_salary_model.joblib")
