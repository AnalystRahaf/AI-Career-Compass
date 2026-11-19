import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MultiLabelBinarizer

#import os

#print(os.getcwd())
#print(os.listdir())
df = pd.read_csv("data_science_job_realistic.csv")
print(df.columns)
df["skills"] = df["skills"].str.split(", ")
print(df["skills"].head())
mlb = MultiLabelBinarizer()

skills_encoded = pd.DataFrame(
    mlb.fit_transform(df["skills"]),
    columns=mlb.classes_,
    index=df.index
)
print(skills_encoded.head())
df = pd.concat(
    [df, skills_encoded],
    axis=1
)


y = df["salary_in_usd"]
basic_features = [
    "work_year",
    "job_title",
    "job_category",
    "employee_residence",
    "experience_level",
    "employment_type",
    "work_setting",
    "company_location",
    "company_size"
]

skill_columns = list(mlb.classes_)
X = df[basic_features + skill_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
categorical_features = [
    "job_title",
    "job_category",
    "employee_residence",
    "experience_level",
    "employment_type",
    "work_setting",
    "company_location",
    "company_size"
]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
))
   
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)

new_person = pd.DataFrame(0, index=[0], columns=X.columns)

#Org column
new_person["work_year"] = 2026
new_person["job_title"] = "Data Scientist"
new_person["job_category"] = "Data Science"
new_person["employee_residence"] = "US"
new_person["experience_level"] = "SE"
new_person["employment_type"] = "FT"
new_person["work_setting"] = "Remote"
new_person["company_location"] = "US"
new_person["company_size"] = "L"

# Skills
new_person["Python"] = 1
new_person["SQL"] = 1
new_person["AWS"] = 1
new_person["Power BI"] = 1

# Predict
salary = model.predict(new_person)

print("Predicted Salary: $", round(salary[0], 2))

import joblib

joblib.dump(model, "salary_model.pkl")

print("Model Saved!")
