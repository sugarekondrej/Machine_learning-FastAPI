from fastapi import FastAPI, HTTPException
from enum import Enum
import pickle
import numpy as np
import json

app=FastAPI()
class Educations(str,Enum):
    Bachelors_degree ="Bachelor’s degree"
    Masters_degree = "Master’s degree"
    Professional_degree = "Post grad"
    Less_than_bachelors = "Less than a Bachelors"

class Countries(str,Enum):
    United_states="United States"
    India = "India"
    United_kingdom ="United Kingdom"
    Germany = "Germany"
    Canada = "Canada"
    Brazil = "Brazil"
    France = "France"
    Spain = "Spain"
    Australia = "Australia"
    Netherlands= "Netherlands"
    Poland = "Poland"
    Italy = "Italy"
    Russian = "Russian Federation"
    Sweden = "Sweden"


def load_model():
    with open('salary_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
data= load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


@app.post("/")
def salary_predict(country:Countries,education:Educations,exp_years:float):
    if exp_years > 50:
        raise HTTPException(status_code=400, detail="Value of experience years must be lower than 50")
    elif exp_years <=0:
        raise HTTPException(status_code=400, detail="Value of experience years must be higher than 0")
    X = np.array([[country.value, education.value, exp_years ]])
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)
    salary = regressor.predict(X)
    salary[0] = float(f"{salary[0]:.0f}")
    return {"Predicted annual salary is: ": f"{json.dumps(salary.tolist())}$"}