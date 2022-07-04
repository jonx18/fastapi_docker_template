# Load the libraries
from fastapi import FastAPI, HTTPException
from sklearn import datasets
from joblib import load
import pandas as pd

# Load the model
clf = load(open('./models/model.pkl','rb'))
iris = datasets.load_iris()


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Classification FastAPI"}


# Define the route to the predictor
@app.post("/predict")
def predict(sepal_length,sepal_width,petal_length,petal_width):

    polarity = ""

    if(not(sepal_length)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid sepal_length")
    if(not(sepal_width)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid sepal_width")
    if(not(petal_length)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid petal_length")
    if(not(petal_width)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid petal_width")


    prediction = clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]

        
    return {
            "specie": iris.target_names[prediction]
           }