from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class phishingSitePrediction(BaseModel):
    SFH: int
    popUpWindow: int
    SSLfinal_State: int
    Request_URL: int
    URL_of_Anchor: int
    web_traffic: int
    URL_Length: int
    age_of_domain: int

# Defining a function to load the model
def load_model(filename):
    with open(f'../Models/{filename}.obj', 'rb') as f:
        return pickle.load(f)

@app.post("/predict")
def predict(req: phishingSitePrediction) -> dict[str, str]:
    rfcObj = load_model('phishingSitePredictorRfc')
    predicted_ans = rfcObj.predict([[req.SFH, req.popUpWindow, req.SSLfinal_State, req.Request_URL, req.URL_of_Anchor, req.web_traffic, req.URL_Length, req.age_of_domain]])
    if (predicted_ans == -1):
        return {"status": "Phishy site"}
    elif (predicted_ans == 1):
        return {"status": "Legitimate site"}
    else:
        return {"status": "Suspecious site"}
    