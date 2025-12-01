from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="AG News Text Classifier Microservice")

# Load model and vectorizer
model = joblib.load("model.pkl")

class InputText(BaseModel):
    text: str

labels = ["World", "Sports", "Business", "Sci/Tech"]

@app.post("/predict")
def predict_text(input_data: InputText):
    text = input_data.text
    vect = model["vectorizer"].transform([text])
    pred = model["model"].predict(vect)[0]
    return {"prediction": labels[pred]}
