from joblib import load
from fastapi import FastAPI

model = load("./model.joblib")

app = FastAPI()


@app.post("/predict")
def predict(excerpts: list[str]) -> list[float]:
    result: list[float] = model.predict(excerpts).tolist()
    return [round(value, 2) for value in result]
