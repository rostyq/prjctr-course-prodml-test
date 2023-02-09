from joblib import load
from fastapi import FastAPI
from starlette.responses import FileResponse

model = load("./model.joblib")

app = FastAPI()


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict(excerpts: list[str]) -> list[float]:
    result: list[float] = model.predict(excerpts).tolist()
    return [round(value, 2) for value in result]
