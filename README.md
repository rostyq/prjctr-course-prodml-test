# Prjctr course "Machine Learning in Production" test task

## Stage 1

Open [the notebook](./CommonLit%20Readability%20Prize.ipynb) and run it to the last cell to get a trained model at `./model.joblib`.

> Kaggle CLI should be installed.

Or:

1. Install train dependencies (use virtual environment):

```
pip install -r requirements.train.txt
```

2. Download [CommonLib Readability Prize dataset](https://www.kaggle.com/competitions/commonlitreadabilityprize/data) and unzip data to `./commonlitreadabilityprize/`.

3. Run train script to get a trained model at `./model.joblib`:

```
python train.py
```

## Stage 2

1. Install API dependencies (use virtual environment):

```
pip install -r requirements.api.txt
```

2. Run server:

```
python -m uvicorn api:app
```

3. Test predict using `./example.json`:

```
curl -H "Content-Type: application/json" --data "@example.json" -X POST http://localhost:8000/predict
```

4. Output should be:

```
[-0.8]
```

## Stage 3

TODO: