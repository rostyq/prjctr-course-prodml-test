from pandas import read_csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from joblib import dump


df = read_csv("./commonlitreadabilityprize/train.csv")


model = Pipeline([("tfidf", TfidfVectorizer()), ("svr", GradientBoostingRegressor())])

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

model.fit(df_train["excerpt"], df_train["target"])

print(
    "train RMSE: %.2f"
    % mean_squared_error(
        df_train["target"], model.predict(df_train["excerpt"]), squared=False
    )
)
print(
    "test: RMSE: %.2f"
    % mean_squared_error(
        df_test["target"], model.predict(df_test["excerpt"]), squared=False
    )
)

dump(model, "model.joblib")