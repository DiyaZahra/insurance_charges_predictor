"""Train the insurance-charges model and save it to model.joblib.

Usage:  python train.py            (expects insurance.csv in the project root)
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("insurance.csv").drop_duplicates()

X = df.drop(columns="charges")
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encoding happens inside the pipeline, so the app can pass raw inputs
# and there is no train/test leakage.
preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
      ["sex", "smoker", "region"])],
    remainder="passthrough",
)

model = Pipeline([
    ("prep", preprocess),
    ("gb", GradientBoostingRegressor(random_state=42)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"R2:  {r2_score(y_test, pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, pred):.0f}")

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
