import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🏥")


@st.cache_resource
def train_model():
    # Trains once when the app starts, from insurance.csv in the repo.
    df = pd.read_csv("insurance.csv").drop_duplicates()
    X = df.drop(columns="charges")
    y = df["charges"]

    preprocess = ColumnTransformer(
        [("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
          ["sex", "smoker", "region"])],
        remainder="passthrough",
    )
    model = Pipeline([
        ("prep", preprocess),
        ("gb", GradientBoostingRegressor(random_state=42)),
    ])
    model.fit(X, y)
    return model


model = train_model()

st.title("🏥 Medical Insurance Charges Predictor")
st.write("Enter details to estimate yearly medical insurance charges.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
    }])
    prediction = model.predict(row)[0]
    st.success(f"Estimated charges: **${prediction:,.0f}**")

st.caption("Model: Gradient Boosting Regressor. For educational use only.")
