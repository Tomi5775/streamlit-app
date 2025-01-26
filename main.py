import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
data_path = "Regression.csv"
data = pd.read_csv(data_path)

# Streamlit app
def main():
    st.title("Health Insurance Cost Prediction")

    # Show dataset
    if st.checkbox("Show Dataset"):
        st.dataframe(data)

    st.sidebar.header("User Input Features")

    # User input fields
    age = st.sidebar.slider("Age", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
    sex = st.sidebar.selectbox("Sex", data.sex.unique())
    bmi = st.sidebar.slider("BMI", float(data.bmi.min()), float(data.bmi.max()), float(data.bmi.mean()))
    children = st.sidebar.slider("Number of Children", int(data.children.min()), int(data.children.max()), int(data.children.mean()))
    smoker = st.sidebar.selectbox("Smoker", data.smoker.unique())
    region = st.sidebar.selectbox("Region", data.region.unique())

    user_input = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    st.subheader("User Input Features")
    st.write(user_input)

    # Model selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])
    
    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    
    # Preprocessing and model pipeline
    categorical_features = ["sex", "smoker", "region"]
    numeric_features = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
        ], remainder="passthrough"
    )

    if model_type == "Linear Regression":
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])
    else:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=n_estimators, random_state=42))
        ])

    # Split the data
    X = data.drop(columns="charges")
    y = data["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Model Evaluation")
    st.write(f"Selected Model: {model_type}")
    st.write(f"RMSE: {rmse:.2f}")

    # Make predictions
    prediction = model.predict(user_input)

    st.subheader("Prediction")
    st.write(f"Predicted Insurance Cost: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
