import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
data_path = "Regression.csv"
data = pd.read_csv(data_path)

# Streamlit app
def main():
    st.set_page_config(page_title="Prediksi Biaya Asuransi Kesehatan dengan Regresi Linear dan Random Forest", page_icon="💰", layout="wide")
    
    # Menampilkan penjelasan tujuan aplikasi
    st.markdown("""
    **Tujuan Aplikasi Prediksi Biaya Asuransi Kesehatan**

    Aplikasi ini bertujuan untuk memberikan perkiraan biaya asuransi kesehatan berdasarkan data pengguna, 
    seperti usia, jenis kelamin, BMI (Indeks Massa Tubuh), jumlah anak, status merokok, dan wilayah tempat tinggal.

    Aplikasi ini menggunakan model pembelajaran mesin (Linear Regression atau Random Forest) untuk memprediksi biaya 
    asuransi kesehatan yang mungkin dikenakan berdasarkan faktor-faktor tersebut.

    **Fitur Utama:**
    1. Input data pengguna seperti usia, jenis kelamin, BMI, jumlah anak, status merokok, dan wilayah.
    2. Prediksi biaya asuransi kesehatan berdasarkan data yang dimasukkan.
    3. Evaluasi kinerja model dengan menunjukkan RMSE (Root Mean Squared Error).
    4. Visualisasi prediksi untuk membantu pengguna memahami estimasi biaya.

    **Manfaat Aplikasi:**
    - Membantu pengguna merencanakan biaya asuransi kesehatan dengan lebih realistis.
    - Memberikan perkiraan biaya yang dipengaruhi oleh berbagai faktor.
    - Memudahkan pemahaman biaya asuransi melalui teknologi pembelajaran mesin.
    """)

    # Show dataset
    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(data)

    # Menampilkan grafik distribusi biaya asuransi
    st.subheader("Distribusi Biaya Asuransi")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['charges'], kde=True, color='skyblue')
    plt.title("Distribusi Biaya Asuransi Kesehatan")
    plt.xlabel("Biaya Asuransi (Charges)")
    plt.ylabel("Frekuensi")
    st.pyplot(plt)

    # Menampilkan grafik hubungan antara usia dan biaya asuransi
    st.subheader("Hubungan Usia dan Biaya Asuransi")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', data=data, color='orange')
    plt.title("Hubungan Usia dan Biaya Asuransi Kesehatan")
    plt.xlabel("Usia")
    plt.ylabel("Biaya Asuransi")
    st.pyplot(plt)

    # Menampilkan grafik hubungan antara BMI dan biaya asuransi
    st.subheader("Hubungan BMI dan Biaya Asuransi")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', data=data, color='green')
    plt.title("Hubungan BMI dan Biaya Asuransi Kesehatan")
    plt.xlabel("BMI (Indeks Massa Tubuh)")
    plt.ylabel("Biaya Asuransi")
    st.pyplot(plt)

    st.sidebar.header("Input Data Pengguna")

    # User input fields
    age = st.sidebar.slider("Usia", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
    sex = st.sidebar.selectbox("Jenis Kelamin", data.sex.unique())
    bmi = st.sidebar.slider("BMI (Indeks Massa Tubuh)", float(data.bmi.min()), float(data.bmi.max()), float(data.bmi.mean()))
    children = st.sidebar.slider("Jumlah Anak", int(data.children.min()), int(data.children.max()), int(data.children.mean()))
    smoker = st.sidebar.selectbox("Status Merokok", data.smoker.unique())
    region = st.sidebar.selectbox("Wilayah", data.region.unique())

    user_input = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    st.subheader("Data Input Pengguna")
    st.write(user_input)

    # Model selection
    st.sidebar.header("Pilih Model")
    model_type = st.sidebar.selectbox("Pilih Model", ["Regresi Linier", "Random Forest"])
    
    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider("Jumlah Pohon", 10, 200, 100)
    
    # Preprocessing and model pipeline
    categorical_features = ["sex", "smoker", "region"]
    numeric_features = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    if model_type == "Regresi Linier":
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

    st.subheader("Evaluasi Model")
    st.write(f"Model yang dipilih: {model_type}")
    st.write(f"RMSE: {rmse:.2f}")

    # Make predictions
    prediction = model.predict(user_input)

    st.subheader("Prediksi")
    st.write(f"Perkiraan Biaya Asuransi Kesehatan: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
