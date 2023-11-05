import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("trained_model.sav",'rb'))

# Creating a funtion for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape to a row vector
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The person is not Diabetic"
    else:
        return "The Person is Diabetic"


def main():
    st.title("Diabetics Predictor")
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    # Creating input data field
    Pregnancies = st.text_input("Total Pregancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Thicknes of the skin")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("Body Mass Index Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Predigree Value")
    Age = st.text_input("Age of the Person")


    diagnosis = ''

    if st.button("Classify"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == "__main__":
    main()
