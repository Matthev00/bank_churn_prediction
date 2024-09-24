import streamlit as st
from data.data_prep import format_app_input_features
from utils import load_model
import mlflow.pyfunc


def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 350, 850, 715)
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure", 0, 10, 5)
    balance = st.sidebar.slider("Balance", 0, 250000, 50000)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.sidebar.selectbox("Has Credit Card", ("Yes", "No"))
    is_active_member = st.sidebar.selectbox("Is Active Member", ("Yes", "No"))
    estimated_salary = st.sidebar.slider("Estimated Salary", 0, 200000, 100000)
    country = st.sidebar.selectbox(
        "Country", ("France", "Spain", "Germany")
    )
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography": country,
        "Gender": gender
    }
    return input_dict



def main():
    st.write(
        """
            # Bank Churn Prediction App

            This app predicts the **Bank Churn** of a customer!
        """
    )
    st.sidebar.header("User Input Features")
    # example input

    input_dict = user_input_features()
    X = format_app_input_features(input_dict)

    # clf = load_model()

    # prediction = clf.predict(X)
    # print(prediction)


if __name__ == "__main__":
    main()
