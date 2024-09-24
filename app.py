import streamlit as st
from data.data_prep import format_app_input_features, get_example_pd_input
from utils import load_model
import pandas as pd
import numpy as np
import xgboost


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


def color_negative_red(val):
    color = 'red' if val == 0 else 'green'
    return f'color: {color}'


def main():
    st.write(
        """
            # Bank Churn Prediction App

            This app predicts the **Bank Churn** of a customer!
        """
    )
    st.write("""
             Please fill in the details of the customer in the sidebar to predict the churn.
             This is example input
             """)
    st.dataframe(get_example_pd_input())
    st.sidebar.header("User Input Features")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file)
        
    else:
        input_dict = user_input_features()
        X = format_app_input_features(input_dict)

    clf = xgboost.XGBClassifier()
    clf.load_model("model.json")

    preds = clf.predict(X)
    preds_df = pd.DataFrame(preds, columns=['Exited'])

    st.subheader("Prediction")
    colored_pred = preds_df.style.applymap(color_negative_red, subset=['Exited'])
    st.dataframe(colored_pred)


if __name__ == "__main__":
    main()
