# Bank Churn Prediction App


This project aims to develop a predictive model to determine the likelihood of a bank customer leaving the bank (churn). The core of this project is a machine learning model trained on historical customer data to make accurate predictions about customer churn.

### Model Selection and Experimentation

For this project, I selected the XGBoost algorithm due to its robustness and efficiency in handling structured data. To fine-tune the model and achieve optimal performance, I conducted a series of numerical experiments, adjusting various hyperparameters. These experiments were meticulously tracked and logged using MLflow, ensuring reproducibility and transparency in the model development process.

### Deployment

The trained model is deployed as a user-friendly web application using Streamlit. This application allows users to input customer data and receive real-time predictions on the likelihood of churn. The web interface is designed to be intuitive and accessible, making it easy for users to interact with the model and understand the predictions.

### Key Features

- **Model Training**: Utilizes XGBoost for high-performance predictions.
- **Hyperparameter Tuning**: Conducts extensive numerical experiments to fine-tune model parameters.
- **Experiment Tracking**: Uses MLflow to log and track experiments, ensuring reproducibility.
- **Web Deployment**: Deploys the model as a Streamlit web application for real-time predictions.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/bank-churn-prediction.git
    cd bank-churn-prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv bank_churn_prediction
    source bank_churn_prediction/Scripts/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Web Application

To run the Streamlit web application, execute the following command:
```sh
streamlit run app.py
```
This will start the web application, and you can interact with it through your web browser.

Running Experiments
To run experiments for training the model, execute:
```sh
python3 experiment.py
```
Running the Main Script
To run the main script, execute:
```sh
python3 train.py
```

Project Components

- app.py:
This file contains the Streamlit web application code. It allows users to input customer data and get churn predictions.

- experiment.py:
This file contains the code for running experiments with different hyperparameters using MLflow for tracking.

- data/data_prep.py:
This file contains functions for data loading, preprocessing, and creating data loaders.

- utils.py:
This file contains utility functions, including argument parsing and MLflow logging decorators.

- model.json:
This file contains the pre-trained XGBoost model used for predictions in the web application.