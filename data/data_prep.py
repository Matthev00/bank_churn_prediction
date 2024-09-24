import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import Tuple


def load_data(data_path: str) -> pd.DataFrame:
    current_path = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(current_path, data_path))


def drop_irrelevant_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=columns, axis=1, inplace=True)


def format_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column], axis=1).copy()
    y = df[target_column].copy()
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include="object").columns)
    return X_encoded, y


def create_dataloaders(
    data_path: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = load_data(data_path=data_path)
    drop_irrelevant_columns(df=data, columns=["RowNumber", "CustomerId", "Surname"])
    X, y = format_data(df=data, target_column="Exited")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_some_X(data_path: str = "Churn_Modelling.csv") -> pd.DataFrame:
    data = load_data(data_path=data_path)
    drop_irrelevant_columns(df=data, columns=["RowNumber", "CustomerId", "Surname"])
    X = data.drop(columns=["Exited"], axis=1).copy()
    return X


def format_app_input_features(user_input_features):
    X = pd.DataFrame(user_input_features, index=[0])

    yes_no_columns = ["HasCrCard", "IsActiveMember"]
    for col in yes_no_columns:
        if col in X.columns:
            X[col] = X[col].map({"Yes": 1, "No": 0})

    X = pd.concat([X, get_some_X()], axis=0)
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include="object").columns)

    return X_encoded.iloc[0]


def main():
    X_train, X_test, y_train, y_test = create_dataloaders(
        data_path="Churn_Modelling.csv"
    )
    print(X_train.head())
    print(y_train.head())


if __name__ == "__main__":
    main()
