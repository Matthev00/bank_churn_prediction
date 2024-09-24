from data.data_prep import create_dataloaders
import xgboost as xgb
import mlflow
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools


def main():
    data_path = "Churn_Modelling.csv"
    X_train, X_test, y_train, y_test = create_dataloaders(data_path=data_path)

    params = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.05],
        "gamma": [0, 0.25, 1.0],
        "reg_lambda": [0, 1.0, 10.0],
        "scale_pos_weight": [1, 4, 6],
    }

    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    mlflow.set_experiment("Bank_Churn_Modelling")
    for combo in combinations:
        model_name = "_".join([f"{key}_{value}" for key, value in combo.items()])
        with mlflow.start_run(run_name=model_name):
            clf = xgb.XGBClassifier(
                objective="binary:logistic",
                seed=42,
                subsample=0.9,
                colsample_bytree=0.5,
                **combo,
            )

            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="auc",
                verbose=True,
                early_stopping_rounds=10,
            )
            mlflow.xgboost.log_model(clf, artifact_path=model_name)

            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(
                y_test,
                y_pred,
            )

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("balanced_accuracy", balanced_accuracy)
            mlflow.log_metric("f1_score", f1)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Did not leave", "Left"]
            )
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            cf_name = (
                "cf"
                + "_".join([f"{key}_{value}" for key, value in combo.items()])
                + ".png"
            )
            mlflow.log_figure(fig, cf_name)


if __name__ == "__main__":
    main()
