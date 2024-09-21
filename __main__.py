from data.data_prep import create_dataloaders
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    X_train, X_test, y_train, y_test = create_dataloaders(data_path="Churn_Modelling.csv")

    clf = xgb.XGBClassifier(objective="binary:logistic", seed=42)

    mlflow.set_experiment("Bank_Churn_Modelling")
    with mlflow.start_run(run_name="basic_model"):
        mlflow.xgboost.autolog()
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc", verbose=True, early_stopping_rounds=10)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("balanced_accuracy", balanced_accuracy)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not leave', 'Left'])
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")


if __name__ == "__main__":
    main()