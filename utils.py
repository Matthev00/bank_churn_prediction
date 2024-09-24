import argparse
import mlflow
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Bank Churn Modelling using XGBoost")

    parser.add_argument("--data_path", type=str, default="Churn_Modelling.csv")
    parser.add_argument("--run_name", type=str, default="Best Model")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--reg_lambda", type=float, default=0)
    parser.add_argument("--scale_pos_weight", type=int, default=1)
    parser.add_argument("--logging_enabled", type=str2bool, default=True)

    return parser.parse_args()


def mlflow_logging_decorator(func):
    def wrapper(*args, **kwargs):
        run_name = kwargs.get("run_name", "default_run")
        logging_enabled = kwargs.get("logging_enabled", True)

        if not logging_enabled:
            return func(*args, **kwargs)
        mlflow.set_experiment("Bank_Churn_Modelling")
        with mlflow.start_run(run_name=run_name):
            mlflow.xgboost.autolog()

            y_test, y_pred = func(*args, **kwargs)

            cm = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("balanced_accuracy", balanced_accuracy)

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Did not leave", "Left"]
            )
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            mlflow.log_figure(fig, "confusion_matrix.png")

        return y_test, y_pred

    return wrapper


def load_model(model_uri: str = "models:/Churn_model/1"):
    model = mlflow.pyfunc.load_model(model_uri)
    return model
