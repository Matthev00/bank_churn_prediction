from data.data_prep import create_dataloaders
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import get_args
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
import xgboost as xgb


def main():
    args = get_args()

    X_train, X_test, y_train, y_test = create_dataloaders(
        data_path=args.data_path
    )

    clf = xgb.XGBClassifier(objective="binary:logistic", seed=42,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            max_depth=args.max_depth,
                            learning_rate=args.learning_rate,
                            gamma=args.gamma,
                            reg_lambda=args.reg_lambda,
                            scale_pos_weight=args.scale_pos_weight)

    mlflow.set_experiment("Bank_Churn_Modelling")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.xgboost.autolog()
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            verbose=True,
            early_stopping_rounds=10,
        )

        y_pred = clf.predict(X_test)
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
        mlflow.log_figure(fig, "cf_" + args.run_name + ".png")


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
