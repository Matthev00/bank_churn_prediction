from data.data_prep import create_dataloaders
import warnings

warnings.simplefilter(action="ignore", category=Warning)
from utils import get_args, mlflow_logging_decorator  # noqa 5501
import xgboost as xgb  # noqa 5501


@mlflow_logging_decorator
def main(logging_enabled: bool = True, run_name: str = "default Model"):
    args = get_args()

    X_train, X_test, y_train, y_test = create_dataloaders(data_path=args.data_path)

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        seed=42,
        subsample=0.9,
        colsample_bytree=0.5,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=args.scale_pos_weight,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        verbose=True,
        early_stopping_rounds=10,
    )

    y_pred = clf.predict(X_test)

    clf.save_model("model.json")

    return y_test, y_pred


if __name__ == "__main__":
    args = get_args()
    main(run_name=args.run_name, logging_enabled=args.logging_enabled)
