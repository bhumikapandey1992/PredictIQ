import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(X, y, task: str, model_type: str = "xgboost"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "xgboost":
        if task == "classification":
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
            )
    else:
        if task == "classification":
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbose=-1,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, verbose=-1,
            )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        }
    else:
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }

    importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(X.columns.tolist(), model.feature_importances_)
    }
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return model, metrics, importance
