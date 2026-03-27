from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import os

# ✅ IMPORTANT: Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Set experiment
mlflow.set_experiment("breast-cancer-rf-hp")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# GridSearch
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Start MLflow run
with mlflow.start_run(run_name="rf-gridsearch-parent") as parent:

    grid_search.fit(X_train, y_train)

    # ✅ Log all hyperparameter combinations as nested runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True, run_name=f"child_run_{i}"):

            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric(
                "accuracy",
                grid_search.cv_results_["mean_test_score"][i]
            )

    # ✅ Best results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best params & score
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    # ✅ Log datasets
    train_df = X_train.copy()
    train_df['target'] = y_train

    test_df = X_test.copy()
    test_df['target'] = y_test

    mlflow.log_input(mlflow.data.from_pandas(train_df), "training")
    mlflow.log_input(mlflow.data.from_pandas(test_df), "testing")

    # ✅ Log source code (safe way)
    script_path = "gridsearch_script.py"
    if os.path.exists(script_path):
        mlflow.log_artifact(script_path)

    # ✅ Log model
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "random_forest_model"
    )

    # ✅ Register model (VERY IMPORTANT 🔥)
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/random_forest_model",
        "breast_cancer_rf_model"
    )

    # ✅ Tags
    mlflow.set_tags({
        "author": "Shubham",
        "model_type": "RandomForest",
        "task": "classification"
    })

    print("Best Params:", best_params)
    print("Best Score:", best_score)