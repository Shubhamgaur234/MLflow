import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ IMPORTANT: Set tracking URI (fixes your main issue)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Set experiment
mlflow.set_experiment("Wine_Classification_Experiment")
# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Model parameters
max_depth = 10
n_estimators = 10

# Start MLflow run
with mlflow.start_run():

    # Train model
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ✅ Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # ✅ Log parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # ✅ Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")

    # Save plot
    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    # Log artifact
    mlflow.log_artifact(plot_path)

    # ✅ Tags
    mlflow.set_tags({
        "Author": "Shubham",
        "Project": "Wine Classification"
    })

    # ✅ Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")

    print("Accuracy:", accuracy)