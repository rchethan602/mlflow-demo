import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

MLFLOW_URL = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "my-first-experiment")

mlflow.set_tracking_uri(MLFLOW_URL)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="baseline-run"):

    # Simple dataset — guaranteed high accuracy
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,  # ← easiest possible classification
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    C = 1.0
    max_iter = 1000

    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("model_type", "LogisticRegression")

    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy:.4f}")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="my-classifier"
    )

    print(f"Run complete. Accuracy: {accuracy:.4f}")
    print(f"View at: {MLFLOW_URL}")
