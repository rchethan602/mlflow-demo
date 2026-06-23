import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
# ── 1. Point to YOUR MLflow server ──────────────────────────────
MLFLOW_URL = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ.get("MODEL_NAME", "my-classifier")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "my-first-experiment")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))

mlflow.set_tracking_uri(MLFLOW_URL)

# ── 2. Experiments = folders that group related runs ────────────
#    Think: one experiment per project/model type
mlflow.set_experiment("my-first-experiment")

# ── 3. A "run" = one training attempt with its own params/metrics
with mlflow.start_run(run_name="baseline-run"):

    # Fake dataset — ignore the ML, focus on the MLflow calls
    X, y = make_classification(n_samples=1000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Hyperparameters you want to track
    C = 0.1
    max_iter = 100

    # ── 4. LOG PARAMS — things you set before training ──────────
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("model_type", "LogisticRegression")

    # Train
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # ── 5. LOG METRICS — results after training ─────────────────
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    # ── 6. LOG MODEL — saves to S3 automatically ────────────────
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",           # folder name inside S3
        registered_model_name="my-classifier"  # registers in Model Registry
    )

    print(f"Run complete. Accuracy: {accuracy:.4f}")
    print(f"View at: {MLFLOW_URL}")