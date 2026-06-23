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

mlflow.set_tracking_uri(MLFLOW_URL)

# ── 2. Set experiment ────────────────────────────────────────────
mlflow.set_experiment(EXPERIMENT_NAME)

# ── 3. Start run ─────────────────────────────────────────────────
with mlflow.start_run(run_name="baseline-run"):

    # Dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        random_state=42,
        n_informative=8,
        n_redundant=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Hyperparameters
    C = 5.0
    max_iter = 500

    # ── 4. LOG PARAMS ────────────────────────────────────────────
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("model_type", "LogisticRegression")

    # Train
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # ── 5. LOG METRICS ───────────────────────────────────────────
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)

    # ── 6. LOG MODEL ─────────────────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="my-classifier"
    )

    print(f"Run complete. Accuracy: {accuracy:.4f}")
    print(f"View at: {MLFLOW_URL}")