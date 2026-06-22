import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MLFLOW_URL = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URL)

client = MlflowClient()  # ← this is your programmatic control panel

# ── Train 3 versions with different params ──────────────────────
configs = [
    {"C": 0.01, "max_iter": 50},
    {"C": 1.0,  "max_iter": 100},
    {"C": 10.0, "max_iter": 200},
]

mlflow.set_experiment("my-first-experiment")

run_ids = []

for cfg in configs:
    with mlflow.start_run() as run:
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"])
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_params(cfg)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="my-classifier"  # auto-increments version
        )

        run_ids.append(run.info.run_id)
        print(f"C={cfg['C']} | accuracy={accuracy:.4f} | run_id={run.info.run_id}")

# ── Find the BEST run by accuracy ───────────────────────────────
# Replace the search_runs block with this:
print("\n── Finding best run ──")

# Search across ALL experiments
runs = mlflow.search_runs(
    experiment_ids=None,          # ← None means all experiments
    search_all_experiments=True,  # ← explicit flag
    order_by=["metrics.accuracy DESC"],
    max_results=10                # get top 10, not just 1
)

# Match against registered versions
all_versions = client.search_model_versions("name='my-classifier'")
version_run_ids = {v.run_id: v.version for v in all_versions}

best_version = None
best_run_id = None
best_accuracy = None

for _, row in runs.iterrows():
    if row["run_id"] in version_run_ids:
        best_run_id = row["run_id"]
        best_version = version_run_ids[best_run_id]
        best_accuracy = row["metrics.accuracy"]
        break

print(f"Best run: {best_run_id} | accuracy: {best_accuracy:.4f} | version: {best_version}")

# ── Promote best version to Production ───────────────────────────
client.set_registered_model_alias(
    name="my-classifier",
    alias="production",         # alias = human-readable pointer
    version=best_version
)

# ── Tag it with metadata ──────────────────────────────────────────
client.set_model_version_tag(
    name="my-classifier",
    version=best_version,
    key="validated_by",
    value="your-name"
)

client.set_model_version_tag(
    name="my-classifier",
    version=best_version,
    key="accuracy",
    value=str(round(best_accuracy, 4))
)

print(f"\n✅ Version {best_version} promoted to Production alias!")

# ── How to LOAD production model anywhere in your org ────────────
# This is the magic line — your app team uses this, not a version number
model = mlflow.sklearn.load_model(f"models:/my-classifier@production")
print(f"\n✅ Model loaded successfully: {type(model)}")