import mlflow
import sys
import os
from mlflow.tracking import MlflowClient

# ── Config ───────────────────────────────────────────────────────
MLFLOW_URL = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ.get("MODEL_NAME", "my-classifier")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "my-first-experiment")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))

mlflow.set_tracking_uri(MLFLOW_URL)
client = MlflowClient()

# ── Get latest registered version ────────────────────────────────
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    print("❌ No registered versions found")
    sys.exit(1)

latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
candidate_version = latest.version
candidate_run_id = latest.run_id

print(f"Candidate: {MODEL_NAME} Version {candidate_version}")
print(f"Run ID: {candidate_run_id}")

# ── Get candidate metrics ─────────────────────────────────────────
run = client.get_run(candidate_run_id)
candidate_accuracy = run.data.metrics.get("accuracy", 0)
print(f"Candidate accuracy: {candidate_accuracy:.4f}")

# ── Check threshold ───────────────────────────────────────────────
if candidate_accuracy < ACCURACY_THRESHOLD:
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=candidate_version,
        key="rejected_reason",
        value=f"accuracy {candidate_accuracy:.4f} below threshold {ACCURACY_THRESHOLD}"
    )
    print(f"❌ Rejected — below threshold {ACCURACY_THRESHOLD}")
    sys.exit(1)

# ── Compare against current champion ─────────────────────────────
try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "production")
    champion_run = client.get_run(champion.run_id)
    champion_accuracy = champion_run.data.metrics.get("accuracy", 0)
    print(f"Champion accuracy: {champion_accuracy:.4f}")

    if candidate_accuracy <= champion_accuracy:
        print(f"❌ Candidate doesn't beat champion — keeping Version {champion.version}")
        sys.exit(1)

    print(f"✅ Candidate beats champion by {candidate_accuracy - champion_accuracy:.4f}")

except Exception:
    print("No champion yet — first promotion")

# ── Promote to production ─────────────────────────────────────────
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=candidate_version
)

client.set_model_version_tag(
    name=MODEL_NAME,
    version=candidate_version,
    key="promoted_by",
    value="github-actions"
)

print(f"✅ Version {candidate_version} promoted to @production!")
sys.exit(0)
