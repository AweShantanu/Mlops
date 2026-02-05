from mlflow.tracking import MlflowClient
import json
from src.mlflow_config import setup_mlflow
setup_mlflow()

client = MlflowClient()

# Option 1: Read from file
info_path = "reports/experiment_info.json"
try:
    info = json.load(open(info_path))
    run_id = info["run_id"]
    print("Using run_id from", info_path, ":", run_id)
except Exception as e:
    print("Could not read reports/experiment_info.json:", e)
    
    # Fallback: Get the latest run from the experiment
    experiment = client.get_experiment_by_name("my-dvc-pipeline")
    if experiment:
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            print("Using latest run_id:", run_id)
        else:
            run_id = input("Enter run_id to inspect: ").strip()
    else:
        run_id = input("Enter run_id to inspect: ").strip()

print("\n" + "="*50)
print(f"Checking artifacts for run: {run_id}")
print("="*50 + "\n")

artifacts = client.list_artifacts(run_id)
print("Top-level artifacts:")
for a in artifacts:
    print(" -", a.path, "(is_dir=" + str(a.is_dir) + ")")
    
# If there is a folder (like 'model'), list inside it
for a in artifacts:
    if a.is_dir:
        print("\nContents of", a.path, ":")
        sub = client.list_artifacts(run_id, path=a.path)
        for s in sub:
            print("  -", s.path, "(is_dir=" + str(s.is_dir) + ")")