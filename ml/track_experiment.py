
from pathlib import Path
import json

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "models" / "candidates"

MODEL_PATH = CANDIDATE_DIR / "xgboost_decision_time.json"
PREPROCESSOR_PATH = CANDIDATE_DIR / "decision_time_preprocessor.joblib"
THRESHOLD_PATH = CANDIDATE_DIR / "threshold_comparison.csv"
TEST_RESULTS_PATH = CANDIDATE_DIR / "test_evaluation.json"

# Store MLflow tracking data in a local SQLite database.
TRACKING_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("fraud-detection")

def main():
    # Load the recorded validation threshold experiment.
    threshold_df = pd.read_csv(THRESHOLD_PATH)

    with open(TEST_RESULTS_PATH, encoding="utf-8") as file:
        test_results = json.load(file)

    with mlflow.start_run(run_name="decision-time-candidate-v1"):
        # Log model identity and the provisional decision threshold.
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("feature_set", "decision_time_candidate")
        mlflow.log_param("threshold", test_results["threshold"])
        mlflow.log_param("training_sample_rows", 200000)
        mlflow.log_param("validation_sample_rows", 100000)
        mlflow.log_param("threshold_candidates_tested", len(threshold_df))

        # Log the previously recorded untouched-test metrics.
        # These are evaluation results, not training or tuning inputs.
        mlflow.log_metric(
            "test_average_precision",
            test_results["average_precision"],
        )
        mlflow.log_metric("test_precision", test_results["precision"])
        mlflow.log_metric("test_recall", test_results["recall"])
        mlflow.log_metric("test_f1_score", test_results["f1_score"])
        mlflow.log_metric(
            "test_true_positives", test_results["true_positives"]
        )
        mlflow.log_metric(
            "test_false_positives", test_results["false_positives"]
        )
        mlflow.log_metric(
            "test_false_negatives", test_results["false_negatives"]
        )
        mlflow.log_metric(
            "test_true_negatives", test_results["true_negatives"]
        )

        # Attach existing model and experiment files to the run.
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_files")
        mlflow.log_artifact(
            str(PREPROCESSOR_PATH), artifact_path="model_files"
        )
        mlflow.log_artifact(
            str(THRESHOLD_PATH), artifact_path="validation"
        )
        mlflow.log_artifact(
            str(TEST_RESULTS_PATH), artifact_path="evaluation"
        )

        print("MLflow run recorded successfully.")
        print("Experiment: fraud-detection")
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Test average precision:", test_results["average_precision"])
        print("Test recall:", test_results["recall"])
        print("Test precision:", test_results["precision"])
        print("Production threshold finalized: No")

if __name__ == "__main__":
    main()
