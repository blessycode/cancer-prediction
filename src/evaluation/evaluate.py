import os
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dataset import load_datasets
from src.evaluation.metrics import compute_metrics
from src.utils.model_io import load_keras_model

# =========================================================
# CONFIG
# =========================================================

MODELS = [

    "resnet50",

    "mobilenetv2",

    "efficientnetb0"

]

RESOLUTIONS = [

    224,

    128,

    64,

    32

]

RUNS = 1

NUM_CLASSES = 8
CLASS_NAMES = ["A", "AGC", "ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SC"]

overall_records = []

# =========================================================
# EVALUATION LOOP
# =========================================================

for model_name in MODELS:

    for resolution in RESOLUTIONS:

        for run in range(1, RUNS + 1):

            print("\n" + "="*50)

            print(
                f"Evaluating {model_name} "
                f"{resolution} Run {run}"
            )

            print("="*50)

            # =================================================
            # LOAD DATASET
            # =================================================

            dataset_path = os.path.join(

                "data",

                "processed",

                f"{resolution}x{resolution}"

            )

            _, _, test_ds = load_datasets(

                dataset_path,

                (resolution, resolution)

            )

            # =================================================
            # LOAD MODEL
            # =================================================

            model_path = os.path.join(

                "experiments",

                model_name,

                str(resolution),

                f"run_{run}",

                "best_model.keras"

            )

            model = load_keras_model(model_path)

            # =================================================
            # PREDICTIONS
            # =================================================

            y_true = []
            y_pred = []
            y_prob = []

            for images, labels in test_ds:

                predictions = model.predict(images)

                predicted_classes = np.argmax(
                    predictions,
                    axis=1
                )

                y_true.extend(labels.numpy())

                y_pred.extend(predicted_classes)

                y_prob.extend(predictions)

            y_true = np.array(y_true)

            y_pred = np.array(y_pred)

            y_prob = np.array(y_prob)

            # =================================================
            # METRICS
            # =================================================

            evaluation = compute_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
            metrics = evaluation["summary"]
            metrics_record = {
                "model": model_name,
                "resolution": resolution,
                "run": run,
                **metrics,
            }
            overall_records.append(metrics_record)

            print(json.dumps(metrics, indent=4))

            # =================================================
            # SAVE METRICS
            # =================================================

            save_path = os.path.join(

                "results",

                "metrics",

                f"{model_name}_{resolution}_run{run}.json"

            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:

                json.dump(evaluation, f, indent=4)

            overall_json_path = os.path.join(
                "results",
                "overall_metrics",
                f"{model_name}_{resolution}_run{run}_overall_metrics.json",
            )
            os.makedirs(os.path.dirname(overall_json_path), exist_ok=True)
            with open(overall_json_path, "w") as f:
                json.dump(metrics_record, f, indent=4)

            overall_csv_path = os.path.join(
                "results",
                "overall_metrics",
                f"{model_name}_{resolution}_run{run}_overall_metrics.csv",
            )
            pd.DataFrame([metrics_record]).to_csv(overall_csv_path, index=False)

            report_path = os.path.join(
                "results",
                "classification_reports",
                f"{model_name}_{resolution}_run{run}_classification_report.csv",
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            pd.DataFrame(evaluation["classification_report"]).transpose().to_csv(report_path)

            cm_path = os.path.join(
                "results",
                "confusion_matrices",
                f"{model_name}_{resolution}_run{run}_confusion_matrix.csv",
            )
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            pd.DataFrame(
                evaluation["confusion_matrix"],
                index=CLASS_NAMES,
                columns=CLASS_NAMES,
            ).to_csv(cm_path)

if overall_records:
    overall_df = pd.DataFrame(overall_records)
    os.makedirs(os.path.join("results", "overall_metrics"), exist_ok=True)
    overall_df.to_csv(
        os.path.join("results", "overall_metrics", "all_overall_metrics.csv"),
        index=False,
    )
    overall_df.to_json(
        os.path.join("results", "overall_metrics", "all_overall_metrics.json"),
        orient="records",
        indent=4,
    )

print("\nEvaluation Completed.")
