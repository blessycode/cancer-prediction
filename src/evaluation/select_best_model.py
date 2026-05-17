import os
import json
import shutil
import pandas as pd


RESULTS_PATH = "results/metrics"
BEST_MODEL_DIR = "models/best_model"

os.makedirs(BEST_MODEL_DIR, exist_ok=True)

records = []

for file in os.listdir(RESULTS_PATH):
    if file.endswith(".json"):

        with open(os.path.join(RESULTS_PATH, file), "r") as f:
            data = json.load(f)
        if "summary" in data:
            data = data["summary"]

        name = file.replace(".json", "")
        parts = name.split("_")

        model = parts[0]
        resolution = parts[1]
        run = parts[2].replace("run", "")

        records.append({
            "model": model,
            "resolution": resolution,
            "run": run,
            **data
        })

df = pd.DataFrame(records)

selection_metric = "f1_macro" if "f1_macro" in df.columns else "f1_score"
best = df.sort_values(selection_metric, ascending=False).iloc[0]

print("Best Model:")
print(best)
print(f"Selection metric: {selection_metric}")

source_model = f"experiments/{best['model']}/{best['resolution']}/run_{best['run']}/best_model.keras"
destination_model = f"{BEST_MODEL_DIR}/best_model.keras"

shutil.copy(source_model, destination_model)

metadata = best.to_dict()

with open(f"{BEST_MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Best model copied to models/best_model/")
