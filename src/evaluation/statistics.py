import os
import json
import pandas as pd
import numpy as np

# =========================================================
# RESULTS PATH
# =========================================================

RESULTS_PATH = "results/metrics"

# =========================================================
# LOAD ALL RESULTS
# =========================================================

records = []

for file in os.listdir(RESULTS_PATH):

    if file.endswith(".json"):

        file_path = os.path.join(
            RESULTS_PATH,
            file
        )

        with open(file_path, "r") as f:

            data = json.load(f)

        file_name = file.replace(".json", "")

        parts = file_name.split("_")

        model = parts[0]
        resolution = parts[1]

        data["model"] = model
        data["resolution"] = resolution

        records.append(data)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(records)

# =========================================================
# MEAN ± STD
# =========================================================

summary = df.groupby(

    ["model", "resolution"]

).agg(

    ["mean", "std"]

)

print(summary)

# =========================================================
# SAVE SUMMARY
# =========================================================

summary.to_csv(

    "results/summary/final_statistics.csv"

)

print("\nStatistics Saved.")