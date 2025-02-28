import os
import json
import csv

# Define the fields to extract
FIELDS = [
    "success_rate",
    "success_rate_oracle",
    "y_initial_confidences",
    "y_final_confidences",
    "steps",
    "y_initial_pred_oracle_mae",
    "y_final_pred_oracle_mae",
]


def process_folder(folder):
    json_path = os.path.join(folder, "cf_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            data = json.load(json_file)

            csv_path = os.path.join(folder, "results_summary.csv")
            with open(csv_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                # Write the header
                writer.writerow(["metric", "mean", "std"])
                for field in FIELDS:
                    value = data.get(field, {})
                    if isinstance(value, dict):
                        writer.writerow(
                            [field, value.get("mean", "-1"), value.get("std", "-1")]
                        )
                    else:
                        writer.writerow([field, value, "-1"])
    else:
        print(f"File {json_path} does not exist.")


def main():
    exp_dir = "/home/tha/thesis_runs/ace/hyperparam"
    for folder in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, folder)
        print(f"Processing {folder_path}")
        if os.path.isdir(folder_path):
            process_folder(folder_path)


if __name__ == "__main__":
    main()
