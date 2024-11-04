import json
import csv
import os

# Path to the .deepeval-cache.json file
cache_file_path = os.path.expanduser(".deepeval-cache.json")
csv_file_path = os.path.expanduser("deepeval_results.csv")

# Check if the cache file exists
if not os.path.exists(cache_file_path):
    print(f"Cache file {cache_file_path} not found.")
else:
    # Load the JSON data
    with open(cache_file_path, "r") as cache_file:
        data = json.load(cache_file)

    # Extract relevant fields for each metric in cached_metrics_data
    test_case_key = list(data["test_cases_lookup_map"].keys())[0]
    metrics_data = data["test_cases_lookup_map"][test_case_key]["cached_metrics_data"]

    # Prepare rows for CSV with two main columns as specified
    rows = []
    for i, metric in enumerate(metrics_data):
        metric_data = metric["metric_data"]
        row = {
            f"cached_metrics_data[{i}]": {
                "name": metric_data["name"],
                "score": metric_data["score"],
                "evaluation_model": metric_data["evaluationModel"],
                "reason": metric_data["reason"]
            }
        }
        rows.append(row)

    # Write the extracted data to CSV
    with open(csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow(["cached_metrics_data[0]", "cached_metrics_data[1]"])
        
        # Write the rows, with each column containing extracted information for a metric
        csv_writer.writerow([
            f"{rows[0]['cached_metrics_data[0]']}",
            f"{rows[1]['cached_metrics_data[1]']}"
        ])

    print(f"Results saved to {csv_file_path}")
