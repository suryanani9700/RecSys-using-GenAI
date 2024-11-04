import json
import csv
import os

# Path to the .deepeval-cache.json file
cache_file_path = os.path.expanduser("/kapil_wanaskar/295B/RAGAs_Evaluation/DEEPEVAL_RESULTS_FOLDER/20241101_204734.json")
csv_file_path = os.path.expanduser("/kapil_wanaskar/295B/RAGAs_Evaluation/DEEPEVAL_RESULTS_FOLDER/5_testcases.csv")
# Check if the cache file exists
if not os.path.exists(cache_file_path):
    print(f"Cache file {cache_file_path} not found.")
else:
    # Load the JSON data
    with open(cache_file_path, "r") as cache_file:
        data = json.load(cache_file)

    # previous:
    # test_case_key = list(data["testCases"].keys())[0]
    # modify to:
    for test_case in data["testCases"]:  # Iterate over each test case

        # Convert the JSON-encoded key back to a dictionary
        test_case_data = test_case  # Each test case is already a dictionary
        input_text = test_case_data.get("input", "")  # Extract input
        expected_output = test_case_data.get("expectedOutput", "")  # Extract expected output
        actual_output = test_case_data.get("actualOutput", "")  # Extract actual output
        metrics_data = test_case_data.get("metricsData", [])  # Extract metricsData list

        # Handle cases where there may not be enough metrics data entries
        metric_0_data = metrics_data[0] if len(metrics_data) > 0 else {}
        metric_1_data = metrics_data[1] if len(metrics_data) > 1 else {}

        # Prepare row data with five columns as specified
        row = [
            input_text,                               # Column 1: input
            expected_output,                          # Column 2: expected_output
            actual_output,                            # Column 3: actual_output
            {                                         # Column 4: metricsData[0]
                "name": metric_0_data.get("name", ""),
                "score": metric_0_data.get("score", ""),
                "evaluation_model": metric_0_data.get("evaluationModel", ""),
                "reason": metric_0_data.get("reason", "")
            },
            {                                         # Column 5: metricsData[1]
                "name": metric_1_data.get("name", ""),
                "score": metric_1_data.get("score", ""),
                "evaluation_model": metric_1_data.get("evaluationModel", ""),
                "reason": metric_1_data.get("reason", "")
            }
        ]

        # Write the extracted data to CSV
        with open(csv_file_path, "a", newline="") as csv_file:  # Append each test case to CSV
            csv_writer = csv.writer(csv_file)

            # Write the header only for the first row
            if os.stat(csv_file_path).st_size == 0:
                csv_writer.writerow([
                    "input", "expected_output", "actual_output", "metricsData[0]", "metricsData[1]"
                ])

            # Write the row with each column containing the relevant information
            csv_writer.writerow([
                row[0],                         # input
                row[1],                         # expected_output
                row[2],                         # actual_output
                json.dumps(row[3]),             # metricsData[0] as JSON string
                json.dumps(row[4])              # metricsData[1] as JSON string
            ])

    print(f"Results saved to {csv_file_path}")