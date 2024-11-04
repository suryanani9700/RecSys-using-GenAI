import json

# Load data from JSON file
with open('/kapil_wanaskar/295B/RAGAs_Evaluation/DEEPEVAL_RESULTS_FOLDER/20241101_204734.json', 'r') as file:
    data = json.load(file)

# Helper function to truncate a value to the first two words
def truncate_value(value):
    if isinstance(value, str):
        return ' '.join(value.split()[:2])  # Get the first two words of the string
    return value  # If it's not a string, return the original value

# Recursive function to print each key and truncated value with indentation
def print_json_structure(data, indent=0):
    spacing = '    ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}:")
            print_json_structure(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(f"{spacing}[{index}]:")
            print_json_structure(item, indent + 1)
    else:
        # Print only the first two words of each value
        print(f"{spacing}{truncate_value(data)}")

# Print the JSON structure
print_json_structure(data)
