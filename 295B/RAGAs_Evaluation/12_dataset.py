import pandas as pd
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Load CSV file
file_path = '/kapil_wanaskar/295B/RAGAs_Evaluation/data_5_receipes.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)


# Extract the first five rows for test case generation
required_columns = ["input", "actual_output", "expected_output"]
data_head = data[required_columns]

# Generate the test cases
test_cases = [
    LLMTestCase(
        input=row["input"],
        actual_output=row["actual_output"],
        expected_output=row["expected_output"]
    )
    for _, row in data_head.iterrows()
]

# # Create the EvaluationDataset
dataset = EvaluationDataset(test_cases=test_cases)

# Display or use the dataset
print(dataset)

# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])

