import os
import pandas as pd
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset

def load_openai_key():
    """Load OpenAI API key from environment variable and set it."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        print("OpenAI API key set successfully.")

def load_dataset_from_csv(file_path, required_columns=["input", "actual_output", "expected_output"]):
    """Load dataset from a CSV file and generate test cases."""
    data = pd.read_csv(file_path)
    data_head = data[required_columns].head(5)  # Limit to first 5 rows
    
    test_cases = [
        LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"]
        )
        for _, row in data_head.iterrows()
    ]
    return EvaluationDataset(test_cases=test_cases)

def define_metrics(threshold=0.5, model_name="gpt-4"):
    """Define the answer relevancy and correctness metrics."""
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold)
    
    correctness_metric = GEval(
        name="Correctness",
        model=model_name,
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        evaluation_steps=[
            "Evaluate how well the actual output aligns with the key ideas and concepts in the expected output.",
            "Check if the main points in the expected output are represented in the actual output, even if phrasing or specific details differ.",
            "Assess whether the actual output is consistent in meaning and relevance with the expected output without requiring an exact match."
        ]
    )
    return [answer_relevancy_metric, correctness_metric]

@pytest.mark.parametrize("test_case", load_dataset_from_csv('/kapil_wanaskar/295B/RAGAs_Evaluation/data_5_receipes.csv'))
def test_customer_chatbot(test_case: LLMTestCase):
    try:
        """Run the test for each test case in the dataset with defined metrics."""
        metrics = define_metrics()
        assert_test(test_case, metrics)
    except AssertionError as e:
        pytest.fail(str(e))
        print("Tests failed with the following details:")
        print(e)

# Execute the test
if __name__ == "__main__":
    # Load OpenAI key
    load_openai_key()
    
    # Run pytest
    pytest.main(["-v", __file__])
