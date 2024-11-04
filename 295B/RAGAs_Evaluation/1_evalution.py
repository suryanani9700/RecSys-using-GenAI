# Import necessary libraries
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# Load the dataset from a CSV file
csv_file_path = "/kapil_wanaskar/295B/RAGAs_Evaluation/LLM recommendation Evaluation - Sheet1.csv"  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

print("first 3 column names: ", data.columns[:3])

# Convert the DataFrame to a Ragas EvaluationDataset with specified columns
eval_dataset = EvaluationDataset.from_dataframe(
    data,
    question_column="question",
    answer_column="answer",
    reference_column="ground_truth"
)

# Get the OpenAI API key from the environment
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.")
os.environ["OPENAI_API_KEY"] = openai_key

print("OpenAI API key set successfully.")

# Initialize Langchain LLM and Embeddings Wrappers
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Define the metrics for evaluation
metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

# Run the evaluation
results = evaluate(dataset=eval_dataset, metrics=metrics)

# Export and display results as a DataFrame
df = results.to_pandas()
df.head()
