import pandas as pd
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from tqdm import tqdm

# Load the dataset from a CSV file
csv_file_path = "/kapil_wanaskar/295B/RAGAs_Evaluation/LLM recommendation Evaluation - Sheet1.csv"
data = pd.read_csv(csv_file_path)

print("Column names and data types:")
print(data.dtypes)

# Convert float columns to string and replace NaN with empty string
float_columns = data.select_dtypes(include=['float64']).columns
for col in float_columns:
    data[col] = data[col].astype(str).replace('nan', '')

# Create a function to simulate context retrieval
def get_relevant_documents(query, vectorstore, k=3):
    if not isinstance(query, str) or query.strip() == '':
        return []
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error in similarity search for query '{query}': {e}")
        return []

# Create a vector store from the ground truth
loader = DataFrameLoader(data, page_content_column="ground_truth")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Generate retrieved_contexts
def safe_get_contexts(q):
    try:
        return [doc.page_content for doc in get_relevant_documents(q, vectorstore)]
    except Exception as e:
        print(f"Error processing query '{q}': {e}")
        return []

data['retrieved_contexts'] = data['question'].apply(safe_get_contexts)

# Get the OpenAI API key from the environment
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.")
os.environ["OPENAI_API_KEY"] = openai_key

print("OpenAI API key set successfully.")

# Initialize Langchain LLM and Embeddings
llm = ChatOpenAI(model="gpt-4")

# Define the metrics for evaluation
metrics = [
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall
]

# Function to evaluate a single row
def evaluate_row(row):
    results = {}
    for metric in metrics:
        try:
            metric_instance = metric()
            score = metric_instance.score(
                question=row['question'],
                answer=row['answer'],
                contexts=row['retrieved_contexts'],
                ground_truths=[row['ground_truth']],
                llm=llm
            )
            results[metric.__name__] = score
        except Exception as e:
            print(f"Error evaluating {metric.__name__} for question '{row['question']}': {e}")
            results[metric.__name__] = None
    return results

# Evaluate each row
all_results = []
for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating rows"):
    row_results = evaluate_row(row)
    all_results.append(row_results)

# Add results to the DataFrame
for metric in metrics:
    metric_name = metric.__name__
    data[metric_name] = [result[metric_name] for result in all_results]

# Display results
print("\nEvaluation Results:")
print(data[['question', 'answer', 'ground_truth'] + [metric.__name__ for metric in metrics]])

# Save results to CSV
output_csv_path = "/kapil_wanaskar/295B/RAGAs_Evaluation/evaluation_results.csv"
data.to_csv(output_csv_path, index=False)
print(f"\nResults saved to {output_csv_path}")