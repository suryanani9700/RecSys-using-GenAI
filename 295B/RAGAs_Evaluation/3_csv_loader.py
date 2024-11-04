import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Load the dataset from a CSV file
csv_file_path = "/kapil_wanaskar/295B/RAGAs_Evaluation/LLM recommendation Evaluation - Sheet1.csv"
data = pd.read_csv(csv_file_path)

print("first 3 column names: ", data.columns[:3])

# convert each cell to string
data = data.astype(str)
# check the data types of the columns
print(data.dtypes)

# Create a function to simulate context retrieval
def get_relevant_documents(query, vectorstore, k=3):
    return vectorstore.similarity_search(query, k=k)

# Create a vector store from the ground truth
loader = DataFrameLoader(data, page_content_column="ground_truth")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Generate retrieved_contexts
data['retrieved_contexts'] = data['question'].apply(lambda q: [doc.page_content for doc in get_relevant_documents(q, vectorstore)])

# Convert the DataFrame to a Hugging Face Dataset
eval_dataset = Dataset.from_pandas(data)

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

# Run the evaluation
results = evaluate(
    eval_dataset,
    metrics,
    llm=llm,
    embeddings=embeddings,
    column_map={
        "question": "question",
        "answer": "answer",
        "ground_truth": "ground_truth",
        "contexts": "retrieved_contexts"
    }
)

# Display results
print(results)