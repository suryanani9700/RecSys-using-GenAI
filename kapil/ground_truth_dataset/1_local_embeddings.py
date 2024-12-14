import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pickle

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# Load environment variables
load_dotenv()

# Fetch the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation")

# Path to save FAISS index
faiss_index_path = "faiss_index.pkl"

# Define a function to save FAISS index
def save_faiss_index(faiss_index):
    with open(faiss_index_path, "wb") as f:
        pickle.dump(faiss_index, f)

# Define a function to load saved FAISS index
def load_faiss_index():
    if os.path.exists(faiss_index_path):
        with open(faiss_index_path, "rb") as f:
            faiss_index = pickle.load(f)
        return faiss_index
    return None

@st.cache_resource
def load_data():
    # Load the documents
    loader = CSVLoader(file_path="/kapil_wanaskar/295A/Grocery_and_Gourmet_Food_filtered_1000.csv")  # Use 20k dataset here
    documents = loader.load()

    # Check if FAISS index is already saved locally
    faiss_index = load_faiss_index()

    if faiss_index is not None:
        # If the FAISS index is saved, use it
        retriever = faiss_index.as_retriever()
    else:
        # Generate embeddings using OpenAI API and save them locally
        embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Create a FAISS vector store
        faiss_index = FAISS.from_documents(documents, embeddings_model)
        
        # Save the FAISS index for future use
        save_faiss_index(faiss_index)
        
        retriever = faiss_index.as_retriever()
    
    return retriever

retriever = load_data()

# Create a question-answering chain using the retriever
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=api_key),
    chain_type="stuff",
    retriever=retriever,
    input_key="question"
)

# User input
input_dish = st.text_input("Enter the dish you want to cook:", placeholder="Enter a dish here...")

user_query = f"From the column named 'title' in the dataset, recommend me 5 items to cook [not ready made] {input_dish}. Do not recommend similar items of the same category. e.g., do not recommend 2 types of rice as the same ingredient."

# When the user presses the 'Recommend Shopping List' button
if st.button("Recommend Shopping List"):
    response = chain({"question": user_query})
    if 'result' in response:
        st.write("Recommended Shopping List:")
        st.write(response['result'])
    else:
        st.write("No recommendations found")
