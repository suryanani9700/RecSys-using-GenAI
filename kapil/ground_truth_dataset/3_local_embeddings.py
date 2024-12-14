# from langchain_community.indexes.vectorstore import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
import streamlit as st


def load_openai_key():
    """Load OpenAI API key from environment variable and set it."""
    openai_key = os.getenv("OPENAI_API_KEY")
    print("OpenAI API key is:", openai_key)
    if openai_key is None:
        raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.\
            check with 'echo $OPENAI_API_KEY'")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
        print("OpenAI API key set successfully.")
    return openai_key
        
# Load environment variables
openai_key = load_openai_key()

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation")

# File paths for FAISS index and CSV file
FAISS_INDEX_PATH = "faiss_index"
CSV_FILE_PATH = "/kapil_wanaskar/295A/Grocery_and_Gourmet_Food_filtered_1000.csv"

# Load the documents and FAISS index only once using Streamlit's caching
@st.cache_resource
def load_data(openai_key=openai_key):
    with get_openai_callback() as cb:

        # Check if the FAISS index already exists locally
        if os.path.exists(FAISS_INDEX_PATH):
            # Load the existing FAISS index
            faiss_index = FAISS.load_local(
                FAISS_INDEX_PATH, 
                OpenAIEmbeddings(api_key=openai_key),
                allow_dangerous_deserialization=True
                )

            st.write("FAISS index loaded from local storage.")
            st.write(f"Cost of loading existing index: ${cb.total_cost:.6f}")
        else:
            # Create embeddings using OpenAI API and store them in a FAISS index
            st.write("Creating a new FAISS index. This might take a while...")
            
            # Load documents from the CSV file
            loader = CSVLoader(file_path=CSV_FILE_PATH)
            docs = loader.load()

            # Use OpenAI Embeddings
            embeddings_model = OpenAIEmbeddings(api_key=openai_key)

            # Create a FAISS index
            faiss_index = FAISS.from_documents(docs, embeddings_model)

            # Save the FAISS index locally for future use
            faiss_index.save_local(FAISS_INDEX_PATH)
            st.write("FAISS index created and saved locally.")
            st.write(f"Cost of creating new index: ${cb.total_cost:.6f}")
            
        st.write(f"Total tokens: {cb.total_tokens}")
        st.write(f"Prompt tokens: {cb.prompt_tokens}")
        st.write(f"Completion tokens: {cb.completion_tokens}")  

    return faiss_index

# Load the document search
docsearch = load_data(openai_key=openai_key)

# Create a question-answering chain using the index
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=openai_key),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    input_key="question"
)

# User input
input_dish = st.text_input("Enter the dish you want to cook:",
                           placeholder="Enter a dish here...")

user_query = f"From the column named 'title' in the dataset, \
recommend me 5 items to cook [not ready made] {input_dish}. \
Do not recommend similar items of the same category. \
e.g., do not recommend 2 types of rice as the same ingredient."

# When the user presses the 'Recommend Shopping List' button
if st.button("Recommend Shopping List"):
    with get_openai_callback() as cb:
        response = chain({"question": user_query})
        if 'result' in response:
            st.write("Recommended Shopping List:")
            st.write(response['result'])
            
            # Print the cost of the API call
            st.write(f"Total Tokens: {cb.total_tokens}")
            st.write(f"Prompt Tokens: {cb.prompt_tokens}")
            st.write(f"Completion Tokens: {cb.completion_tokens}")
            st.write(f"Total Cost (USD): ${cb.total_cost:.6f}")
        else:
            st.write("No recommendations found")
