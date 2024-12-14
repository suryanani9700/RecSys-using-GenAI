import streamlit as st
import langchain_community
from langchain_community.document_loaders import CSVLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation")

# Load the documents only once using Streamlit's caching
@st.cache_resource
def load_data():
    # Load the documents from the CSV file
    loader = CSVLoader(file_path="/kapil_wanaskar/295A/Grocery_and_Gourmet_Food_filtered_1000.csv")

    # Build embeddings model using Ollama
    embeddings_model = OllamaEmbeddings(model="llama3.1")

    # Create an index using the loaded documents with the local embeddings model
    index_creator = VectorstoreIndexCreator(embedding=embeddings_model, vectorstore_cls=FAISS)
    docsearch = index_creator.from_loaders([loader])
    
    return docsearch

docsearch = load_data()

# Create a question-answering chain using the index and local Llama3.1 model
llm = OllamaLLM(model="llama3.1")
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.vectorstore.as_retriever(),
    input_key="question"
)

# User input
input_dish = st.text_input("Enter the dish you want to cook:",
        placeholder="Enter a dish here...")

user_query = f"from the column named 'title' in the dataset,\
                recommend me 5 items to cook [not ready made] {input_dish}.\
                Do not recommend similar items of the same category.\
                e.g., do not recommend 2 types of rice as the same ingredient."

# When the user presses the 'Recommend Shopping List' button
if st.button("Recommend Shopping List"):
    response = chain({"question": user_query})
    if 'result' in response:
        st.write("Recommended Shopping List:")
        st.write(response['result'])
    else:
        st.write("No recommendations found")