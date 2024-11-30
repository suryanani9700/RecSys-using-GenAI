from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
import streamlit as st

def load_llama_model():
    """Load the Llama 3.1 8B model using Ollama."""
    return Ollama(model="phi3.5")

def get_main_ingredients(dish_name):
    """Get only the 3 main ingredients for a given dish using Llama."""
    llm = load_llama_model()
    prompt = f"List only the 3 most essential ingredients needed for {dish_name}, return only the ingredient names separated by commas, no quantities"
    response = llm.invoke(prompt)
    ingredients = [ing.strip() for ing in response.split(',')]
    return ingredients[:3]  # Ensure only 3 ingredients are returned

def get_product_recommendations(ingredient, docsearch):
    """Get unique product recommendations from the title column for a specific ingredient."""
    query = f"Find products in the title column that contain or are related to {ingredient}"
    results = docsearch.similarity_search(query, k=4)
    
    unique_recommendations = []
    seen = set()
    
    for doc in results:
        if hasattr(doc, 'page_content'):
            title = doc.page_content.strip()
            if title not in seen:
                unique_recommendations.append(title)
                seen.add(title)
                if len(unique_recommendations) == 2:
                    break
    
    return unique_recommendations

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation with phi 3.5")

# File paths
FAISS_INDEX_PATH = "faiss_index"
CSV_FILE_PATH = "raw_titles.csv"

@st.cache_resource
def load_or_create_faiss_index():
    """Load the FAISS index if it exists, otherwise create and save it."""
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = FAISS.load_local(
            FAISS_INDEX_PATH,
            OllamaEmbeddings(model="phi3.5"),  # Removed batch_size here
            allow_dangerous_deserialization=True
        )
    else:
        loader = CSVLoader(
            file_path=CSV_FILE_PATH,
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': ['title']
            }
        )
        docs = loader.load()
        embeddings_model = OllamaEmbeddings(model="phi3.5")  # Removed batch_size here
        faiss_index = FAISS.from_documents(docs, embeddings_model)
        faiss_index.save_local(FAISS_INDEX_PATH)  # Save the new index
    return faiss_index


# Load the document search
with st.spinner('Loading data...'):
    docsearch = load_or_create_faiss_index()

# User input
input_dish = st.text_input("Enter the dish you want to cook:",
                          placeholder="Enter a dish here...")

if st.button("Recommend Shopping List"):
    # Get only 3 main ingredients
    main_ingredients = get_main_ingredients(input_dish)
    
    # Display main ingredients
    st.write("1. Main Ingredients:")
    for ingredient in main_ingredients:
        st.write(f"• {ingredient}")
    
    # Get and display recommendations for main ingredients
    st.write("\n2. Items for Each Main Ingredient:")
    for ingredient in main_ingredients:
        recommendations = get_product_recommendations(ingredient, docsearch)
        if recommendations:
            st.write(f"• {ingredient}:")
            for i, item in enumerate(recommendations, 1):
                st.write(f"    {i}. {item}")