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
    if openai_key is None:
        raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key'")
    os.environ["OPENAI_API_KEY"] = openai_key
    return openai_key

def get_main_ingredients(dish_name):
    """Get only the 3 main ingredients for a given dish using OpenAI."""
    llm = OpenAI()
    prompt = f"List only the 3 most essential ingredients needed for {dish_name}, return only the ingredient names separated by commas, no quantities"
    response = llm(prompt)
    ingredients = [ing.strip() for ing in response.split(',')]
    return ingredients[:3]  # Ensure only 3 ingredients are returned

def get_product_recommendations(ingredient, docsearch):
    """Get unique product recommendations from the title column for a specific ingredient."""
    query = f"Find products in the title column that contain or are related to {ingredient}"
    results = docsearch.similarity_search(query, k=4)  # Get more results initially
    
    # Create a list of unique recommendations
    unique_recommendations = []
    seen = set()
    
    for doc in results:
        if hasattr(doc, 'page_content'):
            title = doc.page_content.strip()
            if title not in seen:
                unique_recommendations.append(title)
                seen.add(title)
                if len(unique_recommendations) == 2:  # Stop after finding 2 unique items
                    break
    
    return unique_recommendations

# Initialize Streamlit app
st.title("Amazon Shopping Recommendation")

# Load environment variables
openai_key = load_openai_key()

# File paths
FAISS_INDEX_PATH = "faiss_index"
CSV_FILE_PATH = "raw_titles.csv"

@st.cache_resource
def load_data(openai_key):
    with get_openai_callback() as cb:
        if os.path.exists(FAISS_INDEX_PATH):
            faiss_index = FAISS.load_local(
                FAISS_INDEX_PATH,
                OpenAIEmbeddings(api_key=openai_key),
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
            embeddings_model = OpenAIEmbeddings(api_key=openai_key)
            faiss_index = FAISS.from_documents(docs, embeddings_model)
            faiss_index.save_local(FAISS_INDEX_PATH)
    return faiss_index

# Load the document search
docsearch = load_data(openai_key=openai_key)

# User input
input_dish = st.text_input("Enter the dish you want to cook:",
                          placeholder="Enter a dish here...")

if st.button("Recommend Shopping List"):
    with get_openai_callback() as cb:
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
        
        # Display API usage statistics
        st.write("\nAPI Usage Statistics:")
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost:.6f}")