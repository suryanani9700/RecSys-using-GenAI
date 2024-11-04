import streamlit as st
import subprocess

# Function to run Llama 3.1 using ollama
def run_llama(query):
    # Run the ollama command using subprocess and pass the query as an argument
    result = subprocess.run(['ollama', 'run', 'llama3.1', query], capture_output=True, text=True)
    return result.stdout

# Main function to run the streamlit app
def main():
    # Streamlit app title
    st.title("Llama 3.1 Query App")

    # Text input from the user
    user_query = st.text_input("Enter your query:")

    # When the user enters a query
    if user_query:
        st.write("Running Llama 3.1...")
        response = run_llama(user_query)
        
        # Display the result
        st.subheader("Llama 3.1 Response:")
        # Streamlit markdown can handle multiline responses
        st.markdown(f"```\n{response}\n```")

# Call the main function
if __name__ == "__main__":
    main()
