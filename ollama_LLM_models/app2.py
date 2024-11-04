import streamlit as st
import subprocess

# Function to run Llama 3.1 using ollama and display the result in real time
def run_llama_real_time(query):
    # Create a placeholder that we will update with each line
    output_placeholder = st.empty()

    # Run the Llama model using subprocess and read output line by line
    process = subprocess.Popen(['ollama', 'run', 'llama3.1', query], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream the output to Streamlit
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            # As each line is generated, append it to output_lines
            output_lines.append(line.strip())

            # Join all lines and update the placeholder to show them
            output_placeholder.text('\n'.join(output_lines))

# Main function to run the Streamlit app
def main():
    # Streamlit app title
    st.title("Llama 3.1 Query App")

    # Text input from the user
    user_query = st.text_input("Enter your query:")

    # When the user enters a query
    if user_query and st.button("Run Llama"):
        st.write("Running Llama 3.1...")
        run_llama_real_time(user_query)

# Call the main function
if __name__ == "__main__":
    main()
