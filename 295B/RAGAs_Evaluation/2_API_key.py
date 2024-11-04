import os

# Get the OpenAI API key from the environment
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OpenAI API key is missing. Please set it using 'export OPENAI_API_KEY=your-openai-key' in the terminal.")
os.environ["OPENAI_API_KEY"] = openai_key

print("OpenAI API key set successfully.")