import os
import pinecone
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

# Refactored function to load API keys
def load_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            api_key = file.read().strip()
            if not api_key:
                raise ValueError(f"API key in {file_path} is not set or empty.")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file {file_path} not found.")

# Load Pinecone API key from file
PINECONE_API_KEY = load_api_key("API-Keys/pinecones.txt")

# Create Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load Groq API key from file
GROQ_API_KEY = load_api_key("API-Keys/groq.txt")
client = Groq(api_key=GROQ_API_KEY)

# Ensure Pinecone index exists
index_name = "testing"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(name=index_name)

def vectorize_text(text):
    # Convert each character to its ASCII value and then to float
    vector = [float(ord(char)) for char in text]
    # Ensure the vector is exactly 1024 dimensions
    vector = vector[:1024]  # Truncate if longer
    if len(vector) < 1024:
        vector += [0.0] * (1024 - len(vector))  # Pad with zeros if shorter
    return vector

def store_response(response_id, response_text):
    try:
        vector = vectorize_text(response_text)
        index.upsert(vectors=[(response_id, vector)])
    except Exception as e:
        print(f"An error occurred while storing the response: {e}")

try:
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model="mixtral-8x7b-32768",
        )

        response_message = chat_completion.choices[0].message.content
        print("Verza:", response_message)
        store_response(response_id=user_input, response_text=response_message)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Add any necessary cleanup code here
    print("Exiting gracefully.")

