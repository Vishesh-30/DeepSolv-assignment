import streamlit as st
from sentence_transformers import SentenceTransformer
from src.vector_store import VectorStore
import numpy as np
import requests
from dotenv import load_dotenv
import os
import google.generativeai as genai
from streamlit_chat import message
import random

# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Load the FAISS index and embeddings
dimension = 512
vector_store = VectorStore(dimension)
vector_store.load_index("faiss_index.bin")

# Load the SentenceTransformer model for query embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
encoder = SentenceTransformer(model_name)

# Load the text chunks (for context retrieval)
text_chunks = np.load("text_chunks.npy", allow_pickle=True)

# Function to interact with Gemini API
def generate_response(prompt):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.title("Q/A Chatbot")
st.write("Ask questions about the Apple Vision Pro:")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("Your question", key="input")

if user_input:
    query_embedding = encoder.encode([user_input])
    D, I = vector_store.search(query_embedding, k=5)
    # print(D, I)

    # Retrieve top result content for context
    top_result_content = " ".join([text_chunks[idx] for idx in I[0]])

    # Generate prompt with context
    prompt = f"Context: {top_result_content}\n Note: Answer the questions as in you are a sales Person and don't make it look like you are just summerizing from a vector embedding. \nQuestion: {user_input}\nAnswer:"
    print(prompt)

    # Get response from Gemini API
    response = generate_response(prompt)

    # Store user input and response in session state
    st.session_state.messages.append({"message": response, "is_user": False})
    st.session_state.messages.append({"message": user_input, "is_user": True})
    

    # # Clear the input box after submitting
    # st.session_state.input = ""

# Display the conversation history
for message_data in reversed(st.session_state.messages):
    message(message_data['message'], is_user=message_data['is_user'], key=random.randint(0, 1000000))
