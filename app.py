# Install required libraries (if not installed, uncomment and run)
# !pip install transformers torch streamlit

import streamlit as st
from transformers import pipeline

# Load an open-source, public model from Hugging Face (No authentication required)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="google/gemma-2b-it", torch_dtype="auto", device_map="auto")

# Load the model
model = load_model()

# Streamlit UI
st.title("💬 Local AI Chatbot (Hugging Face Transformers)")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(chat)

# User input
user_query = st.chat_input("Ask anything...")

if user_query:
    with st.spinner("Thinking..."):
        response = model(user_query, max_length=150, do_sample=True)[0]["generated_text"]

    # Store chat history
    st.session_state.chat_history.append(f"**You:** {user_query}")
    st.session_state.chat_history.append(f"**AI:** {response}")

    # Display response
    st.markdown(f"**AI:** {response}")
