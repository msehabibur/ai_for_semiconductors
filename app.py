import streamlit as st
from transformers import pipeline

# Load Hugging Face model for text generation
@st.cache_resource  # Cache model for faster loads
def load_model():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype="auto", device_map="auto")

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
