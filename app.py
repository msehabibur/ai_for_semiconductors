import streamlit as st
from composition_ml import run_composition_model
from graph_based_mlff import run_graph_based_mlff
from gpt4all import GPT4All  # Local AI model, No API needed

# Load GPT4All model (Ensure you have downloaded a compatible model first)
model_path = "path_to_your_model.gguf"  # Replace with your local model path
gpt_model = GPT4All(model_path)

# Main Title
st.title("🚀 AI for Accelerating Discovery of A₂BCX₄ and ABX₂ Semiconductors for Thin-Film Solar Cells")

# Developer Information
st.markdown("""
**Developed by**:  
Md Habibur Rahman (rahma103@purdue.edu), Arun Mannodi-Kanakkithodi (amannodi@purdue.edu)  
School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA
""")

# Background Information
st.markdown("""
### 🌞 Background: Challenges in Thin-Film Solar Cells
Thin-film solar cells based on **A₂BCX₄ (e.g., kesterite)** and **ABX₂ (e.g., chalcopyrite)** compounds are gaining attention as **low-cost, earth-abundant alternatives** to silicon photovoltaics (PVs). However, efficiency remains below **15%**, significantly lower than **Si (>26%)** and **perovskites (>25%)**.

💡 **AI-Powered Material Discovery**
- **Machine learning (ML) models** predict stable, defect-tolerant materials.
- This platform integrates **composition-based ML predictions** and **graph-based MLFF optimization** to **accelerate the discovery of novel semiconductors**.
""")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📊 Composition-Based ML Model", "📈 Graph-Based MLFF Optimization", "💬 AI Chatbot"])

with tab1:
    run_composition_model()

with tab2:
    run_graph_based_mlff()

with tab3:
    st.subheader("💬 AI Chatbot for Materials Science")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(chat)

    # User Input
    user_query = st.chat_input("Ask me anything about materials science!")

    if user_query:
        with st.spinner("Thinking..."):
            response = gpt_model.generate(user_query)  # Use local GPT4All model

        # Store chat history
        st.session_state.chat_history.append(f"**You:** {user_query}")
        st.session_state.chat_history.append(f"**AI:** {response}")

        # Display response
        st.markdown(f"**AI:** {response}")
