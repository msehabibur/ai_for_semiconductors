import streamlit as st
from composition_ml import run_composition_model
from graph_based_mlff import run_graph_based_mlff

# Main Title
st.title("🚀 AI for Accelerating Discovery of A₂BCX₄ and ABX₂ Semiconductors for Thin-Film Solar Cells")

# Single-line developer info with names, emails, and affiliation
st.markdown("""
**Developed by**: Md Habibur Rahman (rahma103@purdue.edu), Arun Mannodi-Kanakkithodi (amannodi@purdue.edu), School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA
""")

# Background Section
st.markdown("""
### 🌞 Background: Challenges in Thin-Film Solar Cells
Thin-film solar cells based on **A₂BCX₄ (e.g., kesterite)** and **ABX₂ (e.g., chalcopyrite)** compounds have gained attention as **low-cost, earth-abundant alternatives** to silicon photovoltaics (PVs). However, their **efficiency remains below 15%**, significantly lower than **Si (>26%)** and **perovskites (>25%)**.

🔬 **Challenges include:**
- **Defect-driven efficiency losses** due to deep-level defects.
- **Inefficient charge transport** caused by cation disorder.
- **Narrow chemical space exploration** limiting material breakthroughs.

💡 **AI-Powered Material Discovery**
- **Machine learning (ML)** models can efficiently screen vast chemical spaces to **predict stable, defect-tolerant materials**.
- This platform integrates **composition-based ML predictions** and **graph-based MLFF optimization** to **accelerate the discovery of novel semiconductors** for next-generation thin-film solar cells.
""")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Composition-Based ML Model", "📈 Graph-Based MLFF Optimization"])

with tab1:
    # Runs the composition-based ML model defined in composition_ml.py
    run_composition_model()

with tab2:
    # Runs the graph-based MLFF optimizer defined in graph_based_mlff.py
    run_graph_based_mlff()
