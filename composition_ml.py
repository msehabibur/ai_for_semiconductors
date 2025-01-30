import streamlit as st
from pymatgen.core import Structure
import pandas as pd
import joblib
import tempfile
import os
import py3Dmol
import re

# ---------------------------------------------------------------------
# 1. Load Model and Data
# ---------------------------------------------------------------------
rf_model = joblib.load("random_forest_model_filtered.pkl")
elemental_properties = pd.read_excel("Elemental_properties.xlsx", sheet_name="Sheet1")
data_df = pd.read_csv("data.csv")

# Define element groups (Alphabetically ordered for consistency)
A_elements_A2BCX4 = ["Ag", "Cs", "Cu", "K", "Na", "Rb"]
B_elements_A2BCX4 = ["Ba", "Ca", "Cd", "Mg", "Sr", "Zn"]
C_elements_A2BCX4 = ["Ge", "Sn", "Zr"]
X_elements_A2BCX4 = ["S", "Se", "Te"]

A_elements_ABX2 = ["Ag", "Cs", "Cu", "K", "Na", "Rb"]
B_elements_ABX2 = ["Al", "Ga", "In"]
X_elements_ABX2 = ["S", "Se", "Te"]

# ---------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------
def classify_type(elements):
    """Determine whether a compound is A₂BCX₄ or ABX₂ based on its elements."""
    for element in elements:
        if element in B_elements_ABX2:
            return "ABX2"
    return "A2BCX4"

def parse_elements(formula_str):
    """Extract elements and quantities from a formula string."""
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    found = pattern.findall(formula_str)
    return {el: float(qty) if qty else 1.0 for el, qty in found}

def normalize_composition(elements, compound_type):
    """Normalize compositions based on the compound type."""
    scale_factor = 8 if compound_type == "A2BCX4" else 16
    return {el: round(qty / scale_factor, 2) for el, qty in elements.items()}

def format_normalized_composition(elements, compound_type):
    """Return a correctly ordered formula string for A2BCX4 or ABX2."""
    
    # Sorting elements into respective sites
    A_site, B_site, C_site, X_site = {}, {}, {}, {}
    
    for el, amt in elements.items():
        if el in A_elements_A2BCX4:
            A_site[el] = amt
        elif el in B_elements_A2BCX4:
            B_site[el] = amt
        elif el in C_elements_A2BCX4:
            C_site[el] = amt
        elif el in X_elements_A2BCX4:
            X_site[el] = amt

    # Sort elements within each site alphabetically
    sorted_A = sorted(A_site.items())
    sorted_B = sorted(B_site.items())
    sorted_C = sorted(C_site.items())
    sorted_X = sorted(X_site.items())

    # Build formula string
    formula_parts = []
    for group in [sorted_A, sorted_B, sorted_C if compound_type == "A2BCX4" else [], sorted_X]:
        for (el, amt) in group:
            formula_parts.append(f"{el}{int(amt) if amt.is_integer() else amt}")

    return "".join(formula_parts)

# ---------------------------------------------------------------------
# 3. Main Streamlit App
# ---------------------------------------------------------------------
def run_composition_model():
    st.header("📊 ML Model for Predicting Properties of A₂BCX₄ & ABX₂ Semiconductors")

    # Upload CIF file
    uploaded_file = st.file_uploader("Upload a CIF Structure File", type=["cif"])
    
    if uploaded_file:
        selected_phase = st.selectbox("Select the phase:", ["kesterite", "stannite"])
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            temp_filepath = tmpfile.name

        # Load structure & parse formula
        structure = Structure.from_file(temp_filepath)
        raw_formula = structure.composition.formula

        # Process the composition
        parsed_elements = parse_elements(raw_formula)
        compound_type = classify_type(parsed_elements)
        normalized_composition = normalize_composition(parsed_elements, compound_type)
        formatted_formula = format_normalized_composition(normalized_composition, compound_type)

        # Clean up temporary file
        os.remove(temp_filepath)

        # -----------------------------------
        # Display Results
        # -----------------------------------
        st.subheader("Extracted Composition")
        st.write(f"**Raw Formula:** `{raw_formula}`")
        st.write(f"**Normalized Composition (Ordered):** `{formatted_formula}`")

        # **Crystal Structure Visualization**
        st.subheader("Crystal Structure Visualization")
        view = py3Dmol.view(width=600, height=600)
        view.addModel(structure.to(fmt="cif"), "cif")
        view.setStyle({"stick": {}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=600)

        # **Predict Bandgap**
        st.subheader("Predicted Bandgap")
        descriptor_df = pd.DataFrame([normalized_composition]).reindex(columns=rf_model.feature_names_in_, fill_value=0)
        prediction = rf_model.predict(descriptor_df)[0]
        
        results_df = pd.DataFrame({
            "Compound": [formatted_formula],
            "Phase": [selected_phase],
            "RF Predicted Bandgap (HSE+SOC)": [f"{prediction:.2f} eV"]
        })
        st.dataframe(results_df.style.hide(axis="index"))

# Run the Streamlit App
if __name__ == "__main__":
    run_composition_model()
