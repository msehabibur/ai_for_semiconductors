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

# Define element groups
A_elements_A2BCX4 = ["Na", "K", "Rb", "Cs", "Ag", "Cu"]
B_elements_A2BCX4 = ["Mg", "Ca", "Ba", "Sr", "Cd", "Zn"]
C_elements_A2BCX4 = ["Sn", "Ge", "Zr"]
X_elements_A2BCX4 = ["S", "Se", "Te"]

A_elements_ABX2 = ["Na", "K", "Rb", "Cs", "Cu", "Ag"]
B_elements_ABX2 = ["Al", "Ga", "In"]
X_elements_ABX2 = ["S", "Se", "Te"]

# ---------------------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------------------
def classify_type(elements):
    """Decide whether the compound is A2BCX4 or ABX2 based on presence of ABX2 B-site elements."""
    for element in elements:
        if element in B_elements_ABX2:
            return "ABX2"
    return "A2BCX4"

def parse_elements(formula_str):
    """Extract each element and its quantity from a chemical formula string."""
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    found = pattern.findall(formula_str)
    return {el: float(qty) if qty else 1.0 for el, qty in found}

def normalize_composition(elements, compound_type):
    """
    Normalize composition by dividing all amounts by:
    - 8 if it's A2BCX4
    - 16 if it's ABX2
    """
    scale_factor = 8 if compound_type == "A2BCX4" else 16
    return {el: round(qty / scale_factor, 2) for el, qty in elements.items()}

def format_normalized_composition(elements, compound_type):
    """Return a correctly ordered formula string."""
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

    # Sort elements alphabetically within each site
    sorted_A = sorted(A_site.items())
    sorted_B = sorted(B_site.items())
    sorted_C = sorted(C_site.items())
    sorted_X = sorted(X_site.items())

    # Build formula string in correct order
    formula_parts = []
    for group in [sorted_A, sorted_B, sorted_C if compound_type == "A2BCX4" else [], sorted_X]:
        for (el, amt) in group:
            formula_parts.append(f"{el}{int(amt) if amt.is_integer() else amt}")

    return "".join(formula_parts)

def get_dft_bandgap(normalized_composition, selected_phase):
    """
    Look up the DFT bandgap from data_df (if present) by matching composition + phase.
    Returns a string like '1.23' or 'Not Available'.
    """
    norm_dict = parse_elements(normalized_composition)
    for _, row in data_df.iterrows():
        csv_formula = row["Semiconductors"].replace("_kesterite", "").replace("_stannite", "")
        csv_dict = parse_elements(csv_formula)
        # Compare composition dicts + phase
        if csv_dict == norm_dict and row["Semiconductors"].endswith(selected_phase):
            return f"{row['gap']:.2f}"
    return "Not Available"

# ---------------------------------------------------------------------
# 3. Main Streamlit App
# ---------------------------------------------------------------------
def run_composition_model():
    st.header("📊 Composition-Based ML Model for Predicting Properties of any A₂BCX₄ and ABX₂ Semiconductors")

    # Upload a CIF file
    uploaded_file = st.file_uploader("Upload a CIF Structure File", type=["cif"])
    
    if uploaded_file:
        selected_phase = st.selectbox("Select the phase:", ["kesterite", "stannite"])
        
        # Save the uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            temp_filepath = tmpfile.name

        # Load structure & parse formula
        structure = Structure.from_file(temp_filepath)
        raw_formula = structure.composition.formula

        # Generate site-based descriptors & normalized composition
        descriptors, normalized_formula = extract_elemental_properties(raw_formula)

        # Add phase info
        descriptors["kesterite_phase"] = 1 if selected_phase == "kesterite" else 0
        descriptors["stannite_phase"] = 1 if selected_phase == "stannite" else 0

        # Align descriptor DataFrame with model features
        descriptor_df = pd.DataFrame([descriptors])
        descriptor_df = descriptor_df.reindex(columns=rf_model.feature_names_in_, fill_value=0)

        # Predict using RF model
        prediction = rf_model.predict(descriptor_df)[0]

        # Look up DFT bandgap if available
        dft_gap = get_dft_bandgap(normalized_formula, selected_phase)

        # Remove temp file
        os.remove(temp_filepath)

        # -----------------------------------
        # Display Results
        # -----------------------------------
        # 1) Composition
        st.subheader("Extracted Composition")
        st.write(f"**Raw Formula:** {raw_formula}")
        st.write(f"**Normalized Composition (Ordered):** `{normalized_formula}`")

        # 2) Visualize Crystal Structure
        st.subheader("Crystal Structure Visualization")
        view = py3Dmol.view(width=600, height=600)
        view.addModel(structure.to(fmt="cif"), "cif")
        view.setStyle({"stick": {}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=600)

        # 3) Generated Descriptors
        st.subheader("Generated Descriptors")
        st.dataframe(descriptor_df)

        # 4) Bandgap Prediction
        st.subheader("Predicted Bandgap")
        results_df = pd.DataFrame({
            "Compound": [normalized_formula],
            "Phase": [selected_phase],
            "DFT Bandgap (HSE+SOC)": [dft_gap],
            "RF Predicted Bandgap (HSE+SOC)": [f"{prediction:.2f}"]
        })
        st.dataframe(results_df.style.hide(axis="index"))

# Run the Streamlit App
if __name__ == "__main__":
    run_composition_model()
