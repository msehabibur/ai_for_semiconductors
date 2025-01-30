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
    """
    Return a formula string in the correct site order:
      - For A2BCX4: A -> B -> C -> X
      - For ABX2:   A -> B -> X
    """
    if compound_type == "A2BCX4":
        order = A_elements_A2BCX4 + B_elements_A2BCX4 + C_elements_A2BCX4 + X_elements_A2BCX4
    else:  # ABX2
        order = A_elements_ABX2 + B_elements_ABX2 + X_elements_ABX2

    parts = []
    for el in order:
        if el in elements and elements[el] > 0:
            amt = elements[el]
            # If amt is an integer, strip decimal; otherwise include decimal
            if amt.is_integer():
                parts.append(f"{el}{int(amt)}")
            else:
                parts.append(f"{el}{amt}")
    return "".join(parts)

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

def extract_elemental_properties(semiconductor):
    """
    1. Parse formula, classify type (A2BCX4 or ABX2), normalize composition.
    2. Group each element into the correct site (A, B, C, X).
       - For ABX2, duplicate B->C to keep consistent site dimension.
    3. Compute weighted-average elemental properties for each site.
    4. Return (descriptor_dict, normalized_formula_str).
    """
    # Parse and classify
    parsed = parse_elements(semiconductor)
    ctype = classify_type(parsed)
    normalized_elems = normalize_composition(parsed, ctype)

    # Group elements into A, B, C, X
    site_elements = {"A": [], "B": [], "C": [], "X": []}
    if ctype == "A2BCX4":
        for el, amt in normalized_elems.items():
            if el in A_elements_A2BCX4:
                site_elements["A"].append((el, amt))
            elif el in B_elements_A2BCX4:
                site_elements["B"].append((el, amt))
            elif el in C_elements_A2BCX4:
                site_elements["C"].append((el, amt))
            elif el in X_elements_A2BCX4:
                site_elements["X"].append((el, amt))
    else:  # ABX2
        for el, amt in normalized_elems.items():
            if el in A_elements_ABX2:
                site_elements["A"].append((el, amt))
            elif el in B_elements_ABX2:
                site_elements["B"].append((el, amt))
            elif el in X_elements_ABX2:
                site_elements["X"].append((el, amt))
        # Duplicate B to C for ABX2
        site_elements["C"] = site_elements["B"]

    # Initialize all site-property keys to 0
    props = {f"{site}_{col}": 0
             for site in ["A", "B", "C", "X"]
             for col in elemental_properties.columns[1:]}

    # Weighted average for each site
    for site, elem_list in site_elements.items():
        total_amt = sum(qty for _, qty in elem_list)
        if total_amt == 0:
            continue

        # Sum up property * quantity
        for (element, qty) in elem_list:
            row = elemental_properties[elemental_properties["M"] == element]
            if not row.empty:
                row_data = row.iloc[0].to_dict()
                for key, val in row_data.items():
                    if key == "M":
                        continue
                    site_key = f"{site}_{key}"
                    props[site_key] += val * qty

        # Divide by total_amt to get average
        for k in list(props.keys()):
            if k.startswith(site + "_"):
                props[k] /= total_amt

    # Format the normalized composition using the new ordering function
    norm_str = format_normalized_composition(normalized_elems, ctype)
    return props, norm_str

# ---------------------------------------------------------------------
# 3. Main Streamlit App
# ---------------------------------------------------------------------
def run_composition_model():
    st.header("📊 Composition-Based ML Model for Predicting Properties of any A₂BCX₄ and ABX₂ Semiconductors")

    st.markdown("""
    📌 **Note:**  
    - The model currently handles 2×2×1 supercells.  
    - Mixing at multiple sites is supported at 1/4 fraction.  
    """)

    st.markdown("""
    **This MLFF model is trained on a chemical space that includes:**
    - **A₂BCX₄-type**:  
      - A-site: `Ag, Cs, Cu, K, Na, Rb`  
      - B-site: `Ba, Ca, Cd, Mg, Sr, Zn`  
      - C-site: `Ge, Sn, Zr`  
      - X-site: `S, Se, Te`  

    - **ABX₂-type**:  
      - A-site: `Ag, Cs, Cu, K, Na, Rb`  
      - B-site: `Al, Ga, In`  
      - X-site: `S, Se, Te`
    """)

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
        st.subheader("🧪 Extracted Composition")

        raw_formula_display = f'<p style="color:green; background-color:lightgrey; padding:5px; border-radius:5px; font-size:16px;">Raw Formula: {raw_formula}</p>'
        normalized_formula_display = f'<p style="color:green; background-color:lightgrey; padding:5px; border-radius:5px; font-size:16px;">Normalized Composition: {normalized_formula}</p>'
        st.markdown(raw_formula_display, unsafe_allow_html=True)
        st.markdown(normalized_formula_display, unsafe_allow_html=True)

        #st.write(f"**Raw Formula:** {raw_formula}")
        # The normalized formula will now appear in the correct A→B→C→X or A→B→X order
        #st.write(f"**Normalized Composition:** {normalized_formula}")

        st.subheader("Crystal Structure Visualization")
        view = py3Dmol.view(width=600, height=600)
        view.addModel(structure.to(fmt="cif"), "cif")
        view.setStyle({"stick": {}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=600)

        st.subheader("Generated Descriptors")
        st.dataframe(descriptor_df)

        st.subheader("Predicted Bandgap")
        results_df = pd.DataFrame({
            "Compound": [normalized_formula],
            "Phase": [selected_phase],
            "DFT Bandgap (HSE+SOC)": [dft_gap],
            "ML Predicted Bandgap (RF)": [f"{prediction:.2f}"]
        })
        st.dataframe(results_df.style.hide(axis="index"))


# Run the Streamlit app
if __name__ == "__main__":
    run_composition_model()
