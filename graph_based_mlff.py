import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from matgl import load_model
from matgl.ext.ase import Relaxer
import re
import py3Dmol

# Load trained MLFF model
trained_mlff = load_model(path="/Users/habibur/Downloads/streamlit/")

# Reordered element groups (alphabetical)
A_elements_A2BCX4 = ["Ag", "Cs", "Cu", "K", "Na", "Rb"]
B_elements_A2BCX4 = ["Ba", "Ca", "Cd", "Mg", "Sr", "Zn"]
C_elements_A2BCX4 = ["Ge", "Sn", "Zr"]
X_elements_A2BCX4 = ["S", "Se", "Te"]

A_elements_ABX2 = ["Ag", "Cs", "Cu", "K", "Na", "Rb"]
B_elements_ABX2 = ["Al", "Ga", "In"]
X_elements_ABX2 = ["S", "Se", "Te"]

# Function to classify compound type
def classify_type(elements):
    """
    If at least one element is in the B_elements_ABX2 list, we assume ABX2 type.
    Otherwise, A2BCX4 is assumed by default.
    """
    for element in elements.keys():
        if element in B_elements_ABX2:
            return "ABX2"
    return "A2BCX4"

# Function to parse elements from formula
def parse_elements(semiconductor):
    """
    Parses a formula string (e.g., 'Ag2Ca0.5Ge0.5S2Sn0.5Sr0.5Te2')
    into a dict {element: stoichiometric_quantity}.
    """
    element_pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    elements = element_pattern.findall(semiconductor)
    return {el: float(qty) if qty else 1.0 for el, qty in elements}

# Function to normalize composition
def normalize_composition(elements, compound_type):
    """
    For A2BCX4, the supercell size is 2×2×1 = 8 formula units.
    For ABX2, the supercell size is 2×2×1 = 4 formula units, but typically 
    we treat it as 16 for the internal normalization. 
    Adjust if your model's scaling differs.
    """
    scale_factor = 8 if compound_type == "A2BCX4" else 16
    return {el: round(qty / scale_factor, 2) for el, qty in elements.items()}

# Function to reorder and format the composition by site
def format_normalized_composition(elements, compound_type):
    """
    Groups elements into A, B, C, X sites (for A2BCX4) or A, B, X (for ABX2),
    sorts them alphabetically within each site, and concatenates in the order
    A → B → C → X (A2BCX4) or A → B → X (ABX2).
    """
    
    # For A2BCX4
    if compound_type == "A2BCX4":
        A_site = {}
        B_site = {}
        C_site = {}
        X_site = {}
        
        # Assign elements to their respective sites
        for el, amt in elements.items():
            if el in A_elements_A2BCX4:
                A_site[el] = amt
            elif el in B_elements_A2BCX4:
                B_site[el] = amt
            elif el in C_elements_A2BCX4:
                C_site[el] = amt
            elif el in X_elements_A2BCX4:
                X_site[el] = amt
            else:
                # If an element doesn't match any known site, just keep it separate
                # (unlikely if your input strictly follows the above sets)
                A_site[el] = amt
        
        # Sort each site by element symbol
        A_sorted = sorted(A_site.items(), key=lambda x: x[0])
        B_sorted = sorted(B_site.items(), key=lambda x: x[0])
        C_sorted = sorted(C_site.items(), key=lambda x: x[0])
        X_sorted = sorted(X_site.items(), key=lambda x: x[0])
        
        # Build the output formula string: A -> B -> C -> X
        formula_parts = []
        for group in [A_sorted, B_sorted, C_sorted, X_sorted]:
            for (el, amt) in group:
                # If amt is an integer (e.g., 2.0), show as 2; else show 2.0
                part = f"{el}{int(amt) if amt.is_integer() else amt}"
                formula_parts.append(part)
        
        return "".join(formula_parts)
    
    # For ABX2
    else:
        A_site = {}
        B_site = {}
        X_site = {}
        
        # Assign elements to their respective sites
        for el, amt in elements.items():
            if el in A_elements_ABX2:
                A_site[el] = amt
            elif el in B_elements_ABX2:
                B_site[el] = amt
            elif el in X_elements_ABX2:
                X_site[el] = amt
            else:
                # If an element doesn't match any known site, default to A
                A_site[el] = amt
        
        # Sort each site by element symbol
        A_sorted = sorted(A_site.items(), key=lambda x: x[0])
        B_sorted = sorted(B_site.items(), key=lambda x: x[0])
        X_sorted = sorted(X_site.items(), key=lambda x: x[0])
        
        # Build the output formula string: A -> B -> X
        formula_parts = []
        for group in [A_sorted, B_sorted, X_sorted]:
            for (el, amt) in group:
                part = f"{el}{int(amt) if amt.is_integer() else amt}"
                formula_parts.append(part)
        
        return "".join(formula_parts)

# Function to visualize CIF structure using py3Dmol
def display_structure_with_py3Dmol(structure):
    cif_string = structure.to(fmt="cif")
    viewer = py3Dmol.view(width=600, height=400)  # Smaller visualization
    viewer.addModel(cif_string, "cif")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer

# Streamlit App
def run_graph_based_mlff():
    st.title("📈 Optimization of any A₂BCX₄ and ABX₂ Semiconductors using Graph Based MLFF ")

    # **User Note**
    st.markdown("""
    📌 **Note:**  
    - Right now, the model can handle only **2×2×1 supercells**.  
    - Mixing at multiple sites is supported at **1/4 fraction**.  
    """)

    # **Chemical Space Description**
    st.markdown("""
    **This MLFF model is trained on a chemical space that includes:**
    - **A₂BCX₄-type compounds** with elements:  
      - A-site: `Ag, Cs, Cu, K, Na, Rb`  
      - B-site: `Ba, Ca, Cd, Mg, Sr, Zn`  
      - C-site: `Ge, Sn, Zr`  
      - X-site: `S, Se, Te`  
    - **ABX₂-type compounds** with elements:  
      - A-site: `Ag, Cs, Cu, K, Na, Rb`  
      - B-site: `Al, Ga, In`  
      - X-site: `S, Se, Te`  
    """)

    uploaded_opt_file = st.file_uploader("📂 Upload a CIF Structure File for MLFF Optimization", type=["cif"])
    
    if uploaded_opt_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmpfile:
            tmpfile.write(uploaded_opt_file.getvalue())
            temp_filepath = tmpfile.name

        # Load the structure
        input_structure = Structure.from_file(temp_filepath)
        input_lattice_params = tuple(round(p, 2) for p in input_structure.lattice.abc)  # Round to 2 decimals
        raw_formula = input_structure.composition.formula

        # **Normalize Composition**
        elements_dict = parse_elements(raw_formula)
        compound_type = classify_type(elements_dict)
        normalized_elements = normalize_composition(elements_dict, compound_type)
        normalized_formula = format_normalized_composition(normalized_elements, compound_type)

        # **Show Crystal Structure**
        st.subheader("🔬 Uploaded Crystal Structure")
        with st.spinner("Rendering Structure..."):
            viewer = display_structure_with_py3Dmol(input_structure)
            st.components.v1.html(viewer._make_html(), height=400)

        # **Show Extracted Composition**
        st.subheader("🧪 Extracted Composition")
        st.write(f"**Raw Formula:** `{raw_formula}`")
        st.write(f"**Normalized Composition:** `{normalized_formula}`")

        # **User Input: Enable/Disable Cell Relaxation**
        cell_relax = st.radio("⚙️ Enable Cell Relaxation?", ["False", "True"]) == "True"
        relaxer = Relaxer(potential=trained_mlff, relax_cell=cell_relax, optimizer="FIRE")

        try:
            # **Progress Bar**
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔄 Starting MLFF optimization...")

            relax_results = relaxer.relax(input_structure, fmax=0.0001, steps=200)
            final_structure = relax_results["final_structure"]
            relaxed_lattice_params = tuple(round(p, 2) for p in final_structure.lattice.abc)  # Round to 2 decimals
            final_energy = round(float(relax_results["trajectory"].energies[-1]), 2)

            progress_bar.progress(100)
            status_text.text("✅ Optimization complete!")

            # **Plot Energy vs. Optimization Step**
            st.subheader("📉 Energy Optimization Plot")
            fig, ax = plt.subplots(figsize=(4, 3))  # Smaller plot
            ax.plot(
                range(len(relax_results["trajectory"].energies)),
                relax_results["trajectory"].energies,
                marker="o",
                linestyle="-",
                markersize=4
            )
            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("Energy (eV)")
            ax.set_title("Energy vs. Optimization Steps")
            ax.grid(True)
            st.pyplot(fig)

            # **Optimization Results**
            st.subheader("📊 Optimization Results")
            results_df = pd.DataFrame({
                "Raw Composition": [raw_formula],
                "Normalized Composition": [normalized_formula],
                "Input Lattice Parameters (a, b, c)": [input_lattice_params],
                "Relaxed Lattice Parameters (a, b, c)": [relaxed_lattice_params],
                "MLFF Final Energy (eV)": [final_energy]
            })
            st.dataframe(results_df.style.hide(axis="index"))

            # **Download Optimized Structure**
            optimized_vasp_path = f"{normalized_formula}.vasp"
            Poscar(final_structure).write_file(optimized_vasp_path)
            with open(optimized_vasp_path, "rb") as f:
                st.download_button(
                    "📥 Download Optimized Structure (.vasp)",
                    data=f,
                    file_name=os.path.basename(optimized_vasp_path),
                    mime="chemical/x-poscar"
                )

            # Clean up
            os.remove(temp_filepath)

        except Exception as e:
            st.error(f"❌ An error occurred: {traceback.format_exc()}")

# Run the function
if __name__ == "__main__":
    run_graph_based_mlff()
