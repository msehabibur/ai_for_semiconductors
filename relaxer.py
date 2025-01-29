import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from matgl import load_model
from matgl.ext.ase import Relaxer
import traceback

# Define paths
input_folder = "structures_to_optimize"
output_folder = "MLFF_optimized_structures"
dft_folder = "DFT_optimized_structures"
csv_output_file = "optimization_results.csv"

# Load trained model
model_save_path = "/Users/habibur/Downloads/streamlit/"
trained_model = load_model(path=model_save_path)

# Initialize Relaxer
relaxer = Relaxer(potential=trained_model, relax_cell=False, optimizer="FIRE")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize results storage
results = []

# Loop through all .vasp files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".vasp"):
        file_path = os.path.join(input_folder, file_name)
        print(f"Processing structure: {file_name}")

        try:
            # Load input structure
            input_structure = Structure.from_file(file_path)
            input_structure.perturb(0.00)
            input_lattice_params = input_structure.lattice.abc
            print("Input Lattice Parameters:", input_lattice_params)

            # Perform relaxation
            print("Starting relaxation...")
            relax_results = relaxer.relax(input_structure, fmax=0.001, steps=100)
            final_structure = relax_results["final_structure"]

            # Save the relaxed structure
            output_file = os.path.join(output_folder, f"relaxed_{file_name}")
            Poscar(final_structure).write_file(output_file)
            print(f"Relaxed structure saved to {output_file}")

            # Get lattice parameters of relaxed structure
            relaxed_lattice_params = final_structure.lattice.abc

            # Record optimization history (energies only)
            energy_history = relax_results["trajectory"].energies

            # Check for DFT data
            dft_file_path = os.path.join(dft_folder, file_name)
            if os.path.exists(dft_file_path):
                dft_structure = Structure.from_file(dft_file_path)
                dft_lattice_params = dft_structure.lattice.abc
            else:
                dft_lattice_params = "DFT data not available"

            # Add results to the list
            results.append({
                "File Name": file_name,
                "Input Lattice Parameters (a, b, c)": input_lattice_params,
                "Relaxed Lattice Parameters (a, b, c)": relaxed_lattice_params,
                "DFT Lattice Parameters (a, b, c)": dft_lattice_params,
                "Optimization History": energy_history,
                "Final Energy (eV)": energy_history[-1],
            })
            print(f"Added results for {file_name}. Current results length: {len(results)}")

        except Exception as e:
            print(f"Error relaxing structure {file_name}: {e}")
            print(traceback.format_exc())

# Save results to CSV
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_output_file, index=False)
    print(f"Optimization results saved to {csv_output_file}")
else:
    print("No results to save. Check for errors during processing.")
