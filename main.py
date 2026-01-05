import os
import pandas as pd
import h5py
from preprocessing import get_all_file_paths, load_medical_data
from quantum_models import get_frqi_circuit, get_neqr_circuit, get_novel_qvpr_circuit
from qiskit import transpile


DATA_DIR = r'C:\Users\sagar\Downloads\quantum_image_representaion\data'         # Folder where your 3064 files are
OUTPUT_CSV = r'C:\Users\sagar\Downloads\quantum_image_representaion\outputs' # Results will be saved here
IMAGE_SIZE = 16            
LIMIT_FILES = 10           # Set to 5 or 10 for testing; increase later

def run_batch_experiment():
    file_list = get_all_file_paths(DATA_DIR)
    results = []
    
    # Limit files for the first run
    process_list = file_list[:LIMIT_FILES]
    print(f"Starting analysis on {len(process_list)} files...")

    for file_path in process_list:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")
        
        img, mask = load_medical_data(file_path, size=IMAGE_SIZE)
        if img is None: continue

        # Define the models to compare
        models = {
            "FRQI": get_frqi_circuit(img),
            "NEQR": get_neqr_circuit(img),
            "QVPR_Novel": get_novel_qvpr_circuit(img, mask)
        }
        
        row = {"File": file_name}
        
        for name, qc in models.items():
            # Transpile to get real-world hardware costs
            # 'u' and 'cx' are standard universal gates
            t_qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=2)
            
            # Record metrics
            row[f"{name}_Depth"] = t_qc.depth()
            row[f"{name}_Gates"] = sum(t_qc.count_ops().values())
            row[f"{name}_CNOTs"] = t_qc.count_ops().get('cx', 0)
            
        results.append(row)

    # Save data to a CSV for your final report
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Results saved to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    run_batch_experiment()