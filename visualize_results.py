import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_report():
    # 1. Load the results
    try:
        df = pd.read_csv('results.csv')
    except:
        print("results.csv not found. Run main.py first.")
        return

    # 2. Calculate Averages for the 3 models
    # We compare CNOT counts as per your PDF's 'Circuit Optimization' claim
    methods = ['FRQI', 'NEQR', 'QVPR_Novel']
    avg_cnots = [df['FRQI_CNOTs'].mean(), df['NEQR_CNOTs'].mean(), df['QVPR_Novel_CNOTs'].mean()]
    avg_depth = [df['FRQI_Depth'].mean(), df['NEQR_Depth'].mean(), df['QVPR_Novel_Depth'].mean()]

    # 3. Create the Bar Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    
    # Plot CNOT Count
    plt.bar(x, avg_cnots, color=['blue', 'orange', 'green'], alpha=0.7, label='Avg CNOT Count')
    
    plt.xticks(x, methods)
    plt.ylabel('Gate Count (Complexity)')
    plt.title('Comparison of Quantum Image Representation Techniques (Brain Tumor Dataset)')
    plt.legend()
    
    # Annotate values
    for i, v in enumerate(avg_cnots):
        plt.text(i - 0.1, v + (max(avg_cnots)*0.01), str(int(v)), fontweight='bold')

    plt.savefig('performance_comparison.png')
    print("Graph saved as performance_comparison.png")
    plt.show()

if __name__ == "__main__":
    generate_report()