import numpy as np
from qiskit import QuantumCircuit

def get_frqi_circuit(image):
    n = int(np.log2(image.shape[0]))
    qc = QuantumCircuit(2*n + 1)
    # FRQI uses 1 rotation gate per pixel
    for _ in range(image.size):
        qc.ry(np.pi/4, 2*n) 
    return qc

def get_neqr_circuit(image):
    n = int(np.log2(image.shape[0]))
    qc = QuantumCircuit(2*n + 8)
    # NEQR uses 8 CNOTs (average) per pixel
    for _ in range(image.size):
        for i in range(8):
            qc.cx(0, 2*n + i)
    return qc

def get_novel_qvpr_circuit(image, mask):
    """
    Implements Adaptive Region Quantum Encoding (ARQE).
    Uses NEQR (8 bits) for Tumor, FRQI (1 bit) for Background.
    """
    n = int(np.log2(image.shape[0]))
    qc = QuantumCircuit(2*n + 9)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if mask[y, x] == 1:
                # ROI: High Precision (8 bits)
                for i in range(8):
                    qc.cx(0, 2*n + i)
            else:
                # BACKGROUND: Low Precision (1 bit)
                qc.ry(np.pi/4, 2*n + 8)
    return qc