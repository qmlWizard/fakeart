import pennylane as qml
from pennylane import numpy as np
import pennylane as qml
import numpy as np

def naqss(combined_angles, num_qubits, pixels):

    for i in range(num_qubits - 1):
        qml.Hadamard(wires=[i])
    # Apply the first RX rotation to the first qubit
    qml.RX(combined_angles[0], wires=0)

    # Apply subsequent controlled rotations based on the NAQSS scheme
    index = 1  # To track the position in combined_angles
    for i in range(1, num_qubits - 1):
        for j in range(int(np.power(2, i))):
            # Apply RX rotation to qubit i
            qml.RX(combined_angles[index], wires=i)
            index += 1
            
            # Determine which control qubits to flip (i.e., where the binary value is 0)
            control_wires = [qubit for qubit in range(i)]
            binary_rep = format(j, f'0{i}b')  # Get binary representation of the loop index
            for ctrl_idx, bit in enumerate(binary_rep):
                if bit == '0':
                    qml.PauliX(wires=control_wires[ctrl_idx])  # Apply X gate to flip the qubit if needed

            # Apply MultiControlledX with previous qubits as control
            qml.MultiControlledX(control_wires=control_wires, wires=i)

            # Revert the qubits back by applying X gate again
            for ctrl_idx, bit in enumerate(binary_rep):
                if bit == '0':
                    qml.PauliX(wires=control_wires[ctrl_idx])  # Revert the flip
    
    # Apply rotations and controlled operations for the pixels
    for k in range(pixels):
        if index >= len(combined_angles):
            pass
        else:
            qml.RX(combined_angles[index], wires=num_qubits - 1)
            index += 1

            # Determine which control qubits to flip (all previous qubits)
            control_wires = [qubit for qubit in range(num_qubits - 1)]
            binary_rep = format(k, f'0{num_qubits - 1}b')
            for ctrl_idx, bit in enumerate(binary_rep):
                if bit == '0':
                    qml.PauliX(wires=control_wires[ctrl_idx])  # Apply X gate to flip the qubit if needed

            # MultiControlledX with all previous qubits as control
            qml.MultiControlledX(control_wires=control_wires, wires=num_qubits - 1)

            # Revert the qubits back by applying X gate again
            for ctrl_idx, bit in enumerate(binary_rep):
                if bit == '0':
                    qml.PauliX(wires=control_wires[ctrl_idx])  # Revert the flip
