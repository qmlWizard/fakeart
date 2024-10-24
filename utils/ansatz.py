import pennylane as qml
from pennylane import numpy as np

def efficientSU2(params, wires):
	for i in range(len(wires)):
		qml.RX(params[i, 0], wires=wires[i])
		qml.RY(params[i, 1], wires=wires[i])

	for i in range(len(wires) - 1):
		qml.CNOT(wires=[wires[i], wires[i + 1]])
    
	qml.CNOT(wires=[wires[len(wires) - 1], wires[0]])
    
	for i in range(len(wires)):
		qml.RX(params[i, 2], wires=wires[i])
		qml.RY(params[i, 3], wires=wires[i])