import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.ansatz import qnnhe, qnncovariant, qnnembedding_paper
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class QNN(nn.Module):
    
    def __init__(self, device, n_qubits, trainable, input_scaling, data_reuploading, ansatz, ansatz_layers):
        super().__init__()
        
        self._device = device
        self._n_qubits = n_qubits
        self._trainable = trainable
        self._input_scaling = input_scaling
        self._data_reuploading = data_reuploading
        self._ansatz = ansatz
        self._layers = ansatz_layers
        self._wires = range(self._n_qubits)
        self._projector = torch.zeros((2**self._n_qubits,2**self._n_qubits))
        self._projector[0,0] = 1
        self._circuit_executions = 0
        self._n_classes = 2

        if self._ansatz == 'he':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=False))
            self.register_parameter(name="variational", param= nn.Parameter((torch.rand(self._layers, self._n_qubits * 2) * 2 * math.pi) - math.pi, requires_grad=True))

        elif self._ansatz == 'embedding_paper':
            if self._input_scaling:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param= nn.Parameter(torch.ones(self._layers, self._n_qubits), requires_grad=False))
            self.register_parameter(name="variational", param= nn.Parameter((torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi, requires_grad=True))
            self.register_parameter(name="rotational", param= nn.Parameter((torch.rand(self._layers, self._n_qubits) * 2 * math.pi) - math.pi, requires_grad=True))

        dev = qml.device(self._device, wires = range(self._n_qubits))
        if self._ansatz == 'he':
            self._qnn = qml.QNode(qnnhe, dev, interface='torch')
        elif self._ansatz == 'embedding_paper':
            self._qnn = qml.QNode(qnnembedding_paper, dev, interface='torch')
        elif self._ansatz == 'covariant':
            self._qnn = qml.QNode(qnnhe, dev, diff_method='adjoint', interface='torch')
        else:
            #self._kernel = qml.QNode(qkhe, dev, diff_method='adjoint', interface='torch')
            print("No Kernel Ansatz selected!")

    def _decode(self, output):
        __num_probabilities = 2 ** self._n_qubits
        if __num_probabilities >= self._n_classes:
            __output = output[:self._n_classes]
        else:
            raise ValueError(f"Number of classes ({self._n_classes}) exceeds the number of qubit states ({__num_probabilities}).")
        return __output
        
    def forward(self, x1):
        __output = self._qnn(x1, self._parameters, self._wires, self._layers, self._projector, self._data_reuploading)
        print(__output)
        #__pred = self._decode(__output)
        return __output