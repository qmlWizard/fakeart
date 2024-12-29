import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import yaml
import time
import os


class TrainQNN():
    def __init__(self, qnn, training_data, training_labels, testing_data, testing_labels,  optimizer, lr,
                 epochs, train_method, target_accuracy=None, base_path=None
                ):
        
        super().__init__()
        self._qnn = qnn
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = epochs
        self._target_accuracy = target_accuracy
        self._training_data = training_data
        self._training_labels = training_labels
        self._training_labels = self._training_labels.to(torch.float32)
        self._testing_data = testing_data
        self._testing_labels = testing_labels
        self._testing_labels = self._testing_labels.to(torch.float32)
        self._n_classes = torch.unique(training_labels)
        self._executions = 0
        self._lr = lr
        self._base_path = base_path
        self._loss_arr = []
        self.validation_accuracy_arr = []
        self.initial_training_accuracy = None
        self.final_training_accuracy = None
        self.initial_testing_accuracy = None
        self.final_testing_accuracy = None
        self._per_epoch_executions = None

        self._loss_function = nn.CrossEntropyLoss()
            
        if optimizer == 'adam':
            self._qnn_optimizer = optim.Adam(self._qnn.parameters(), lr = self._lr)
        elif optimizer == 'gd':
            self._qnn_optimizer = optim.SGD(self._qnn.parameters(), lr = self._lr)

        print("Epochs: ", self._epochs)


    def fit(self, training_data, training_labels):
        for epoch in range(self._epochs):
            x_0 = training_data
            self._qnn_optimizer.zero_grad()   
            _output = self._qnn(x_0)
            loss = self._loss_function(_output, training_labels)
            loss.backward()
            self._qnn_optimizer.step()
            self._loss_arr.append(loss.item())

            print(f"Epoch {epoch}th: Loss: {self._loss_arr[-1]}")