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


class TrainQkernel():
    def __init__(self, kernel, training_data, training_labels, testing_data, testing_labels,  optimizer, lr,
                 epochs, train_method, target_accuracy=None, get_alignment_every=10, validate_every_epoch=10, 
                 base_path=None, clusters=4
                ):
        
        super().__init__()
        self._kernel = kernel
        self._optimizer = optimizer
        self._method = train_method
        self._epochs = epochs
        self._target_accuracy = target_accuracy
        self._get_alignment_every = get_alignment_every
        self._validate_every_epoch = validate_every_epoch
        self._sampling_size = clusters 
        self._clusters = clusters 
        self._training_data = training_data
        self._training_labels = training_labels
        self._training_labels = self._training_labels.to(torch.float32)
        self._testing_data = testing_data
        self._testing_labels = testing_labels
        self._testing_labels = self._testing_labels.to(torch.float32)
        self._n_classes = torch.unique(training_labels)
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._executions = 0
        self._lr = lr
        self._base_path = base_path
        self._main_centroids = []
        self._main_centroids_labels = []
        self._class_centroids = []
        self._class_centroid_labels = []
        self._loss_arr = []
        self.alignment_arr = []
        self.validation_accuracy_arr = []
        self.initial_training_accuracy = None
        self.final_training_accuracy = None
        self.initial_testing_accuracy = None
        self.final_testing_accuracy = None
        self._per_epoch_executions = None
            
        if self._method in ['random', 'full']:
            if optimizer == 'adam':
                self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr = self._lr)
            elif optimizer == 'gd':
                self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr = self._lr)
        if self._method == 'random':
            self._loss_function = self._loss_ta
            self._sample_function = self._sampler_random_sampling
        elif self._method == 'full':
            self._loss_function = self._loss_ta
            self._sample_function = self._full_data
            
        print("Epochs: ", self._epochs)
 
    def _loss_ta(self, K, y):
        N = y.shape[0]
        assert K.shape == (N,N), "Shape of K must be (N,N)"
        yT = y.view(1,-1) #Transpose of y, shape (1,N)
        Ky = torch.matmul(K,y) # K*y, shape (N,)
        yTKy = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar
        K2 = torch.matmul(K,K) #K^2, shape (N,N)
        trace_K2 = torch.trace(K2)
        result = yTKy / (torch.sqrt(trace_K2)* N)
        return result.squeeze()

    def _sampler_random_sampling(self, data, data_labels):
        subset_indices = torch.randperm(len(data))[:self._sampling_size]
        return data[subset_indices], data_labels[subset_indices]

    def _full_data(self, data, data_labels):
        return data, data_labels
    
    def fit_kernel(self, training_data, training_labels):
        optimizer = self._kernel_optimizer
        epochs = self._epochs
        loss_func = self._loss_function
        samples_func = self._sample_function
        self._per_epoch_executions = 0
        self.kernel_params_history = []
        self.best_kernel_params = None
        for epoch in range(epochs):
            sampled_data, sampled_labels = samples_func(training_data, training_labels)
            sampled_labels = sampled_labels.to(torch.float32)
            x_0 = sampled_data.repeat(sampled_data.shape[0],1)
            x_1 = sampled_data.repeat_interleave(sampled_data.shape[0], dim=0)
            optimizer.zero_grad()   
            K = self._kernel(x_0, x_1).to(torch.float32) 
            loss = -loss_func(K.reshape(sampled_data.shape[0],sampled_data.shape[0]), sampled_labels)
            loss.backward()
            optimizer.step()
            self._loss_arr.append(loss.item())
            self._per_epoch_executions += x_0.shape[0]
            if self._get_alignment_every and (epoch + 1) % self._get_alignment_every * 10 == 0:
                x_0 = training_data.repeat(training_data.shape[0],1)
                x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)
                K = self._kernel(x_0, x_1).to(torch.float32)
                self._training_labels = torch.tensor(self._training_labels, dtype = torch.float32) 
                current_alignment = loss_func(K.reshape(self._training_data.shape[0],self._training_data.shape[0]), self._training_labels)
                self.alignment_arr.append(current_alignment)
                print(current_alignment)

    def prediction_stage(self, data, labels):

        main_centroids = torch.stack([centroid.detach()[0] for centroid in self._main_centroids])
        x_0 = main_centroids.repeat(data.shape[0],1)
        x_1 = data.repeat_interleave(main_centroids.shape[0], dim=0)
        K = self._kernel(x_0, x_1).to(torch.float32).reshape(data.shape[0],main_centroids.shape[0])
        pred_labels = torch.sign(K[:, 0] - K[:, 1]) 
        correct_predictions = (pred_labels == labels).sum().item()  # Count matches
        total_predictions = len(labels)  # Total number of predictions
        accuracy = correct_predictions / total_predictions

        # Display results
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

     
    def evaluate(self, test_data, test_labels):
        x_0 = self._training_data.repeat(self._training_data.shape[0],1)
        x_1 = self._training_data.repeat_interleave(self._training_data.shape[0], dim=0)    
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(self._training_data.shape[0],self._training_data.shape[0])
        current_alignment = self._loss_ta(_matrix, self._training_labels)
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(self._training_labels):
            self._training_labels = self._training_labels.detach().numpy()
        self._model = SVC(kernel='precomputed', max_iter=10000).fit(_matrix, self._training_labels)
        predictions = self._model.predict(_matrix)
        training_accuracy = accuracy_score(self._training_labels, predictions)
        x_0 = self._testing_data.repeat_interleave(self._training_data.shape[0],dim=0)
        x_1 = self._training_data.repeat(test_data.shape[0], 1)
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(test_data.shape[0],self._training_data.shape[0])
        if torch.is_tensor(_matrix):
            _matrix = _matrix.detach().numpy()
        if torch.is_tensor(test_labels):
            test_labels = test_labels.detach().numpy()
        predictions = self._model.predict(_matrix)
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        metrics = {
            'alignment': current_alignment,
            'executions': self._per_epoch_executions,
            'training_accuracy': training_accuracy,
            'testing_accuracy': accuracy,
            'f1_score': f1,
            'alignment_arr': self.alignment_arr,
            'loss_arr': self._loss_arr,
            'validation_accuracy_arr': self.validation_accuracy_arr
        }
        return metrics