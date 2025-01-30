import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.ansatz import qkhe
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import os

torch.manual_seed(42)
np.random.seed(42)


dev = qml.device("default.qubit", wires = 20)

class qnn(nn.Module):
    
    def __init__(self, wires = None, ansatz = 'he', encoding = 'angle', rotation = 'Z', layers = 1, reuploading = True, n_classes = 2):
        super().__init__()
        self._wires = wires
        if wires is None:
            raise ValueError("Wires cannot be None")
        
        if n_classes < 2:
            raise ValueError("Only 1 class found in dataset! Classes should be >= 2.")
        
        self._layers = layers
        self._reuploading = reuploading
        self.n_qubits = len(wires)
        self._ansatz = ansatz
        self._encoding = encoding
        self._rotation = rotation
        self._n_classes = n_classes

        weight_shape = self._get_ansatz_shape()
        self.weights = nn.Parameter(torch.randn(*weight_shape))


    def _ansatz_layer(self, weights):
        if self._ansatz == 'he':
            efficientSU2(weights, wires = self._wires)
    
    def _get_ansatz_shape(self):
        if self._ansatz == 'he':
            return (self._layers, self.n_qubits, 4)

    def _encoding_circuit(self, input):

        if self._encoding == 'angle':
            qml.AngleEmbedding(input, wires=self._wires, rotation= self._rotation)
        else:
            qml.transforms.broadcast_expand(qml.AmplitudeEmbedding)(input, wires=self._wires, normalize=True)

    @qml.qnode(dev, interface = 'torch')
    def _qnn_circuit(self, input, weights):

        if self._reuploading:
            for l in range(self._layers):
                self._encoding_circuit(input)
                self._ansatz_layer(weights[l])

        else:
            self._encoding_circuit(input)
            for l in range(self._layers):
                self._ansatz_layer(weights[l])

        return qml.probs(wires=self._wires)

    def circuit_and_summary(self, input, weights, mode = 'terminal'):
        
        if mode == 'terminal':
            print(qml.draw(self._qnn_circuit)(self, input=input, weights=weights))
            
        print(self._get_summary())

    def _decode(self, output):
        """
        Decoding logic based on the number of classes and the number of qubits.
        If the number of probabilities (2^n_qubits) is greater than the number of classes,
        we use the first num_classes probabilities.
        """
        # Get the number of probabilities based on the number of qubits
        __num_probabilities = 2 ** self.n_qubits
        
        if __num_probabilities >= self._n_classes:
            # Use the first num_classes probabilities
            __output = output[:self._n_classes]
        else:
            raise ValueError(f"Number of classes ({self._n_classes}) exceeds the number of qubit states ({__num_probabilities}).")

        # For multi-class classification, return class with the highest probability
        #return torch.argmax(output, dim=-1)
        return __output

    def forward(self, inputs):
        __output = self._qnn_circuit(self, inputs, self.weights)
        __pred = self._decode(__output)
        return __pred.view(-1, self._n_classes)
    
    def _get_summary(self):
        d = {
                'encoding': self._encoding,
                'ansatz': self._ansatz,
                'layers': self._layers,
                'parameter_shape': self._get_ansatz_shape()
            }
        
        return d

    def get_random_params(self):
        weight_shape = self._get_ansatz_shape()
        self.weights = nn.Parameter(torch.randn(*weight_shape))
        #return np.random.uniform(-np.pi, np.pi, self._get_ansatz_shape(), requires_grad=True)
        return self.weights
    
    def train(self, dataloader, optimizer, criterion, epochs, save_model_path = 'models/', test_dataloader=None, metrics_file="result/qnn_metrics.json"):
        metrics = {
            "accuracy_per_epoch": [],
            "loss_per_epoch": [],
            "eval_metrics_per_epoch": []  # For evaluation metrics like precision, recall, etc.
        }

        # Ensure the directory exists
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_samples

            # Print the average loss and accuracy for the epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # Store training metrics
            metrics["loss_per_epoch"].append(epoch_loss)
            metrics["accuracy_per_epoch"].append(epoch_accuracy)
     
        # Evaluate model after each epoch if test_dataloader is provided
        if test_dataloader:
            eval_metrics = self.evaluate_model(test_dataloader, num_classes=self._n_classes)
            metrics["eval_metrics_per_epoch"].append(eval_metrics)

        # Save metrics to JSON file after each epoch
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save the model
        torch.save(self.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    def evaluate_model(self, dataloader, num_classes):
        # Set the model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, dim=1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = None

        eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
        }

        # Print evaluation metrics
        print(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        if auc is not None:
            print(f"AUC Score: {auc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Return evaluation metrics for storing in the JSON file
        return eval_metrics
    
class hybrid(nn.Module):
    pass

class classical(nn.Module):
    pass

class qkernel(nn.Module):
    pass

