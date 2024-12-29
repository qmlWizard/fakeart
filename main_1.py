import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import time
import os 
import argparse
import shutil
from collections import namedtuple
import json

from utils.data_generator import DataGenerator
from utils.train_qkernel import TrainQkernel
from utils.qkernel import Qkernel
from utils.qnn import QNN
from utils.train_qnn import TrainQNN
from utils.helper import to_python_native


##Backend Configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS is available. Using device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Neither MPS nor CUDA is available. Using CPU: {device}")

torch.manual_seed(42)
np.random.seed(42)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "This parser receives the yaml config file")
    parser.add_argument("--config", default = "configs/checkerboard.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())

    data_generator = DataGenerator(
        dataset_name=config.dataset['name'],
        file_path=config.dataset['file']
    )

    features, target = data_generator.generate_dataset()
    training_data, testing_data, training_labels, testing_labels = train_test_split(features, target, test_size=0.50, random_state=42)
    training_data = torch.tensor(training_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    testing_data = torch.tensor(testing_data.to_numpy(), dtype=torch.float32, requires_grad=True)
    training_labels = torch.tensor(training_labels.to_numpy(), dtype=torch.int)
    testing_labels = torch.tensor(testing_labels.to_numpy(), dtype=torch.int)

    if config.training['model'] == 'qkernel':
        kernel = Qkernel(
                            device=config.qkernel['device'],
                            n_qubits=config.qkernel['n_qubits'],
                            trainable=config.qkernel['trainable'],
                            input_scaling=config.qkernel['input_scaling'],
                            data_reuploading=config.qkernel['data_reuploading'],
                            ansatz=config.qkernel['ansatz'],
                            ansatz_layers=config.qkernel['ansatz_layers']
                        )
        agent = TrainQkernel(
                                kernel=kernel,
                                training_data=training_data,
                                training_labels=training_labels,
                                testing_data=testing_data,
                                testing_labels=testing_labels,
                                optimizer=config.agent['optimizer'],
                                lr=config.agent['lr'],
                                epochs=config.agent['epochs'],
                                train_method=config.agent['train_method'],
                                target_accuracy=config.agent['target_accuracy'],
                                get_alignment_every=config.agent['get_alignment_every'],
                                validate_every_epoch=config.agent['validate_every_epoch'],
                                base_path=config.agent['base_path'],
                                clusters=config.agent['clusters']
                            )
        
        before_metrics = agent.evaluate(testing_data, testing_labels)
        print(before_metrics)
        agent.fit_kernel(training_data, training_labels)
        print('Training Complete')
        after_metrics = agent.evaluate(testing_data, testing_labels)
        metrics = {
            "num_layers": config.qkernel['ansatz_layers'],
            "accuracy_train_init": before_metrics['training_accuracy'],
            "accuracy_test_init": before_metrics['testing_accuracy'],
            "alignment_train_init": before_metrics['alignment'],
            "accuracy_train_final": after_metrics['training_accuracy'],
            "accuracy_test_final": after_metrics['testing_accuracy'],
            "alignment_train_epochs": after_metrics['alignment_arr'],
            "circuit_executions": after_metrics['executions'],
        }
        metrics = to_python_native(metrics)

        # Specify the filename
        filename = f"results/qkernel_{config.dataset['name']}_{config.agent['clusters']}.json"

        # Write the JSON-serializable results to a file
        with open(filename, "w") as file:
            json.dump(metrics, file, indent=4)

    elif(config.training['model'] == 'qnn'):
        qnn = QNN(
                    device=config.qkernel['device'],
                    n_qubits=config.qkernel['n_qubits'],
                    trainable=config.qkernel['trainable'],
                    input_scaling=config.qkernel['input_scaling'],
                    data_reuploading=config.qkernel['data_reuploading'],
                    ansatz=config.qkernel['ansatz'],
                    ansatz_layers=config.qkernel['ansatz_layers']
                 )
        
        agent = TrainQNN(
                                qnn=qnn,
                                training_data=training_data,
                                training_labels=training_labels,
                                testing_data=testing_data,
                                testing_labels=testing_labels,
                                optimizer=config.agent['optimizer'],
                                lr=config.agent['lr'],
                                epochs=config.agent['epochs'],
                                train_method=config.agent['train_method'],
                                target_accuracy=config.agent['target_accuracy'],
                                base_path=config.agent['base_path'],
                            )
        agent.fit(training_data, training_labels)