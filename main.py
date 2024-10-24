import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import time
import os 

#Custom Libraries

from utils.models import qnn, hybrid, classical, qkernel
from utils.preprocessing import read_data

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

#Read Configs
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Dataset
if config['dataset']['generate']:
    read_data(config['dataset']['name'], config['dataset']['image_size'], config['dataset']['preprocess'])
data = np.load(f"data/{config['dataset']['name']}_{config['dataset']['image_size']}.npy", allow_pickle=True).item()
feature = data['images']
target = data['labels']
feature_dimensions =  len(feature[0]) #math.ceil(math.log2(len(feature[0])))
n_classes = len(np.unique(target))

# Convert features and labels to torch tensors
features_tensor = torch.tensor(feature, dtype=torch.float32)  # Convert features to float tensor
labels_tensor = torch.tensor(target, dtype=torch.long)  # Convert labels to long tensor (for classification)

# Create TensorDataset
dataset = TensorDataset(features_tensor, labels_tensor)

# Split dataset into 75% training and 25% testing
train_size = int(0.75 * len(dataset))  # 75% for training
test_size = len(dataset) - train_size   # Remaining 25% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define DataLoader for training data
batch_size = config['training']['batch_size']  # Assuming batch size is defined in your config
shuffle = config['training']['shuffle']  # Shuffle the data at every epoch

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

# Define DataLoader for testing data (no shuffling)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Now, `train_dataloader` contains 75% of the data and `test_dataloader` contains 25%
print(f"Training Data Size: {train_size}, Testing Data Size: {test_size}")
# Test: Print some information about the DataLoader
print(f"Number of batches: {len(train_dataloader)}")
for i, (inputs, labels) in enumerate(train_dataloader):
    print(f"Batch {i+1} - Input shape: {inputs.shape}, Labels shape: {labels.shape}")
    # Just printing the first batch for checking
    if i == 0:
        break

#Model configs
if config['model']['type'] == 'qnn':
    model = qnn( 
                    wires= range(feature_dimensions),
                    ansatz= config['model']['ansatz'],
                    encoding= config['model']['embedding_type'],
                    rotation= config['model']['embedding_rotation'],
                    layers = config['model']['layers'],
                    reuploading = config['model']['reuploading'],
                    n_classes = n_classes
               )
    init_weights = model.get_random_params()
    print(f"{config['model']['type']} model is created with following configurations, ")
    print(model.circuit_and_summary(input = feature[0], weights = init_weights, mode='terminal'))
    print(model._get_summary())
    print(model.parameters)

    #Optimizer Configs
    if config['training']['optimizer'] == 'gd':
        opt = torch.optim.SGD(model.parameters(), lr=config['gd']['learning_rate'])
        epochs = config['gd']['max_iter']
        if config['gd']['criterion'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
    if config['training']['optimizer'] == 'adam':
        opt = torch.optim.adam(model.parameters(), lr=config['adam']['learning_rate'])     
        epochs = config['adam']['max_iter']
        if config['adam']['criterion'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()

    model.train(dataloader = train_dataloader,
              optimizer = opt,
              criterion = criterion,
              epochs = epochs,
              test_dataloader=test_dataloader
             )