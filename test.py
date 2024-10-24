import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
import os

def flatten_data(loader):
    data = []
    labels = []
    for images, label in loader:
        images = images.view(images.size(0), -1).numpy()  # Flattening the 28x28 images to 1D (784)
        data.append(images)
        labels.append(label.numpy())
    return np.vstack(data), np.concatenate(labels)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Set up MNIST Fashion dataset and DataLoader
fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader_fashion = DataLoader(fashion_mnist_train, batch_size=64, shuffle=True)
test_loader_fashion = DataLoader(fashion_mnist_test, batch_size=64, shuffle=False)

# Fashion MNIST class labels
# 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

# For binary classification, we are choosing shirts (class 6) and pants/trousers (class 1)
def filter_shirts_and_pants(data, labels, shirt_class=6, pants_class=1):
    binary_mask = np.isin(labels, [shirt_class, pants_class])
    binary_data = data[binary_mask]
    binary_labels = labels[binary_mask]
    # Convert labels to 0 (shirts) and 1 (pants)
    binary_labels = np.where(binary_labels == shirt_class, 0, 1)
    return binary_data, binary_labels

# Flatten data for PCA
fashion_train_data, fashion_train_labels = flatten_data(train_loader_fashion)

# Filter for shirts and pants
binary_fashion_train_data, binary_fashion_train_labels = filter_shirts_and_pants(fashion_train_data, fashion_train_labels)

# Apply PCA for different numbers of components: 2, 4, 8, 12, 24, 48, 96
binary_pca_fashion_results = {}
components = [2, 4, 8, 12, 24, 48, 96]


for n_components in components:
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(binary_fashion_train_data)
    binary_pca_fashion_results[n_components] = transformed_data

# Saving the binary classification (shirts vs pants) transformed data
save_dir = './fashion_mnist_pca_binary_data'
os.makedirs(save_dir, exist_ok=True)

# Save the transformed data as a dictionary with 'images' and 'labels' keys
for n_components, transformed_data in binary_pca_fashion_results.items():
    data_dict = {
        'images': transformed_data,
        'labels': binary_fashion_train_labels
    }
    filename = os.path.join(save_dir, f'fashion_mnist_{n_components}_binary.npy')
    np.save(filename, data_dict)
