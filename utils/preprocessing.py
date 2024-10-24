import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

def calculate_alpha(angles, num_qubits):
    alpha_values = []
    for j in range(1, num_qubits + 1):
        if j == 1:
            denominator = np.sum(np.abs(angles[0::2])**2)
            alpha_1 = 0 if denominator == 0 else np.arctan(np.sqrt(np.sum(np.abs(angles[1::2])**2) / denominator))
            alpha_values.append(alpha_1)
        else:
            for i in range(2**(j-1)):
                indices_0 = [k for k in range(2**num_qubits) if f"{k:0{num_qubits}b}"[:j] == f"{i:0{j}b}" + "0"]
                indices_1 = [k for k in range(2**num_qubits) if f"{k:0{num_qubits}b}"[:j] == f"{i:0{j}b}" + "1"]
                denominator = np.sum(np.abs(angles[indices_0])**2)
                alpha_j = 0 if denominator == 0 else np.arctan(np.sqrt(np.sum(np.abs(angles[indices_1])**2) / denominator))
                alpha_values.append(alpha_j)
    return alpha_values

def pca_reduction(images, n_components):
    num_images, height, width = images.shape
    # Flatten each image into a 1D array
    images_flattened = images.reshape(num_images, height * width)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(images_flattened)
    return pca_transformed

def read_data(dataset, image_size, preprocess='pca'):
    data = []
    labels = []
    qubits = int(np.log2(image_size * image_size)) + 1
    M = 256
    path = f'data/{dataset}'
    
    ldir = [f for f in os.listdir(path) if not f.startswith('.DS_Store')]
    label = 0

    for d in ldir:
        limages = [f for f in os.listdir(path + '/' + d) if not f.startswith('.DS_Store')]
        
        if preprocess == 'naqss':
            for image in limages:
                painting = cv2.imread(path + '/' + d + '/' + image, cv2.IMREAD_GRAYSCALE)
                painting = cv2.resize(painting, (image_size, image_size))
                theta = np.pi * ((painting - 1) / (M - 1))
                theta = theta / np.sum(theta)
                alpha = calculate_alpha(theta, qubits - 1)
                palpha = [item for sublist in theta for item in sublist]
                angles = alpha + palpha
                data.append(np.asarray(angles))
                labels.append(label)
        
        if preprocess == 'pca':
            images = []
            for image in limages:
                painting = cv2.imread(path + '/' + d + '/' + image, cv2.IMREAD_GRAYSCALE)
                painting = cv2.resize(painting, (image_size, image_size))
                images.append(painting)
            
            images = np.array(images)
            pca_result = pca_reduction(images=images, n_components=image_size)
            
            # Add the PCA-transformed data to the dataset
            data.append(pca_result)
            
            # Add the label for each image
            labels.extend([label] * len(pca_result))
        
        label += 1
    
    # Convert data to a structured format and save
    data = np.vstack(data)  # Combine the list of arrays into one 2D array
    d = {
        'images': data,
        'labels': labels
    }
    
    np.save(f'data/{dataset}_{image_size}.npy', d)

