import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay


class DataGenerator:

    def __init__(self, dataset_name=None, file_path=None):
        self.dataset_name = dataset_name
        self.file_path = file_path

    def generate_dataset(self):
        """Load a dataset from a file and return a merged pandas DataFrame and Series."""
        data = np.load(self.file_path, allow_pickle=True).item()
        feature = data['images'][:500]
        target = data['labels'][:500]

        # Apply Min-Max Scaling to the range [0, π]
        scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        X_scaled = scaler.fit_transform(feature)
        y = np.where(target == 1.0, 1, -1)
        
        return (
                    pd.DataFrame(X_scaled, columns=[f'Feature {i+1}' for i in range(X_scaled.shape[1])]),
                    pd.Series(y, name='Label')
               )
    
    