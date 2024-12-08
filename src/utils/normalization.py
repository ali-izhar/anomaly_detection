# src/utils/normalization.py

import pickle


class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Compute mean and std from training data.
        Args:
            data (np.ndarray): Shape (num_samples, num_features)
        """

        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        # To avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Apply normalization.
        Args:
            data (np.ndarray): Shape (num_samples, num_features)
        Returns:
            np.ndarray: Normalized data
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverse normalization.
        Args:
            data (np.ndarray): Normalized data
        Returns:
            np.ndarray: Original scale
        """
        return data * self.std + self.mean

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        norm = cls()
        norm.mean = params["mean"]
        norm.std = params["std"]
        return norm
