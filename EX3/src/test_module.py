
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test_function():
    print("I was imported!")

# Load the Data -> returns 'data'
def load_data(file_path):
    #with open(file_path, 'r') as f:
    return np.loadtxt(file_path, delimiter=' ')


# Center the Data
def center_data(data):
    return data - np.mean(data, axis=0)

# Implement PCA using SVD
def perform_pca (data_centered):
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    return U, S, Vt

# Extract Principal Components
# Here the principal components are the columns of Vt (if data is n x m, Vt is m x m)

# Calculate Energy
def calculate_energy(S):
    return np.square(S) / np.sum(np.square(S))


def plot_data_with_pcs(data_centered, Vt):
    # Plot the Data
    plt.scatter(data_centered[:, 0], data_centered[:, 1])

    # Add Principal Components
    mean_data = data_centered.mean(axis=0)
    plt.quiver(mean_data[0], mean_data[1], Vt[0,0], Vt[0,1], scale=3, color='r')
    plt.quiver(mean_data[0], mean_data[1], Vt[1,0], Vt[1,1], scale=3, color='g')
    plt.title('PCA of Dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.axis('equal')

    # Show plot with principal components
    plt.show()