import numpy as np
def load_data(file_name):
     #loading the data
     return np.loadtxt(file_name)
    
def compute_finite_differences(x0, x1, delta_t):
    #Computing the finite differences using the loaded x0 and x1 data.
    v = (x1 - x0) / delta_t
    return v
