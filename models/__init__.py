import numpy as np

import os
import sys

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from data import numerical_data, N

MODEL_NAMES = ['linear','logarithmic','hyperbolic']
MODELS_COUNT = len(MODEL_NAMES)

# List of design matrices (F) for each model
MODELS_MATRICES = []

# Design matrix for linear model
F = np.concatenate([np.ones((N, 1)), numerical_data], axis=1)
MODELS_MATRICES.append(F)

# Design matrix for logarithmic model
F = np.concatenate([np.ones((N, 1)), np.log(numerical_data)], axis=1) # the factors are logarithmized to make the model linear
MODELS_MATRICES.append(F)

# Design matrix for hyperbolic model
F = np.concatenate([np.ones((N, 1)), 1 / numerical_data], axis=1) # the factors are inverted to make the model linear
MODELS_MATRICES.append(F)