import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

import os
import sys

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from data import y, N
from models import MODEL_NAMES, MODELS_COUNT, MODELS_MATRICES

B = []      # least squares parameters estimates estimates for each model
Y_pred = [] # predicted y values for each model
B_var = []  # least squares parameters variance for each model

for i in range(MODELS_COUNT):
    # get design matrix for the current model
    F = MODELS_MATRICES[i]

    # least squares parameters estimates for the current model
    b = inv(F.T @ F) @ F.T @ y

    # predicted y values for the current model
    y_pred = np.dot(F, b)

    # LS-estimates variance matrix for the current model
    b_variance = inv(F.T @ F)

    B.append(b)
    Y_pred.append(y_pred)
    B_var.append(b_variance)

if __name__ == '__main__':
    for i in range(MODELS_COUNT):
        print(f'\n{MODEL_NAMES[i].capitalize()} regression: ')
        print('Least squares estimates:', B[i], sep='\n')
        print('Predicted y values:', Y_pred[i], sep='\n')
        print('Prediction Mean Squared Error: ', np.mean((y - Y_pred[i])**2), sep='\n')

        # Plot LS prediction values vs real ones
        x = np.arange(0, N)

        plt.scatter(x, y, label='Real rating')
        plt.scatter(x, Y_pred[i], label=f'Predicted rating: {MODEL_NAMES[i]} regression')
        plt.legend(loc='lower right')
        plt.show()