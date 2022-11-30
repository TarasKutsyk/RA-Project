import numpy as np
from scipy.stats import t

np.set_printoptions(suppress=True)

import os
import sys

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from ols import B, B_var, Y_pred
from data import N, M, y
from models import MODEL_NAMES

alpha = np.array([0.05, 0.01]) # significance values for significance test

t_tab = np.zeros_like(alpha)   # Student distribution table values for given alphas
for i, a in enumerate(alpha):
    t_tab[i] = t.ppf(1 - a/2, N - M - 1)

def printTestResults(test_results):
    for i, passed in enumerate(test_results):
        conclusion = 'significant' if passed else 'insignificant'
        print(f'{i+1}) b_{i} is {conclusion}')

I = np.arange(0, M + 1) # index variable

# test parameter significance for each model and for each alpha value
for i, b in enumerate(B):
    b_var_matrix = B_var[i]
    b_sd = np.sqrt(b_var_matrix[I, I]) # calculate standard deviations from variance matrix

    # calculate mean squared error for predicted y values
    sigma = np.sqrt(np.sum((y - Y_pred[i])**2) / (N - M - 1))

    print(f'\n{MODEL_NAMES[i].capitalize()} regression:', f'B = {b}', sep='\n')

    t_values = np.abs(b) / (sigma * b_sd)
    print(f't-criterion values for given parameters B:', t_values, sep='\n')

    for j, a in enumerate(alpha):
        t_test = t_values > t_tab[j]

        print(f'\nAlpha = {a}, t_tab = {t_tab[j]}:')
        printTestResults(t_test)