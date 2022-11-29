import numpy as np
from scipy.stats import t

import os
import sys

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

np.set_printoptions(suppress=True)

from ols import B, B_var
from data import N, M
from models import MODEL_NAMES

alpha = np.array([0.05, 0.01]) # significance values for confidence intervals

t_tab = np.zeros_like(alpha)   # Student distribution table values for given alphas
for i, a in enumerate(alpha):
    t_tab[i] = t.ppf(1 - a/2, N - M)

def printIntervals(left_bounds, right_bounds):
    for i in range(M + 1):
        print(f'{i+1}) {left_bounds[i]:+.4f} < b_{i} < {right_bounds[i]:+.4f}')

I = np.arange(0, M + 1) # index variable

# build confidence intervals for each model and for each alpha value
for i, b in enumerate(B):
    b_var_matrix = B_var[i]
    b_sd = np.sqrt(b_var_matrix[I, I]) # calculate standard deviations from variance matrix

    print(f'\n{MODEL_NAMES[i].capitalize()} regression:', f'B = {b}', sep='\n')
    for j, a in enumerate(alpha):
        left_interval_bounds  = b - b_sd * t_tab[j]
        right_interval_bounds = b + b_sd * t_tab[j]

        print(f'\nAlpha = {a}:')
        printIntervals(left_interval_bounds, right_interval_bounds)