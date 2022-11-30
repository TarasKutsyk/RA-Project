import numpy as np

import os
import sys

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from ols import B
from data import MIN_YEAR, numerical_data
from models import MODEL_NAMES

YEAR = 2010    # year for which the prediction is to be made

# x1 - Year
# x2 - Deaths count
# x3 - Duration
x1, x2, x3 = numerical_data.T

min_x2 = x2.min()
max_x2 = x2.max()

min_x3 = x3.min()
max_x3 = x3.max()

def calculate_prediction(x1, x2, x3, b, model_number):
    # the year value needs to be normalized the same way as it is done in the dataset.py file
    x1 = x1 - MIN_YEAR + 1

    factors = np.array([1, x1, x2, x3])

    if model_number == 1:   # logarithmic model
        factors = np.log(factors)
    elif model_number == 2: # hyperbolic model
        factors = 1 / factors

    y = np.dot(factors, b)

    return y

print('Predicted rating value y for 2010 movie is:')
for i, b in enumerate(B):
    x2_left = max_x2 if b[2] < 0 else min_x2
    x3_left = max_x3 if b[3] < 0 else min_x3

    left_prediction_bound = calculate_prediction(YEAR, x2_left, x3_left, b, i)
    
    x2_right = min_x2 if b[2] < 0 else max_x2
    x3_right = min_x3 if b[3] < 0 else max_x3

    right_prediction_bound = calculate_prediction(YEAR, x2_right, x3_right, b, i)

    # hyperbolic model is monotonically decreasing by each factor, so bounds are switched
    if i == 2:
        left_prediction_bound, right_prediction_bound = right_prediction_bound, left_prediction_bound
    
    print(f'\n{i+1}) {MODEL_NAMES[i].capitalize()} regression:')
    print(f'{left_prediction_bound:.4f} < y < {right_prediction_bound:.4f}')