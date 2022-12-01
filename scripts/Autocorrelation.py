import os
import sys
import numpy as np

# To be able to make imports from another folder
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)

sys.path.append(PROJECT_ROOT)

from data.dataset import y
from LeastSquares.ols import Y_pred

D_H = 1.38
D_B = 1.67

def durbin_watson_criterion(u: np.ndarray):
    N = len(u)
    
    return sum([(u[i] - u[i - 1]) ** 2 for i in range(1, N)]) / sum(u ** 2)

print(f'\nТабличні значення: {D_H}, {D_B}')

for Y in Y_pred:
    u = y - Y
    d = durbin_watson_criterion(u)
    print(f'\n\nКритерій Дарбіна-Уотсона: d = {d}')

    if 0 < d < D_H:
        print(f'0 < {d} < {D_H}')
        print('Присутня позитивна автокореляція.')
    if 4 - D_H < d < 4:
        print(f'{4 - D_H} < {d} < 4')
        print('Присутня негативна автокореляція.')
    if D_B < d < 4 - D_B:
        print(f'{D_B} < {d} < {4 - D_B}')
        print('Автокореляція відсутня.')
    else:
        print('Критерій Дарбіна-Уотсона не дає відповідь щодо присутності автокореляції.')