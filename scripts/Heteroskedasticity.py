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

n = 46

# функція для обчислення мью-критерію
def mu_criterion(y: np.ndarray, k: int):
    # Розбиваємо спостереження на k груп однакового розміру
    n_r = n // k
    s_r = np.zeros(k)

    for i in range(k):
        if i == k - 1:
            y_r = y[n_r * i:]
        else:
            y_r = y[n_r * i: n_r * (i + 1)]
            
        y_r_mean = np.average(y_r)

        s_r[i] = np.sum((y_r - y_r_mean)**2)
        
    s = np.sum(s_r)
    
    mu_num = np.prod((s_r / n_r)**(n_r / 2))
    mu_denom = (s / n) ** (n / 2)
    
    mu = -2 * np.log(mu_num / mu_denom)
    return mu

alphaA = 0.05
chi_tabA = 9.48
alphaB = 0.01
chi_tabB = 13.28

mu = mu_criterion(y, 5)
print(f'\nКритерій мю - {mu}')
print(f'\nРівень значущості alpha = {alphaA}')
print(f'Табличне значення критерію - {chi_tabA}')
if mu < chi_tabA:
    print('Явище гетероскедастичності відсутнє')
else:
    print('Спостерігається гетероскедастичність')

print(f'\nРівень значущості alpha = {alphaB}')
print(f'Табличне значення критерію - {chi_tabB}')
if mu < chi_tabB:
    print('Явище гетероскедастичності відсутнє')
else:
    print('Спостерігається гетероскедастичність')