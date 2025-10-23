import numpy as np
from scipy.optimize import minimize


# Симульовані дані: X1 = перегляди, X2 = завантаження
X = np.array([[5, 2],
              [3, 1],
              [8, 4],
              [6, 3]])
Y = np.array([10, 5, 15, 9])


# Функція втрат (сума квадратів помилок)
def loss(params):
    a, b = params
    predictions = X @ np.array([a, b])
    return np.sum((Y - predictions)**2)


# Початкове наближення
x0 = np.zeros(2)


# Використовуємо BFGS
result = minimize(loss, x0, method='BFGS')
print("Оптимальні коефіцієнти активності (BFGS):", result.x)
print("Значення функції втрат:", result.fun)

