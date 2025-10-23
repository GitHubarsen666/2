import numpy as np
import matplotlib.pyplot as plt


# Симульовані дані: X1 = перегляди, X2 = завантаження
X = np.array([[5, 2],
              [3, 1],
              [8, 4],
              [6, 3]])
# Цільова активність: Y = загальна активність
Y = np.array([10, 5, 15, 9])


# Матриця A для квадратної функції похибки (для спряженого градієнта)
A = X.T @ X
b = X.T @ Y


# Градієнт
def grad(x):
    return A @ x - b


# Gradient Descent
def gradient_descent(x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = x0
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad(x)
        x = x - lr * g
        history.append(x.copy())
        if np.linalg.norm(g) < tol:
            break
    return x, np.array(history)


# Conjugate Gradient
def conjugate_gradient(x0, tol=1e-6, max_iter=100):
    x = x0
    r = b - A @ x
    p = r.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        Ap = A @ p
        alpha = r.T @ r / (p.T @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            history.append(x.copy())
            break
        beta = r_new.T @ r_new / (r.T @ r)
        p = r_new + beta * p
        r = r_new
        history.append(x.copy())
    return x, np.array(history)


x0 = np.zeros(2)
x_gd, hist_gd = gradient_descent(x0)
x_cg, hist_cg = conjugate_gradient(x0)


print("Прогноз активності (GD):", x_gd)
print("Прогноз активності (CG):", x_cg)


plt.plot(hist_gd[:,0], hist_gd[:,1], 'o-', label='Gradient Descent')
plt.plot(hist_cg[:,0], hist_cg[:,1], 's-', label='Conjugate Gradient')
plt.xlabel('Коефіцієнт переглядів')
plt.ylabel('Коефіцієнт завантажень')
plt.title('Оптимізація моделі активності користувачів')
plt.legend()
plt.show()
