import numpy as np
import matplotlib.pyplot as plt

# Генерація більшого набору даних (500 точок)
np.random.seed(42)
n_samples = 500

# Генерація реальних даних: перегляди від 0 до 100, завантаження від 0 до 20
views = np.random.randint(0, 100, n_samples)
downloads = np.random.randint(0, 20, n_samples)

X = np.column_stack([views, downloads])

# Створення цільової змінної з реальними залежностями та шумом
Y = 0.8 * views + 1.5 * downloads + np.random.normal(0, 5, n_samples)

# Матриця A для квадратної функції похибки (для спряженого градієнта)
A = X.T @ X
b = X.T @ Y

# Градієнт
def grad(x):
    return A @ x - b

# Оптимізований Gradient Descent
def gradient_descent(x0, lr=0.000001, tol=1e-8, max_iter=2000):
    x = x0.copy()
    history = [x.copy()]
    for i in range(max_iter):
        g = grad(x)
        x_new = x - lr * g
       
        # Перевірка збіжності
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            history.append(x.copy())
            break
           
        x = x_new
        history.append(x.copy())
       
        # Адаптивне зменшення learning rate
        if i % 100 == 0 and i > 0:
            lr *= 0.95
           
    return x, np.array(history)

# Оптимізований Conjugate Gradient
def conjugate_gradient(x0, tol=1e-10, max_iter=100):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    history = [x.copy()]
   
    for i in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
       
        history.append(x.copy())
       
        if np.linalg.norm(r_new) < tol:
            break
           
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
       
    return x, np.array(history)

# Початкова точка
x0 = np.zeros(2)

# Запуск оптимізації
x_gd, hist_gd = gradient_descent(x0)
x_cg, hist_cg = conjugate_gradient(x0)

print("Прогноз активності (GD):", x_gd)
print("Прогноз активності (CG):", x_cg)

# Створення графіка з покращеним виглядом
plt.figure(figsize=(12, 8))

# Графік траєкторій оптимізації
plt.plot(hist_gd[:, 0], hist_gd[:, 1], 'o-', markersize=3, linewidth=1.5,
         alpha=0.7, label='Gradient Descent', color='blue')
plt.plot(hist_cg[:, 0], hist_cg[:, 1], 's-', markersize=4, linewidth=2,
         alpha=0.8, label='Conjugate Gradient', color='red')

# Позначення початкової та кінцевої точок
plt.plot(hist_gd[0, 0], hist_gd[0, 1], 'ko', markersize=8, label='Початкова точка')
plt.plot(hist_gd[-1, 0], hist_gd[-1, 1], 'bo', markersize=8, label='Кінцева точка (GD)')
plt.plot(hist_cg[-1, 0], hist_cg[-1, 1], 'ro', markersize=8, label='Кінцева точка (CG)')

plt.xlabel('Коефіцієнт переглядів', fontsize=12)
plt.ylabel('Коефіцієнт завантажень', fontsize=12)
plt.title('Оптимізація моделі активності користувачів', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
