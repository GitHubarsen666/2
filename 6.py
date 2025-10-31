import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Параметри оптимізації
c = [-5, -3]  # Цільова функція
A = [[1, 1], [0, 1]]  # Обмеження
b = [500, 1000]  # Ліміти

# Розв'язок
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')
optimal_x, optimal_y = res.x
optimal_value = -res.fun

print("Оптимальні рекомендації:", res.x)
print("Максимальна активність:", optimal_value)

# Графік
plt.figure(figsize=(10, 6))

# Область допустимих рішень
x = np.linspace(0, 600, 100)
plt.fill_between(x, 0, np.minimum(1000, 500 - x), alpha=0.3, label='Допустима область')

# Лінії обмежень
plt.plot(x, 500 - x, 'r-', label='x + y ≤ 500')
plt.axhline(1000, color='g', linestyle='--', label='y ≤ 1000')

# Оптимальна точка
plt.plot(optimal_x, optimal_y, 'ro', markersize=8, label=f'Оптимум ({optimal_x:.0f}, {optimal_y:.0f})')

plt.xlabel('Перегляди')
plt.ylabel('Завантаження')
plt.title('Оптимізація активності користувачів')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 600)
plt.ylim(0, 1200)
plt.show()
