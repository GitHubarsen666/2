import numpy as np
import matplotlib.pyplot as plt


pop_size = 10
num_features = 2  # наприклад, перегляди та завантаження


# 1. Рандомна ініціалізація
random_pop = np.random.randint(0, 10, (pop_size, num_features))


# 2. Експертна ініціалізація (наприклад, на основі історичних даних)
expert_pop = np.array([[5,3], [6,4], [7,2], [4,5], [5,5],
                       [6,3], [5,4], [7,3], [6,5], [5,2]])


# 3. Гібридна ініціалізація (50% рандом, 50% експерт)
hybrid_pop = np.vstack([random_pop[:5], expert_pop[:5]])


# Візуалізація
plt.figure(figsize=(8,6))
plt.scatter(random_pop[:,0], random_pop[:,1], color='blue', label='Random')
plt.scatter(expert_pop[:,0], expert_pop[:,1], color='green', label='Expert')
plt.scatter(hybrid_pop[:,0], hybrid_pop[:,1], color='red', label='Hybrid')
plt.xlabel('Перегляди')
plt.ylabel('Завантаження')
plt.title('Початкові популяції стратегій для користувачів бібліотеки')
plt.legend()
plt.grid(True)
plt.show()
