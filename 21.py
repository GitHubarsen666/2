import numpy as np
import matplotlib.pyplot as plt

pop_size = 1000
num_features = 2  # перегляди та завантаження

# 1. Рандомна ініціалізація
random_pop = np.random.randint(0, 500, (pop_size, num_features))

# 2. Експертна ініціалізація (на основі історичних даних)
expert_pop = np.random.normal(loc=[300, 200], scale=[80, 60], size=(pop_size, num_features))
expert_pop = np.clip(expert_pop, 0, 500).astype(int)

# 3. Гібридна ініціалізація (50% рандом, 50% експерт)
hybrid_pop = np.vstack([random_pop[:pop_size//2], expert_pop[:pop_size//2]])

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(random_pop[:,0], random_pop[:,1], color='blue', label='Random', alpha=0.6, s=20)
plt.scatter(expert_pop[:,0], expert_pop[:,1], color='green', label='Expert', alpha=0.6, s=20)
plt.scatter(hybrid_pop[:,0], hybrid_pop[:,1], color='red', label='Hybrid', alpha=0.6, s=20)
plt.xlabel('Перегляди (до 500)')
plt.ylabel('Завантаження (до 500)')
plt.title('Початкові популяції стратегій для користувачів бібліотеки\n(Розмір популяції: 1000 осіб)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.show()
