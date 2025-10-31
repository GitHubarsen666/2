import numpy as np
import matplotlib.pyplot as plt

# Параметри популяції
pop_size = 1000
num_generations = 50

# Ускладнена функція цілі з обмеженням ресурсів
def fitness(individual):
    x, y = individual
    if x + y > 150:  # Обмеження ресурсів
        return 0
    return 5*x**0.8 + 3*y**0.9 - 0.01*x**2 - 0.005*y**2

# Ініціалізація популяції
population = np.random.randint(0, 100, (pop_size, 2))
best_fitness_history = []

for gen in range(num_generations):
    # Обчислення пристосованості
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_fitness_history.append(np.max(fitness_scores))
   
    # Відбір кращих 50%
    sorted_indices = np.argsort(fitness_scores)[::-1]
    selected = population[sorted_indices[:pop_size//2]]
    
    # Схрещування з мутаціями
    offspring = []
    while len(offspring) < pop_size//2:
        parents = selected[np.random.choice(len(selected), 2, replace=False)]
        child = (parents[0] + parents[1]) // 2
        # Мутації
        if np.random.random() < 0.3:
            child += np.random.randint(-5, 6, size=2)
            child = np.clip(child, 0, 100)
        offspring.append(child)
    
    population = np.vstack([selected, offspring])

best = population[np.argmax([fitness(ind) for ind in population])]
print("Оптимальна стратегія:", best)
print("Максимальна активність:", fitness(best))

# ГРАФІК
plt.figure(figsize=(12, 6))
plt.plot(best_fitness_history, 'o-', color='#2E86AB', linewidth=3, markersize=6)
plt.xlabel('Покоління', fontsize=12)
plt.ylabel('Активність', fontsize=12)
plt.title('Еволюційний пошук оптимальної стратегії', fontsize=14)
plt.grid(True, alpha=0.3)

# Анотація з результатами
result_text = f'Оптимальне рішення:\nПерегляди: {best[0]}\nЗавантаження: {best[1]}\nАктивність: {fitness(best):.1f}'
plt.annotate(result_text, xy=(0.7, 0.3), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
             fontsize=11)

plt.tight_layout()
plt.show()
