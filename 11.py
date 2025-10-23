import numpy as np
import matplotlib.pyplot as plt


# Параметри популяції
pop_size = 10
num_generations = 20
num_features = 2


# Симульована функція цілі: максимізувати активність користувачів
def fitness(individual):
    return individual[0] * 5 + individual[1] * 3


# Ініціалізація популяції
population = np.random.randint(0, 5, (pop_size, num_features))
best_fitness_history = []


for gen in range(num_generations):
    # Обчислення пристосованості
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_fitness_history.append(np.max(fitness_scores))
   
    # Відбір кращих 50%
    sorted_indices = np.argsort(fitness_scores)[::-1]
    population = population[sorted_indices[:pop_size//2]]
   
    # Схрещування: генеруємо нових нащадків
    offspring = []
    while len(offspring) < pop_size//2:
        parents = population[np.random.choice(pop_size//2, 2, replace=False)]
        child = (parents[0] + parents[1]) // 2
        offspring.append(child)
    population = np.vstack([population, offspring])


best = population[np.argmax([fitness(ind) for ind in population])]
print("Оптимальна стратегія розподілу ресурсів:", best)


# Графік розвитку найкращої пристосованості
plt.plot(best_fitness_history, 'o-', color='blue')
plt.xlabel('Покоління')
plt.ylabel('Найкраща активність')
plt.title('Еволюційний пошук: зміна найкращої пристосованості')
plt.grid(True)
plt.show()
