import numpy as np
import matplotlib.pyplot as plt


# Параметри GA
pop_size = 10
num_generations = 20
num_features = 2
mutation_rate = 0.2
crossover_rate = 0.8


# Функція пристосованості: максимізація активності користувачів
def fitness(individual):
    return individual[0]*5 + individual[1]*3


# Ініціалізація популяції
population = np.random.randint(0, 10, (pop_size, num_features))
best_history = []


for gen in range(num_generations):
    # Оцінка пристосованості
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_history.append(np.max(fitness_scores))
   
    # Відбір (турнірний)
    sorted_idx = np.argsort(fitness_scores)[::-1]
    population = population[sorted_idx]
    parents = population[:pop_size//2]
   
    # Схрещування
    offspring = []
    while len(offspring) < pop_size//2:
        if np.random.rand() < crossover_rate:
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            point = np.random.randint(1, num_features)
            child = np.hstack([p1[:point], p2[point:]])
            offspring.append(child)
    offspring = np.array(offspring)
   
    # Мутація
    for child in offspring:
        if np.random.rand() < mutation_rate:
            gene_idx = np.random.randint(num_features)
            child[gene_idx] = np.random.randint(0, 10)
   
    # Формування нового покоління (елітизм 1)
    population = np.vstack([population[:1], offspring, population[1+len(offspring):]])


# Вивід результатів
best_individual = population[np.argmax([fitness(ind) for ind in population])]
print("Оптимальна стратегія для користувачів бібліотеки:", best_individual)


# Графік
plt.plot(best_history, 'o-', color='blue')
plt.xlabel('Покоління')
plt.ylabel('Найкраща активність')
plt.title('GA: зміна найкращої пристосованості')
plt.grid(True)
plt.show()
