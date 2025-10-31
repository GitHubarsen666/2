import numpy as np
import matplotlib.pyplot as plt

# Параметри GA
pop_size = 1000
num_generations = 100
num_features = 5  # перегляди, завантаження, час на сайті, коментарі, лайки
mutation_rate = 0.3
crossover_rate = 0.85
elitism_count = 50

# Функція пристосованості: максимізація активності користувачів з обмеженнями
def fitness(individual):
    views, downloads, time_spent, comments, likes = individual
    # Обмеження загальних ресурсів
    total_resources = views + downloads + time_spent + comments + likes
    if total_resources > 1000:
        return 0
    # Нелінійна функція з спадаючою віддачею
    return (views**0.8 * 5 + 
            downloads**0.9 * 4 + 
            time_spent**0.7 * 3 + 
            comments**0.6 * 2 + 
            likes**0.5 * 1 -
            0.001 * total_resources**1.2)

# Ініціалізація популяції
population = np.random.randint(0, 200, (pop_size, num_features))
best_history = []
avg_history = []

for gen in range(num_generations):
    # Оцінка пристосованості
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_history.append(np.max(fitness_scores))
    avg_history.append(np.mean(fitness_scores))
   
    # Відбір (турнірний)
    tournament_size = 5
    selected = []
    for _ in range(pop_size):
        contestants = population[np.random.choice(pop_size, tournament_size, replace=False)]
        contestant_fitness = [fitness(ind) for ind in contestants]
        winner = contestants[np.argmax(contestant_fitness)]
        selected.append(winner)
    population = np.array(selected)
   
    # Схрещування (однорідне)
    offspring = []
    for i in range(0, pop_size, 2):
        if i+1 < pop_size and np.random.rand() < crossover_rate:
            p1, p2 = population[i], population[i+1]
            mask = np.random.rand(num_features) < 0.5
            child1 = np.where(mask, p1, p2)
            child2 = np.where(mask, p2, p1)
            offspring.extend([child1, child2])
        else:
            offspring.extend([population[i], population[i+1]])
   
    # Мутація (різні типи)
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_type = np.random.choice(['random', 'gaussian', 'swap'])
            if mutation_type == 'random':
                gene_idx = np.random.randint(num_features)
                offspring[i][gene_idx] = np.random.randint(0, 200)
            elif mutation_type == 'gaussian':
                gene_idx = np.random.randint(num_features)
                mutation = np.random.normal(0, 20)
                offspring[i][gene_idx] = max(0, min(199, int(offspring[i][gene_idx] + mutation)))
            else:  # swap
                idx1, idx2 = np.random.choice(num_features, 2, replace=False)
                offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
    
    population = np.array(offspring)
    
    # Елітизм
    if gen > 0:
        elite_indices = np.argsort(fitness_scores)[-elitism_count:]
        elite_individuals = population[elite_indices]
        # Замінюємо найгірших на елітних
        worst_indices = np.argsort([fitness(ind) for ind in population])[:elitism_count]
        population[worst_indices] = elite_individuals

# Вивід результатів
best_individual = population[np.argmax([fitness(ind) for ind in population])]
print("Оптимальна стратегія для користувачів бібліотеки:")
print(f"Перегляди: {best_individual[0]}")
print(f"Завантаження: {best_individual[1]}")
print(f"Час на сайті: {best_individual[2]}")
print(f"Коментарі: {best_individual[3]}")
print(f"Лайки: {best_individual[4]}")
print(f"Загальна активність: {fitness(best_individual):.2f}")

# Графік
plt.figure(figsize=(12, 6))
plt.plot(best_history, 'o-', color='blue', linewidth=2, markersize=4, label='Найкраща активність')
plt.plot(avg_history, 's-', color='red', linewidth=1, markersize=2, alpha=0.7, label='Середня активність')
plt.xlabel('Покоління')
plt.ylabel('Активність')
plt.title('Генетичний алгоритм: оптимізація активності користувачів бібліотеки\n'
          f'(Популяція: {pop_size}, Поколінь: {num_generations}, Особливостей: {num_features})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
