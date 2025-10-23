import numpy as np
import matplotlib.pyplot as plt


num_nodes = 6  # наприклад, 6 типів ресурсів: сторінка курсу, лекція, книга, відео, тест, форум
dist_matrix = np.random.randint(1, 10, (num_nodes, num_nodes)).astype(float)
np.fill_diagonal(dist_matrix, np.inf)  # щоб мураха не поверталась у той самий вузол


num_ants = 10          # кількість мурах (користувачів, які шукають оптимальний шлях)
num_iterations = 30    # кількість ітерацій
alpha = 1              # вплив феромону
beta = 2               # вплив відстані (чим коротше — тим краще)
rho = 0.5              # коефіцієнт випаровування феромонів


# Початкові феромони (всі шляхи мають однакову "привабливість")
pheromone = np.ones((num_nodes, num_nodes))


best_path = None
best_cost = np.inf
best_costs = []


for iteration in range(num_iterations):
    all_paths = []
    all_costs = []
   
    for ant in range(num_ants):
        start = np.random.randint(num_nodes)
        path = [start]
        unvisited = list(set(range(num_nodes)) - {start})
        cost = 0
       
        while unvisited:
            i = path[-1]
            probs = []
            for j in unvisited:
                tau = pheromone[i][j] ** alpha
                eta = (1.0 / dist_matrix[i][j]) ** beta
                probs.append(tau * eta)
            probs = np.array(probs) / np.sum(probs)
            next_node = np.random.choice(unvisited, p=probs)
            cost += dist_matrix[i][next_node]
            path.append(next_node)
            unvisited.remove(next_node)
       
        all_paths.append(path)
        all_costs.append(cost)
       
        if cost < best_cost:
            best_cost = cost
            best_path = path


    pheromone *= (1 - rho)
    for path, cost in zip(all_paths, all_costs):
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += 1.0 / cost


    best_costs.append(best_cost)


print("Найкращий знайдений маршрут (послідовність дій користувача):", best_path)
print("Загальна вартість (оптимальність шляху):", best_cost)


plt.plot(best_costs, color='purple', marker='o', linewidth=2)
plt.xlabel("Ітерація")
plt.ylabel("Вартість (довжина маршруту)")
plt.title("Оптимізація маршруту користувача у бібліотеці (Ant Colony Optimization)")
plt.grid(True)
plt.show()
