import numpy as np
import matplotlib.pyplot as plt


# Створення матриці відстаней
num_nodes = 6
dist_matrix = np.random.randint(1, 10, (num_nodes, num_nodes)).astype(float)
np.fill_diagonal(dist_matrix, np.inf)


# Параметри алгоритму
n_ants = 10
n_iterations = 50
alpha = 1     # вплив феромонів
beta = 2      # вплив відстані
rho = 0.5     # коефіцієнт випаровування
pheromone = np.ones_like(dist_matrix)  # початковий рівень феромонів


best_costs = []


def route_length(route):
    return sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))


# Основний цикл ACO
for iteration in range(n_iterations):
    all_routes = []
    all_costs = []


    for ant in range(n_ants):
        start = np.random.randint(num_nodes)
        route = [start]
        visited = {start}


        while len(route) < num_nodes:
            current = route[-1]
            probs = []
            for j in range(num_nodes):
                if j not in visited:
                    tau = pheromone[current, j] ** alpha
                    eta = (1.0 / dist_matrix[current, j]) ** beta
                    probs.append(tau * eta)
                else:
                    probs.append(0)
            probs = np.array(probs) / np.sum(probs)
            next_node = np.random.choice(range(num_nodes), p=probs)
            route.append(next_node)
            visited.add(next_node)


        route.append(start)  # повернення до початку
        cost = route_length(route)
        all_routes.append(route)
        all_costs.append(cost)


    # Оновлення феромонів
    pheromone *= (1 - rho)
    for route, cost in zip(all_routes, all_costs):
        for i in range(len(route) - 1):
            pheromone[route[i], route[i + 1]] += 1.0 / cost


    best_costs.append(min(all_costs))


# Візуалізація збіжності
plt.plot(best_costs)
plt.title("ACO — збіжність оптимізації маршруту")
plt.xlabel("Ітерації")
plt.ylabel("Найкраща вартість")
plt.show()
