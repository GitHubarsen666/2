import numpy as np
import matplotlib.pyplot as plt

# Створення матриці відстаней для великої задачі
num_nodes = 100
np.random.seed(42)
dist_matrix = np.random.randint(5, 150, (num_nodes, num_nodes)).astype(float)
np.fill_diagonal(dist_matrix, np.inf)

# Параметри алгоритму
n_ants = 100
n_iterations = 100
alpha = 1.5     # вплив феромонів
beta = 3.0      # вплив відстані
rho = 0.3       # коефіцієнт випаровування
q0 = 0.7        # параметр для псевдо-випадкового правила
pheromone = np.ones_like(dist_matrix) * 0.1  # початковий рівень феромонів

best_costs = []
avg_costs = []

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
            available = [j for j in range(num_nodes) if j not in visited]
            
            if np.random.random() < q0:
                # Жадібний вибір
                next_node = min(available, 
                              key=lambda j: dist_matrix[current, j] / (pheromone[current, j] ** alpha))
            else:
                # Імовірнісний вибір
                probs = []
                for j in available:
                    tau = pheromone[current, j] ** alpha
                    eta = (1.0 / dist_matrix[current, j]) ** beta
                    probs.append(tau * eta)
                probs = np.array(probs) / np.sum(probs)
                next_node = np.random.choice(available, p=probs)
            
            route.append(next_node)
            visited.add(next_node)

        route.append(start)  # повернення до початку
        cost = route_length(route)
        all_routes.append(route)
        all_costs.append(cost)

    # Оновлення феромонів
    pheromone *= (1 - rho)
    best_route_idx = np.argmin(all_costs)
    best_route = all_routes[best_route_idx]
    best_cost = all_costs[best_route_idx]
    
    # Додаємо феромон тільки на найкращий маршрут
    for i in range(len(best_route) - 1):
        pheromone[best_route[i], best_route[i + 1]] += 1.0 / best_cost

    best_costs.append(best_cost)
    avg_costs.append(np.mean(all_costs))
    
    if iteration % 50 == 0:
        print(f"Ітерація {iteration}: найкраща вартість = {best_cost:.2f}")

# Знаходимо найкращий маршрут
final_best_idx = np.argmin(best_costs)
final_best_cost = best_costs[final_best_idx]

print(f"\nРезультати для {num_nodes} вузлів:")
print(f"Найкраща вартість маршруту: {final_best_cost:.2f}")
print(f"Кількість мурах: {n_ants}")
print(f"Кількість ітерацій: {n_iterations}")

# Візуалізація збіжності
plt.figure(figsize=(12, 6))
plt.plot(best_costs, 'b-', linewidth=2, label='Найкраща вартість')
plt.plot(avg_costs, 'r--', alpha=0.7, label='Середня вартість')
plt.title(f"ACO — оптимізація маршруту для {num_nodes} вузлів\n"
          f"(Мурах: {n_ants}, Ітерацій: {n_iterations})")
plt.xlabel("Ітерації")
plt.ylabel("Вартість маршруту")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Додаткова статистика
print(f"\nСтатистика:")
print(f"Початкова найкраща вартість: {best_costs[0]:.2f}")
print(f"Фінальна найкраща вартість: {best_costs[-1]:.2f}")
print(f"Покращення: {((best_costs[0] - best_costs[-1]) / best_costs[0] * 100):.1f}%")
print(f"Мінімальна відстань у матриці: {np.min(dist_matrix[dist_matrix != np.inf]):.1f}")
print(f"Максимальна відстань у матриці: {np.max(dist_matrix):.1f}")
