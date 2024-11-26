import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import pandas as pd


def boot_function(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

def boot_gradient(x):
    return np.array([0.52*x[0] - 0.48*x[1], -0.48*x[0]+0.52*x[1]])

class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def create_population(self):
        return np.random.uniform(-10, 10, (self.population_size, 2))

    def fitness(self, population):
        return np.apply_along_axis(boot_function, 1, population)

    def select_parents(self, population, fitness_values):
        selected_indices = np.argsort(fitness_values)[:self.population_size // 2]
        return population[selected_indices]

    def crossover(self, parent1, parent2):
        return np.clip((parent1 + parent2) / 2, -10, 10)

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual += np.random.normal(0, 0.5, 2)
        return individual

    def run(self):
        population = self.create_population()
        best_fitness_values = []
        global_best = float('inf')  # Инициализация глобального лучшего значения
        for generation in range(self.generations):
            fitness_values = self.fitness(population)
            generation_best = np.min(fitness_values)
            global_best = min(global_best, generation_best)  # Обновляем глобальный минимум
            best_fitness_values.append(global_best)  # Добавляем накопительный минимум

            parents = self.select_parents(population, fitness_values)
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = parents[np.random.choice(len(parents), size=2, replace=False)]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))

            population = np.array(new_population)

        return global_best, best_fitness_values


def optimize_with_newton(iterations):
    results = []
    for _ in range(iterations):
        start_time = time.time()
        result = minimize(boot_function, [-10, 10], method='Newton-CG', jac=boot_gradient)
        end_time = time.time()
        results.append((result.fun, end_time - start_time))
    return results

iterations = 100
newton_results = optimize_with_newton(iterations)
newton_values, newton_times = zip(*newton_results)

genetic_algorithm = GeneticAlgorithm(population_size=50, generations=100, mutation_rate=0.1)
genetic_results = []
genetic_times = []

for _ in range(iterations):
    start_time = time.time()
    value, best_fitness_values = genetic_algorithm.run()
    end_time = time.time()
    genetic_results.append(value)
    genetic_times.append(end_time - start_time)

def compute_statistics(data):
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance

newton_mean, newton_variance = compute_statistics(newton_values)
genetic_mean, genetic_variance = compute_statistics(genetic_results)

newton_time_mean, newton_time_variance = compute_statistics(newton_times)
genetic_time_mean, genetic_time_variance = compute_statistics(genetic_times)

X = np.linspace(-10, 10, 400)
Y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(X, Y)
Z = boot_function([X, Y])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax.set_title('Функция Матьяса')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('f(X1, X2)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(best_fitness_values, label='Значения функции соответствия (ГА)', color='orange')
plt.title('Значения функции соответствия в зависимости от числа поколений (ГА)')
plt.xlabel('Поколение')
plt.ylabel('Значение функции соответствия')
plt.legend()
plt.grid()
plt.show()

print("\n==============================================================================================\n")
print("+ Результаты оптимизации:")
print(f"+ Алгоритм Ньютона: среднее значение = {newton_mean:.5f}, дисперсия = {newton_variance:.5f}, "
      f"+ время: среднее значение = {newton_time_mean:.5f}, дисперсия = {newton_time_variance:.5f}")
print(f"+ Генетический алгоритм: среднее значение = {genetic_mean:.5f}, дисперсия = {genetic_variance:.5f}, "
      f"+ время: среднее значение = {genetic_time_mean:.5f}, дисперсия = {genetic_time_variance:.5f}")

print("\n==============================================================================================\n")


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

results_table = pd.DataFrame({
    "Алгоритм": ["Алгоритм Ньютона", "Генетический алгоритм"],
    "Среднее значение": [newton_mean, genetic_mean],
    "Дисперсия": [newton_variance, genetic_variance],
    "Среднее время": [newton_time_mean, genetic_time_mean],
    "Дисперсия времени": [newton_time_variance, genetic_time_variance],
})

print(results_table)