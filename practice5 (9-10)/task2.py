import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools
from ucimlrepo import fetch_ucirepo

ionosphere = fetch_ucirepo(id=42)

X = ionosphere.data.features.dropna()
y = ionosphere.data.targets.values.flatten()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    'SVM': lambda params: SVC(C=params[0], gamma=params[1], kernel='rbf'),
    'KNN': lambda params: KNeighborsClassifier(n_neighbors=int(params[2])),
    'RF': lambda params: RandomForestClassifier(n_estimators=int(params[3]), max_depth=int(params[4]))
}


def evaluate(individual):
    C = individual[0]
    gamma = individual[1]
    n_neighbors = int(individual[2])
    n_estimators = int(individual[3])
    max_depth = int(individual[4])

    svm = classifiers['SVM']([C, gamma])
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return (accuracy,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("C", np.random.uniform, 0.1, 10)  # Вместо np.random.rand
toolbox.register("gamma", np.random.uniform, 0.01, 1)  # Вместо np.random.rand
toolbox.register("n_neighbors", np.random.randint, 1, 50)
toolbox.register("n_estimators", np.random.randint, 10, 200)
toolbox.register("max_depth", np.random.randint, 1, 20)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.C, toolbox.gamma, toolbox.n_neighbors,
                  toolbox.n_estimators, toolbox.max_depth), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Добавлено значение indpb
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    population = toolbox.population(n=10)
    NGEN = 10
    for gen in range(NGEN):
        print(f"ПОКОЛЕНИЕ {gen}")

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit if isinstance(fit, tuple) else (fit,)  # Убедитесь, что это кортеж

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

    fits = []
    for ind in population:
        if ind.fitness.values:
            fits.append(ind.fitness.values[0])

    if fits:
        best_idx = np.argmax(fits)
        best_ind = population[best_idx]
        print("Best individual is ", best_ind)
        print("With fitness value of ", fits[best_idx])
    else:
        print("No valid fitness values found.")


if __name__ == '__main__':
    main()