import numpy as np
from deap import base, creator, tools, algorithms

# Define the problem: multiplication of two numbers
def evaluate(individual):
    x, y = individual
    predicted_result = x * y
    error = abs(predicted_result - target)
    return error,

# Define the genetic algorithm parameters
pop_size = 50
gen_count = 20
cx_prob = 0.7
mut_prob = 0.2

# Define the target multiplication result
target = int(input("Choose target: "))

# Create a toolbox with the necessary functions
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, low=-10, high=10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create the initial population
population = toolbox.population(n=pop_size)

# Run the genetic algorithm
algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size*2,
                          cxpb=cx_prob, mutpb=mut_prob, ngen=gen_count, stats=None, halloffame=None)

# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
print("Best Individual:", best_individual)
print("Result:", best_individual[0] * best_individual[1])
