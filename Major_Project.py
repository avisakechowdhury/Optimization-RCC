import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Define problem parameters
C = [3200, 4000, 4800, 5000] # Cost of concrete per m3
S = [45, 50, 55] # Cost of steel per kg
F = 200 # Cost of formwork per m2
L = [3, 4, 5, 6, 7, 8, 9, 10] # Span of beam
LL = [25, 35, 45, 50, 60] # Live load
dc = 50 # Effective cover
fy = [415, 500, 550] # Characteristics strength of steel
fck = [20, 25, 30, 35] # Characteristics strength of concrete

# Define design variables
b = np.arange(150, 600 + 25, 25) # Width of beam
d = np.arange(250, 1000 + 25, 25) # Depth of beam
dia1 = [16, 20, 25] # Diameter of bars for steel in tension zone
bars1 = [4, 5, 6, 7, 8, 9, 10] # No of bars for steel in tension zone
dia2 = [16, 20, 25] # Diameter of bars for steel in compression zone
bars2 = [4, 5, 6, 7, 8, 9, 10] # No of bars for steel in compression zone
dia3 = [8, 10, 12, 16, 20] # Diameter of bars for shear reinforcement
sv = [180, 200, 220, 240, 260, 280, 300] # Spacing for shear reinforcement

# Define objective function
def objective_function(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    V = LL[-1] # Select the maximum live load
    As1 = np.pi * x3**2 * x4 / 4 # Area of steel in tension zone
    As2 = np.pi * x5**2 * x6 / 4 # Area of steel in compression zone
    Vc = 0.5 * 0.45 * fck[-1] * x1 * x2 # Shear capacity of concrete
    Vs = As1 * fy[-1] / (0.87 * Vc) # Shear capacity of steel in tension zone
    Vc1 = 0.8 * 0.45 * fck[-1] * (x1 - dc - x3 / 2) * x2 # Shear capacity of concrete in compression zone
    if abs(Vc1) < 1e-6: # check if Vc1 is close to zero
    	Vs1 = 0.0 # set Vs1 to zero
    else:
    	Vs1 = As2 * fy[-1] / (0.87 * Vc1) # calculate Vs1 normally # Shear capacity of steel in compression zone
    Vsv = 0.87 * dia3[-1]**2 * np.pi * (L[-1] / x8 + 1) * (2 * x1 + 2 * x2) / (sv[-1] * 1000) # Shear capacity of shear reinforcement
    Vu = V + Vs + Vs1 + Vsv # Total shear force
    Mu = Vu * L[-1] / 2 # Bending moment

    # Calculate cost
    Cc = C[-1] * x1 * x2 * L[-1] / (1000 * 100)
    #Cc = np.float64(Cc)
    Cs = S[-1] * (As1 + As2) * L[-1] / 1000
    Cf = F * (x1 * L[-1] + 2 * x1 * x2 + 2 * x2 * L[-1]) / 100000
    Ctotal = Cc + Cs + Cf
    return Ctotal,

# Define constraints
def constraint1(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    As1 = np.pi * x3**2 * x4 / 4 # Area of steel in tension zone
    As2 = np.pi * x5**2 * x6 / 4 # Area of steel in compression zone

    # Check for constraints
    if Mu <= 0:
        return False
    else:
        return True

# Create the Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("x1", np.random.choice, b)
toolbox.register("x2", np.random.choice, d)
toolbox.register("x3", np.random.choice, dia1)
toolbox.register("x4", np.random.choice, bars1)
toolbox.register("x5", np.random.choice, dia2)
toolbox.register("x6", np.random.choice, bars2)
toolbox.register("x7", np.random.choice, dia3)
toolbox.register("x8", np.random.choice, sv)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.x1, toolbox.x2, toolbox.x3, toolbox.x4, toolbox.x5, toolbox.x6, toolbox.x7, toolbox.x8), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxTwoPoint )
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set population size
population_size = 100

# Run the Genetic Algorithm
number_of_generations = 50
crossover_probability = 0.6
mutation_probability = 0.2

population = toolbox.population(n=population_size)
fitness_values = []
best_individuals = []

for generation in range(number_of_generations):
    offspring = list(algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability))
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    population.sort(key=lambda x: x.fitness.values)
    fitness_values.append(population[0].fitness.values)
    best_individuals.append(population[0])

# Visualize results
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
print("Best fitness value:", best_individual.fitness.values[0])

plt.figure()
plt.plot(range(len(fitness_values)), fitness_values)
plt.title("Evolution of Best Fitness Value")
plt.xlabel("Generation")
plt.ylabel("Fitness Value")
plt.show()

design_variables = [
    "Width of beam", "Depth of beam",
    "Diameter of bars for steel in tension zone",
    "No of bars for steel in tension zone",
    "Diameter of bars for steel in compression zone",
    "No of bars for steel in compression zone",
    "Diameter of bars for shear reinforcement", "Spacing for shear reinforcement"
]

# Extract the values of the best individual
x_values = list(range(len(best_individual)))
y_values = [best_individual[i] for i in range(len(best_individual))]

# Plot the results
plt.figure()
plt.bar(x_values, y_values)
plt.xticks(x_values, design_variables, rotation=45)
plt.title("Optimized Design Variables")
plt.xlabel("Design Variables")
plt.ylabel("Value")
plt.show()
