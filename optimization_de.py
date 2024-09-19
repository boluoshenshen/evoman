import random
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import os
import concurrent.futures

n_hidden_neurons = 10
experiment_name = "de_tuned_optimization"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize environment
env = Environment(
    experiment_name=experiment_name,
    enemies=[3],  # Set enemy number, can be changed
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False
)

n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Define the hyperparameters for differential evolution
F_min, F_max = 0.5, 1.0  # Range for mutation factor
CR_min, CR_max = 0.5, 1.0  # Range for crossover probability
population_size_min, population_size_max = 50, 200  # Range for population size
generations = 30  # Number of generations

# Fitness evaluation function: evaluates the performance of an individual (neural network weights)
def evaluate(individual):
    fitness, player_life, enemy_life, _ = env.play(pcont=individual)
    return player_life - enemy_life  # This can be changed

# Initialize population: population size is controlled by the input parameter
def initialize_population(population_size):
    population = np.random.uniform(low=-1, high=1, size=(population_size, n_vars))
    return population

# Differential mutation operation
def mutation(population, F):
    mutant_population = np.zeros_like(population)
    for i in range(len(population)):
        indices = random.sample(range(len(population)), 3)
        x_r1, x_r2, x_r3 = population[indices[0]], population[indices[1]], population[indices[2]]
        mutant_vector = x_r1 + F * (x_r2 - x_r3)
        mutant_population[i] = np.clip(mutant_vector, -1, 1)  # Keep within range [-1, 1]
    return mutant_population

# Differential crossover operation
def crossover(parent, mutant, CR):
    trial = np.copy(parent)
    for j in range(n_vars):
        if random.random() < CR or j == random.randint(0, n_vars - 1):
            trial[j] = mutant[j]
    return trial

# Differential selection operation
def selection(population, trial_population, fitness, trial_fitness):
    for i in range(len(population)):
        if trial_fitness[i] > fitness[i]:
            population[i] = trial_population[i]
            fitness[i] = trial_fitness[i]
    return population, fitness

# Differential Evolution Optimization Function: input F, CR, and population_size
def run_de_optimization(F, CR, population_size):
    # Initialize population
    population = initialize_population(population_size)
    fitness = np.array([evaluate(ind) for ind in population])
    
    # Evolution process
    for gen in range(generations):
        print(f"-- Generation {gen + 1} --")
        
        # Mutation and crossover
        mutant_population = mutation(population, F)
        trial_population = np.array([crossover(population[i], mutant_population[i], CR) for i in range(population_size)])
        trial_fitness = np.array([evaluate(trial_ind) for trial_ind in trial_population])
        
        # Select the better individuals
        population, fitness = selection(population, trial_population, fitness, trial_fitness)
        
        # Print current generation statistics
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)
        print(f"  Best Fitness: {best_fitness:.4f}, Mean Fitness: {mean_fitness:.4f}, Std: {std_fitness:.4f}")
    
    # Save the best individual locally
    best_individual = population[np.argmax(fitness)]
    np.savetxt(experiment_name + '/best_solution.txt', best_individual)
    print(f"Best individual saved with fitness: {np.max(fitness):.4f}")
    return best_fitness

def evaluate_combination(params):
    F, CR, population_size = params
    print(f"Testing F={F}, CR={CR}, Population Size={population_size}")
    fitness = run_de_optimization(F, CR, population_size)
    return fitness, params

# Hyperparameter tuning process: search for the best combination of F, CR, and population_size
def tune_hyperparameters():
    best_fitness = -np.inf
    best_params = None
    param_combinations = []

    # Generate a list of all hyperparameter combinations
    for F in np.linspace(F_min, F_max, 5):  # Search over mutation factor F
        for CR in np.linspace(CR_min, CR_max, 5):  # Search over crossover probability CR
            for population_size in range(population_size_min, population_size_max, 50):  # Search over population size
                param_combinations.append((F, CR, population_size))

    # Use ProcessPoolExecutor to run the evaluations in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_combination, params) for params in param_combinations]
        
        # Iterate through the completed futures
        for future in concurrent.futures.as_completed(futures):
            fitness, params = future.result()
            
            # Record the best hyperparameter combination
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = params
    
    # Output the best result
    print(f"Best parameters: F={best_params[0]}, CR={best_params[1]}, Population Size={best_params[2]} with fitness {best_fitness:.4f}")

# Ensure correct execution in multiprocessing contexts
if __name__ == "__main__":
    # Run hyperparameter tuning and output the best weights
    tune_hyperparameters()