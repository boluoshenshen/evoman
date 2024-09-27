import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from joblib import Parallel, delayed

n_jobs = -1  # Use all available CPU cores
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy" 

# Initialize the population
def initialize_population(size, number_of_variables, dom_l, dom_u):
    return np.random.uniform(dom_l, dom_u, (size, number_of_variables))

# Evaluate population in parallel
def evaluate(population, enemy, fitness_mode, n_hidden, experiment_name):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(simulate)(individual, enemy, fitness_mode, n_hidden, experiment_name) for individual in population))

#Simulation function, used to calculate the fitness of individuals
def simulate(individual, enemy, fitness_mode, n_hidden, experiment_name):
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      randomini="yes")
    fitness, player_energy, enemy_energy, time = env.play(pcont=individual)
    
    # Choose different formulas according to fitness mode
    if fitness_mode == 1:
        gamma = 0.9
        alpha = 0.1
    elif fitness_mode == 2:
        gamma = 0.1
        alpha = 0.9

    fitness_value = gamma * (100 - enemy_energy) + alpha * player_energy - np.log(time)
    return fitness_value

# Calculate individual gain
def individual_gain(individual, enemy, n_hidden, experiment_name):
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      randomini="yes")
    fitness, player_energy, enemy_energy, time = env.play(pcont=individual)
    gain_value = player_energy - enemy_energy
    return gain_value

# Tournament Selection for GA
def tourn_selection(population, fitness):
    index_1, index_2 = np.random.choice(np.arange(len(population)), size=2, replace=False)
    return population[index_1] if fitness[index_1] > fitness[index_2] else population[index_2]

# Uniform crossover for GA
def uniform_crossover(parent_1, parent_2, crossover_parameter):
    crossover_boolean = np.random.rand(len(parent_1)) < crossover_parameter
    offspring = np.where(crossover_boolean, parent_1, parent_2)
    return offspring

# Mutations for GA
def mutate_GA(individual, mutation_probability, dom_l, dom_u):
    mutation_positions = np.random.rand(len(individual)) < mutation_probability
    mutation_change = np.random.normal(0, 1, size=len(individual))
    individual[mutation_positions] += mutation_change[mutation_positions]
    individual = np.clip(individual, dom_l, dom_u)
    return individual

# Differential mutation operation
def mutation(population, F):
    mutant_population = np.zeros_like(population)
    for i in range(len(population)):
        indices = np.random.choice(range(len(population)), 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices[0]], population[indices[1]], population[indices[2]]
        mutant_vector = x_r1 + F * (x_r2 - x_r3)
        mutant_population[i] = np.clip(mutant_vector, -1, 1)  # Keep within range [-1, 1]
    return mutant_population

# Differential crossover operation
def crossover(parent, mutant, CR):
    trial = np.copy(parent)
    for j in range(len(parent)):
        if np.random.rand() < CR or j == np.random.randint(0, len(parent)):
            trial[j] = mutant[j]
    return trial

# Differential selection operation
def selection(population, trial_population, fitness, trial_fitness):
    for i in range(len(population)):
        if trial_fitness[i] > fitness[i]:
            population[i] = trial_population[i]
            fitness[i] = trial_fitness[i]
    return population, fitness

# Differential evolution process
def evolution_DE(best_parameters, fitness_mode, experiment_name, training_enemy):
    size_of_pop = best_parameters["size_of_pop"]
    generations = best_parameters["generations"]
    F = best_parameters.get("F", 0.5)  # Differential evolution parameter F
    CR = best_parameters.get("CR", 0.9)  # Differential evolution crossover probability
    n_hidden = best_parameters["n_hidden"]
    dom_l = best_parameters["dom_l"]
    dom_u = best_parameters["dom_u"]

    # Initialize the environment, only for training_enemy
    env = Environment(experiment_name=experiment_name, enemies=[training_enemy], playermode="ai", 
                      player_controller=player_controller(n_hidden), enemymode="static", level=2, 
                      speed="fastest", visuals=False,randomini="yes")

    # Initialize the population
    amount_of_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
    population = initialize_population(size_of_pop, amount_of_vars, dom_l, dom_u)
    fitness = evaluate(population, training_enemy, fitness_mode, n_hidden, experiment_name)

    for generation in range(generations):
        # Differential mutation and crossover
        mutant_population = mutation(population, F)
        trial_population = np.array([crossover(population[i], mutant_population[i], CR) for i in range(size_of_pop)])
        trial_fitness = evaluate(trial_population, training_enemy, fitness_mode, n_hidden, experiment_name)

        # Select individuals with higher fitness
        population, fitness = selection(population, trial_population, fitness, trial_fitness)

        # Calculate individual gains
        gains = np.array([individual_gain(individual, training_enemy, n_hidden, experiment_name) for individual in population])

        # Save the fitness_values ​​and individual_gains of the current generation
        eval_folder = f"{experiment_name}/generation_{generation}_evaluation_enemy_{training_enemy}"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        np.savetxt(f"{eval_folder}/fitness_values.txt", fitness)
        np.savetxt(f"{eval_folder}/individual_gains.txt", gains)

        # Save the best individual of each generation
        best_individual = population[np.argmax(fitness)]
        np.savetxt(f"{eval_folder}/best_solution_gen_{generation}.txt", best_individual)

    # Return the best individual
    fittest_index = np.argmax(fitness)
    fittest_individual = population[fittest_index]
    return fittest_individual



# GA evolution process 
def evolution_GA(best_parameters, fitness_mode, experiment_name, training_enemy):
    size_of_pop = best_parameters["size_of_pop"]
    generations = best_parameters["generations"]
    mutation_probability = best_parameters["best_mutation_probability"]
    crossover_parameter = best_parameters["best_crossover_parameter"]
    n_hidden = best_parameters["n_hidden"]
    dom_l = best_parameters["dom_l"]
    dom_u = best_parameters["dom_u"]

    # Initialize the environment, only for training_enemy
    env = Environment(experiment_name=experiment_name, enemies=[training_enemy], playermode="ai", 
                      player_controller=player_controller(n_hidden), enemymode="static", level=2, 
                      speed="fastest", visuals=False,randomini="yes")

    amount_of_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
    population = initialize_population(size_of_pop, amount_of_vars, dom_l, dom_u)

    for generation in range(generations):
        # Calculate fitness and individual_gain for training_enemy
        fitness_values = evaluate(population, training_enemy, fitness_mode, n_hidden, experiment_name)
        gains = np.array([individual_gain(individual, training_enemy, n_hidden, experiment_name) for individual in population])

        # Save the fitness_values ​​and individual_gains of the current generation
        eval_folder = f"{experiment_name}/generation_{generation}_evaluation_enemy_{training_enemy}"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        np.savetxt(f"{eval_folder}/fitness_values.txt", fitness_values)
        np.savetxt(f"{eval_folder}/individual_gains.txt", gains)

        # Selection, crossover and mutation
        new_pop = []
        for i in range(size_of_pop // 2):
            parent_1 = tourn_selection(population, fitness_values)
            parent_2 = tourn_selection(population, fitness_values)
            child1 = mutate_GA(uniform_crossover(parent_1, parent_2, crossover_parameter), mutation_probability, dom_l, dom_u)
            child2 = mutate_GA(uniform_crossover(parent_2, parent_1, crossover_parameter), mutation_probability, dom_l, dom_u)
            new_pop.extend([child1, child2])

        population = np.array(new_pop)
        fitness_values = evaluate(population, training_enemy, fitness_mode, n_hidden, experiment_name)  # Recalculate fitness after mutation
        best_individual = population[np.argmax(fitness_values)]
        np.savetxt(f"{eval_folder}/best_solution_gen_{generation}.txt", best_individual)
    
        fittest_index = np.argmax(fitness_values)
        fittest_individual = population[fittest_index]

    return fittest_individual



# Main function, run the experiment
def run(best_parameters, EA_variable, fitness_mode, run_id, training_enemy):
    # Set up basic experiment folder
    base_folder = 'Group_96_data_for_plots'
    algorithm_folder = f"{base_folder}/{EA_variable}"
    fitness_folder = f"{algorithm_folder}/fitness_mode_{fitness_mode}"
    
    
    os.makedirs(fitness_folder, exist_ok=True)

    # Set the run directory
    run_folder = f"{fitness_folder}/enemy_{training_enemy}/run{run_id}" 
    # Create a separate folder for each enemy
    os.makedirs(run_folder, exist_ok=True)

    # Choose to use GA or DE algorithm
    if EA_variable == "GA":
        fittest_individual = evolution_GA(best_parameters, fitness_mode, run_folder, training_enemy)
    elif EA_variable == "DE":
        fittest_individual = evolution_DE(best_parameters, fitness_mode, run_folder, training_enemy)
    else:
        raise ValueError(f"Unknown EA_variable: {EA_variable}")
    
    # Save the best individual
    np.savetxt(f"{run_folder}/best_individual.txt", fittest_individual)
    print(f"\nBest individual saved for {EA_variable} with fitness_mode={fitness_mode} in run{run_id} for enemy {training_enemy}")

# Main function, run 10 experiments
def repeat_experiments(best_parameters, EA_variable, fitness_mode, runtime):
    # Iterate through each training enemy
    for training_enemy in [3,4,5]:  
        for run_id in range(1, runtime + 1):  # Multiple experiments per enemy
            run(best_parameters, EA_variable, fitness_mode, run_id, training_enemy)


def set_best_parameters(EA_variable, fitness_mode):
    best_mutation_probability = 0 # For GA
    best_crossover_parameter = 0  # For GA
    F = 0 # For DE
    CR = 0 # For DE
    # fitness_value = gamma * (100 - enemy_energy) + alpha * player_energy - np.log(time)
    # fitness_mode = 1 : gamma = 0.9 alpha = 0.1
    # fitness_mode = 2 : gamma = 0.1 alpha = 0.
    if EA_variable == "GA" :
        if fitness_mode == 1: 
            best_mutation_probability = 0.0275
            best_crossover_parameter = 0.65 

        if fitness_mode == 2: 
            best_mutation_probability = 0.0975
            best_crossover_parameter = 0.675

    if EA_variable == "DE" :
        if fitness_mode == 1:
            F = 0.13
            CR = 0.97
        if fitness_mode == 2:
            F = 0.075
            CR = 0.925

    # Example
    best_parameters = {
        "best_mutation_probability": best_mutation_probability, # For GA
        "best_crossover_parameter": best_crossover_parameter,  # For GA
        "F": F,  # For DE
        "CR": CR,  # For DE
        "size_of_pop": 100,
        "generations": 30,
        "n_hidden": 10,
        "dom_l": -1,
        "dom_u": 1,
        "n_jobs": -1,
    }

    return best_parameters

# SET EA ,FITNESS MODE and runtime
EA_variable = "GA"
fitness_mode = 1
best_parameters = set_best_parameters(EA_variable, fitness_mode)
runtime = 10
# Choose GA or DE for testing
repeat_experiments(best_parameters, EA_variable, fitness_mode, runtime)

