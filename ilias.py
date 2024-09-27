
from joblib import Parallel, delayed
import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'dummy_demo'
solutions_dir = 'solutions'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(solutions_dir):
    os.makedirs(solutions_dir)

crossover_parameter = 0.5
mutation_probability = 0.2
size_of_pop = 100
generations = 30
n_hidden = 10
dom_l = -1
dom_u = 1
n_jobs = -1  # Use all available CPU cores
runs = 10

# Get fitness for individual
def simulate(individual, enemy):
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    fitness, player_energy, enemy_energy, time = env.play(pcont=individual)
    return fitness

# Evaluate population in parallel
def evaluate(population, enemy):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(simulate)(individual, enemy) for individual in population))

# Select the fittest
def tourn_selection(population, fitness):
    index_1, index_2 = np.random.choice(np.arange(len(population)), size=2, replace=False)
    return population[index_1] if fitness[index_1] > fitness[index_2] else population[index_2]

# Uniform Crossover
def uniform_crossover(parent_1, parent_2, crossover_parameter):
    crossover_boolean = np.random.rand(len(parent_1)) < crossover_parameter
    offspring = np.where(crossover_boolean, parent_1, parent_2)
    return offspring

# Mutation
def mutate(individual, mutation_probability):
    mutation_positions = np.random.rand(len(individual)) < mutation_probability
    mutation_change = np.random.normal(0, 1, size=len(individual))
    individual[mutation_positions] += mutation_change[mutation_positions]
    individual = np.clip(individual, dom_l, dom_u)
    return individual

# Initialize population
def initialize_population(size, number_of_variables):
    return np.random.uniform(dom_l, dom_u, (size, number_of_variables))

# Save final results (only for best params)
def save_final_results(fittest_individual, best_fitness, enemy, best_params):
    with open(f"{experiment_name}/results.txt", "w") as results_file:
        results_file.write(f"Best Parameters: Mutation {best_params['mutation_probability']}, Crossover {best_params['crossover_parameter']}\n")
        results_file.write(f"Best Fitness: {best_fitness}\n")
    
    np.savetxt(f"{experiment_name}/best.txt", fittest_individual)
    with open(f"{experiment_name}/gen.txt", "w") as gen_file:
        gen_file.write("Final best individual saved after tuning.\n")

# Evolution
def evolution(enemy, size_of_pop, generations, mutation_probability, crossover_parameter):
    env = Environment(experiment_name=experiment_name, enemies=[enemy], playermode="ai", player_controller=player_controller(n_hidden), enemymode="static", level=2, speed="fastest", visuals=False)
    amount_of_vars = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
    population = initialize_population(size_of_pop, amount_of_vars)
    fitness = evaluate(population, enemy)

    best_fitness_per_generation = []

    for generation in range(generations):
        new_pop = []

        for i in range(size_of_pop // 2):
            parent_1 = tourn_selection(population, fitness)
            parent_2 = tourn_selection(population, fitness)

            child1 = mutate(uniform_crossover(parent_1, parent_2, crossover_parameter), mutation_probability)
            child2 = mutate(uniform_crossover(parent_2, parent_1, crossover_parameter), mutation_probability)

            new_pop.extend([child1, child2])

        population = np.array(new_pop)
        fitness = evaluate(population, enemy)

        fittest_index = np.argmax(fitness)
        fittest_individual = population[fittest_index]
        best_fitness_per_generation.append(np.max(fitness))

    return fittest_individual, np.max(fitness), best_fitness_per_generation

# Broad search
def grid_search(enemy, size_of_pop, generations):
    best_fitness = -np.inf
    best_params = {'mutation_probability': None, 'crossover_parameter': None}

    mutation_range = np.linspace(0.1, 0.5, 5) 
    crossover_range = np.linspace(0.3, 0.7, 5) 

    for mutation_probability in mutation_range:
        for crossover_parameter in crossover_range:
            print(f"Testing mutation: {mutation_probability}, crossover: {crossover_parameter}")
            fittestgrid, max_fitness, bestfitgengrid = evolution(enemy, size_of_pop, generations, mutation_probability, crossover_parameter)

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_params = {'mutation_probability': mutation_probability, 'crossover_parameter': crossover_parameter}

    return best_params, best_fitness

# Fine search
def fine_tune_search(enemy, size_of_pop, generations, best_params):
    best_fitness = -np.inf
    fine_mutation_range = np.linspace(best_params['mutation_probability'] - 0.05, best_params['mutation_probability'] + 0.05, 5)
    fine_crossover_range = np.linspace(best_params['crossover_parameter'] - 0.05, best_params['crossover_parameter'] + 0.05, 5)

    for mutation_probability in fine_mutation_range:
        for crossover_parameter in fine_crossover_range:
            print(f"Fine-tuning mutation: {mutation_probability}, crossover: {crossover_parameter}")
            fittest_individual, max_fitness, bestfitgentune = evolution(enemy, size_of_pop, generations, mutation_probability, crossover_parameter)

            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = fittest_individual
                best_params = {'mutation_probability': mutation_probability, 'crossover_parameter': crossover_parameter}

    return best_params, best_fitness, best_individual

# Main function
def run():
    all_best_individuals = []
    run_results = []

    for enemy in [3]:
        print(f'\nResults for enemy {enemy}\n')

        for run in range(runs):
            print(f"Run {run+1}/{runs}")

            print("Broad parameter search has started")
            best_params, bestfitgridsearch = grid_search(enemy, size_of_pop, generations)

            print("Fine-tuning search has started")
            best_params, best_fitness, best_individual = fine_tune_search(enemy, size_of_pop, generations, best_params)

            print(f"Run {run+1} - Best parameters after fine-tuning: {best_params}, with fitness: {best_fitness}")
            run_results.append((best_params, best_fitness, best_individual))

    # Calculate mean
    param_fitness_means = {}

    for params, fitness, individual in run_results:
        param_key = (params['mutation_probability'], params['crossover_parameter'])

        if param_key in param_fitness_means:
            param_fitness_means[param_key].append(fitness)
        else:
            param_fitness_means[param_key] = [fitness]

    # Get parameters
    best_mean_params = max(param_fitness_means.items(), key=lambda x: np.mean(x[1]))[0]
    best_mean_fitness = np.mean(param_fitness_means[best_mean_params])

    # Find best individual
    for params, fitness, individual in run_results:
        if (params['mutation_probability'], params['crossover_parameter']) == best_mean_params:
            best_individual = individual
            break

    print(f"\nBest parameters based on mean fitness: Mutation {best_mean_params[0]}, Crossover {best_mean_params[1]} with mean fitness: {best_mean_fitness}")
    
    # Save the final results
    save_final_results(best_individual, best_mean_fitness, enemy, {'mutation_probability': best_mean_params[0], 'crossover_parameter': best_mean_params[1]})

run()