import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
import os
from concurrent.futures import ProcessPoolExecutor

experiment_name = 'optimization_test_de_parallel'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",  
                  visuals=False)  

npop = 100
gens = 30
dom_u = 1
dom_l = -1
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

def initialize_population():
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))

def evaluate(x):
    fitness_values = np.array([simulation(env, ind) for ind in x])
    avg_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    return fitness_values

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

def mutate(population, F):
    mutants = np.zeros_like(population)
    for i in range(npop):
        idxs = list(range(npop))
        idxs.remove(i)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + F * (population[b] - population[c])
        mutants[i] = np.clip(mutant, dom_l, dom_u)
    return mutants

def crossover(population, mutants, CR):
    offspring = np.empty_like(population)
    for i in range(npop):
        cross_points = np.random.rand(n_vars) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, n_vars)] = True
        offspring[i] = np.where(cross_points, mutants[i], population[i])
    return offspring

def select(population, fitness, offspring, offspring_fitness, elite_individual, elite_fitness):
    combined_pop = np.vstack((population, offspring))
    combined_fit = np.concatenate((fitness, offspring_fitness))
    
    elite_fitness_array = np.array([elite_fitness])  
    combined_pop = np.vstack((combined_pop, elite_individual))
    combined_fit = np.concatenate((combined_fit, elite_fitness_array)) 
    
    best_indices = np.argsort(combined_fit)[-npop:]
    return combined_pop[best_indices], combined_fit[best_indices]

def local_search(individual, max_iterations=100, step_size=0.01):
    current = individual.copy()
    current_fitness = simulation(env, current)
    for _ in range(max_iterations):
        perturbation = np.random.randn(n_vars) * step_size
        new_individual = current + perturbation
        new_individual = np.clip(new_individual, dom_l, dom_u)
        new_fitness = simulation(env, new_individual)
        if new_fitness > current_fitness:
            current = new_individual
            current_fitness = new_fitness
    return current

def run_de_for_params(F, CR):
    pop = initialize_population()
    fit_pop = evaluate(pop)
    
    best_idx = np.argmax(fit_pop)
    elite_individual = pop[best_idx].copy()
    elite_fitness = fit_pop[best_idx]

    for generation in range(gens):
        mutants = mutate(pop, F)
        offspring = crossover(pop, mutants, CR)
        offspring_fitness = evaluate(offspring)
        
        pop, fit_pop = select(pop, fit_pop, offspring, offspring_fitness, elite_individual, elite_fitness)
        
        best_idx = np.argmax(fit_pop)
        if fit_pop[best_idx] > elite_fitness:
            elite_individual = pop[best_idx].copy()
            elite_fitness = fit_pop[best_idx]
        
        refined_individual = local_search(elite_individual)
        refined_fitness = simulation(env, refined_individual)
        if refined_fitness > elite_fitness:
            pop[best_idx] = refined_individual
            fit_pop[best_idx] = refined_fitness
            elite_individual = refined_individual
            elite_fitness = refined_fitness

    return elite_fitness, F, CR, elite_individual

def run_simulation_for_params(params):
    F, CR = params
    return run_de_for_params(F, CR)

F_values = [0.4, 0.5, 0.6, 0.7, 0.8]
CR_values = [0.4, 0.5, 0.6, 0.7, 0.8]


def run_experiment():
    best_overall_fitness = -np.inf
    best_overall_params = {}
    best_overall_individual = None

    
    for run in range(10):
        print(f'\nStarting run {run+1}...\n')

        param_combinations = [(F, CR) for F in F_values for CR in CR_values]

        with ProcessPoolExecutor() as executor:
            results = executor.map(run_simulation_for_params, param_combinations)

        best_fitness_in_run = -np.inf
        best_params_in_run = {}
        best_individual_in_run = None

        for fitness, F, CR, individual in results:
            print(f"Run {run+1}: F = {F}, CR = {CR}, Fitness = {fitness}")
            if fitness > best_fitness_in_run:
                best_fitness_in_run = fitness
                best_params_in_run = {'F': F, 'CR': CR}
                best_individual_in_run = individual


        if best_fitness_in_run > best_overall_fitness:
            best_overall_fitness = best_fitness_in_run
            best_overall_params = best_params_in_run
            best_overall_individual = best_individual_in_run.copy()

    
        np.savetxt(experiment_name + f'/best_run{run+1}.txt', best_individual_in_run)

    print('Optimization complete.')
    print(f'Best overall fitness after 10 runs: {best_overall_fitness}')
    print(f'Best parameters: F = {best_overall_params["F"]}, CR = {best_overall_params["CR"]}')
    np.savetxt(experiment_name + '/best_overall.txt', best_overall_individual)

if __name__ == '__main__':
    run_experiment()
