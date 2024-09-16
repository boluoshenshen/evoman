import numpy as np
import random
from evoman.environment import Environment

# 定义遗传算法的参数
population_size = 10  # 种群大小
num_generations = 20  # 代数
mutation_rate = 0.1  # 变异率
crossover_rate = 0.7  # 交叉率

# 初始化 Evoman 环境
env = Environment(enemies=[1], playermode="ai", level=2)

# 定义一个随机生成神经网络权重的函数
def generate_random_weights():
    num_sensors = env.get_num_sensors()  # 获取传感器数量
    num_actions = 5  # 你可以根据你的控制器定义动作数量
    return np.random.uniform(-1, 1, (num_sensors, num_actions))  # 生成随机权重


# 定义适应度函数（fitness function）
def fitness_function(weights):
    env.player_controller = CustomController(weights)  # 用自定义的控制器控制玩家
    fitness, _, _, _ = env.play()  # 运行游戏并获取适应度值
    return fitness

# 初始化种群
def initialize_population():
    return [generate_random_weights() for _ in range(population_size)]

# 选择操作（轮盘赌选择）
def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    selected = np.random.choice(population, p=selection_probs)
    return selected

# 交叉操作
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)  # 随机选交叉点
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2

# 变异操作
def mutate(weights):
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += np.random.uniform(-0.1, 0.1)  # 随机修改
    return weights

# 遗传算法的主要逻辑
def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        fitness_scores = [fitness_function(individual) for individual in population]  # 评估每个个体
        new_population = []
        
        # 选择、交叉和变异生成新个体
        while len(new_population) < population_size:
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        
        # 将新一代个体替换老一代个体
        population = new_population[:population_size]
        print(f"Generation {generation}, Best Fitness: {max(fitness_scores)}")

    # 返回适应度最高的个体
    best_individual = max(population, key=fitness_function)
    return best_individual

# 自定义控制器，使用神经网络控制玩家
class CustomController:
    def __init__(self, weights):
        self.weights = weights

    def set(self, weights, num_sensors):
        """设置控制器的权重."""
        self.weights = weights

    def control(self, inputs):
        # 假设 weights 是一个简单的线性变换
        return np.dot(inputs, self.weights)
    
# 运行遗传算法并找到最优个体
best_solution = genetic_algorithm()
print(f"Best Solution: {best_solution}")
