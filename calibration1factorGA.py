#genetic algorithm
from deap import base, creator, tools
import random
import numpy as np
import time
import loglikelihood as llh

# weights 1.0, 求最大值,-1.0 求最小值, (1.0,-1.0,)求第一个参数的最大值,求第二个参数的最小值
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialization
IND_SIZE = 5  # 种群数

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)   # 调用randon.random(产生[0，1）上均匀分布的随机数)，为每一个基因编码,创建随机初始值
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE) #individule是一个list，包含 n 个数字
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operators
# define evaluate function
def evaluate(x):
    if x[0]<=0:
        x[0]=np.mod(abs(x[0]),1) #make sure that the variable is in the domain
    if x[1]<=0:
        x[1]=np.mod(abs(x[1]),1)
    if x[2]<=0:
        x[2]=np.mod(abs(x[2]),1)
    if x[3] < 0 or x[3] > 1:
        x[3]=np.mod(abs(x[2]),1)*(3/2-(-3))-3
    if x[4]<0 or x[4]>1:
        x[4]=np.mod(abs(x[2]),1)
    a=x[0]+1.1
    b=x[1]+1.1
    sigma=x[2]+1
    LLH = -llh.log_likelihood(a, b, sigma, alpha=[x[3]], theta=[x[4]])  # polynomial degree N=3,
    return LLH


# use tools in deap to creat our application
toolbox.register("mate", tools.cxTwoPoint) # 两点交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # mutate : 变异
toolbox.register("select", tools.selTournament, tournsize=3) # select : 选择保留的最佳个体
toolbox.register("evaluate", evaluate)  # commit our evaluate


# Algorithms
def main():
    # create an initial population of 50 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=50)  #产生size为n的population
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    # NGEN  is the number of generations for which the evolution runs

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))  # 这时候，pop的长度还是50呢
    print("-- Iterative %i times --" % NGEN)

    for g in range(NGEN):
        if g % 10 == 0:
            print("-- Generation %i --" % g)
        offspring = toolbox.select(pop, len(pop))   # Select the next generation individuals

        offspring = list(map(toolbox.clone, offspring))   # Clone the selected individuals（因为下面的操作会改变offspring），Change map to list

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]): #分别为offspring中的偶数项和奇数项
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        #print(pop)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    return best_ind, best_ind.fitness.values  # return the result:Last individual,The Return of Evaluate function


if __name__ == "__main__":
    t1 = time.clock()
    best_ind, best_ind.fitness.values = main()
    # print(pop, best_ind, best_ind.fitness.values)
    # print("pop",pop)
    print("best_ind:\n",best_ind)
    print("best_ind.fitness.values:\n",best_ind.fitness.values)

    t2 = time.clock()

    print('Time cost:\n',t2-t1)