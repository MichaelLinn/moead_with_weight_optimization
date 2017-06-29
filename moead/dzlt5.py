import numpy as np
import math
from copy import deepcopy
from deap import base, creator
import random
from deap import tools
from moead_dzlt5 import moead_dzlt5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = 3
M = 5
IND_SIZE = 14
n_subproblem = 100

def evalDZLT5(individual):
    solution = individual
    fitness = ()
    for i in range(M):
        fitness += (genFunc(i, solution),)
    return fitness

def genFunc(idx, solution):
    result = 1 + func_g(solution)
    if idx == 0:
        for i in range(M - 1):
            result *= math.cos(genTheta(i, solution))
    else:
        if idx == M - 1:
            result *= math.sin(genTheta(0, solution))
        else:
            for i in range(M - idx):
                result *= math.cos(genTheta(i, solution)) \
                          * math.sin(genTheta((M - idx - 1 + 1), solution))
    return result

def func_g(solution):
    g = 0
    for i in range(M - 1, IND_SIZE):
        g += (solution[i] - 0.5) ** 2
    return g

def genTheta(idx, solution):
    if idx <= m - 2:
        theta_ = math.pi / 2 * solution[idx]
    else:
        g = func_g(solution)
        theta_ = math.pi / (4 * (1 + g)) * (1 + 2 * g * solution[idx])

    return theta_

def crxover(ind1, ind2, ind3, i_solution):
    D = len(ind1)
    CR = 1
    F = 0.5
    jrandom = int(random.random() * D)
    offSpring = deepcopy(ind1)
    for index in range(D):
        offSpring[index] = 0
        if random.random() < CR or index == jrandom:
            offSpring[index] = ind1[index] + F * (ind2[index] - ind3[index])
        else:
            offSpring[index] = i_solution[index]
        # repair
        high = 1
        low = 0
        if offSpring[index] > high:
            offSpring[index] = high
        else:
            if offSpring[index] < low:
                offSpring[index] = low
    return offSpring

def mutate(ind):
    dim = len(ind)
    eta_m = 20
    rate = 1.0 / dim
    for i in range(dim):
        if random.random() < rate:
            y = ind[i]
            yl = 0
            yu = 1

            delta1 = (y - yl)/(yu - yl)
            delta2 = (yu - y)/(yu - yl)

            rnd = random.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (math.pow(xy, (eta_m + 1.0)))
                deltaq = math.pow(val, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (math.pow(xy, (eta_m + 1.0)))
                deltaq = 1.0 - (math.pow(val, mut_pow))

            y = y + deltaq * (yu - yl)
            if y < yl:
                y = yl
            if y > yu:
                y = yu
            ind[i] = y
    return ind

def plotResult(paretoF):

    fig1 = plt.figure()

    ax = fig1.add_subplot(1,1,1, projection = '3d')

    ax.scatter(paretoF[:,4].tolist(), paretoF[:,3].tolist(), paretoF[:,2].tolist(),
               c = 'red', marker='o',label = "Pareto Front")
    ax.legend()
    plt.show()



def main(objectives=5):


    creator.create("Fitness", base.Fitness, weights = (-1,-1,-1,-1,-1))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    CXPB, MUTPB, NGEN = 0.7, 0.2, 50

    toolbox.register("mate", crxover)
    toolbox.register("mutate", mutate)

    toolbox.register("evaluate", evalDZLT5)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n = n_subproblem)

    hof = tools.ParetoFront()


    flag = 2
    d_metric = []

    ea_w2 = moead_dzlt5(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
    paretoPoints, weight = ea_w2.execute()

    pf = np.array(toolbox.map(toolbox.evaluate, paretoPoints))

    plotResult(pf)


    weight = np.array(weight)

    return pop, hof

if __name__ == "__main__":
    print "Program is running!"
    objectives = 2

    pop, pf = main(objectives)
