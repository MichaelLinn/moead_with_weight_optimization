from deap import base, creator
import random
from deap import tools
import numpy as np
from moead_ import moead_
from moead_weight import moead_w
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import pandas as pd

IND_SIZE = 10

IND_INIT_SIZE = 5
MAX_ITEM = 50
MAX_WEIGHT = 50
NBR_ITEMS = 20

NGEN = 50
n_subproblem = 100
LAMBDA = 2
CXPB = 0.7
MUTPB = 0.2

# the dimension of the Input_vactor

def evalDTLZ1(individual):
    func1 = 0.5 * individual[0] * (1 + g(individual))
    func2 = 0.5 * (1 - individual[0]) * (1 + g(individual))

    return func1, func2


def g(vector):

    x_v = np.array(vector)
    v_norm = (sum(x_v ** 2))**0.5
    sum_ = 0
    for i in range(len(vector)):
        sum_ += (vector[0] - 0.5)**2 - math.cos(20*math.pi*(vector[i]-0.5))
    result = 100 * (v_norm + sum_)
    return result

def plotResult(f1, f2):

    plt.scatter(f1, f2, marker="x")
    X = np.linspace(0.0, 0.5, 500, endpoint=True)
    Y = 0.5 - X
    plt.plot(X, Y,"r")
    plt.show()


def countD_metrics(f1, f2):
    x = np.linspace(0.0, 0.5, 500, endpoint=True)
    y = 0.5 - x
    sum_dist = 0

    for i in range(500):
        minDist = 1.0 * 10e20
        for j in range(len(f1)):
            d = math.sqrt((x[i] - f1[j])**2 + (y[i] - f2[j])**2)
            if d < minDist:
                minDist = d
        sum_dist += minDist

    result = sum_dist/500
    print "D-Matric value: ", result
    return result

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


def main(objectives=2, seed=64):
    random.seed(seed)

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    CXPB, MUTPB, NGEN = 0.7, 0.2, 50

    toolbox.register("mate", crxover)
    toolbox.register("mutate", mutate)

    toolbox.register("evaluate", evalDTLZ1)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=n_subproblem)

    hof = tools.ParetoFront()

    """
    ea_w = moead_w(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
    paretoPoints = ea_w.execute()

    pf = toolbox.map(toolbox.evaluate, paretoPoints)
    f1 = []
    f2 = []
    for item in pf:
        f1.append(item[0])
        f2.append(item[1])
    countD_metrics(f1, f2)
    plotResult(f1, f2)
    """
    flag = True
    d_metric = []
    for i in range(1):
        print i,"th iteration"
        if flag:
            ea_w = moead_w(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
            paretoPoints = ea_w.execute()
        else:
            ea_ = moead_(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
            paretoPoints = ea_.execute()
        pf = toolbox.map(toolbox.evaluate, paretoPoints)
        f1 = []
        f2 = []
        for item in pf:
            f1.append(item[0])
            f2.append(item[1])
        d_metric.append(countD_metrics(f1, f2))
        plotResult(f1, f2)
    # d_mat = pd.DataFrame(d_metric)
    # d_mat.to_csv("zdt3_d_mat.csv", header=False, index=False)

if __name__ == "__main__":
    print "Program is running!"
    main()
