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
from moead_w2 import moead_w2
from moead_awa import moead_awa

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

def evalZDT3(individual):
    ilist = individual
    func1 = ilist[0]
    gr = g(ilist)
    func2 = gr * (1 - (func1 / gr) ** 0.5 - func1/gr*math.sin(math.pi*10*func1))
    return func1, func2 + 0.8


def g(x_list):
    sum_x = sum(x_list) - x_list[0]
    result = 1.0 + 9.0 * sum_x / (len(x_list) - 1)
    return result

def plotResult(f1, f2):
    X = []
    X.append(np.linspace(0.0, 0.0830015349, 100, endpoint=True))
    X.append(np.linspace(0.1822287280, 0.2577623634, 100, endpoint=True))
    X.append(np.linspace(0.4093136748, 0.4538821041, 100, endpoint=True))
    X.append(np.linspace(0.6183967944, 0.6525117038, 100, endpoint=True))
    X.append(np.linspace(0.8233317983, 0.8518328645, 100, endpoint=True))

    plt.scatter(f1, f2, marker='x', label="MOEA/D")

    for i in range(5):
        tem = [math.sin(10.0*math.pi*x) for x in X[i]]
        Y = 1 - X[i] ** 0.5 - X[i]*tem +0.8
        plt.plot(X[i], Y, color="r", label="PF")

    return plt

def countD_metrics(f1, f2):
    x_1 = (np.linspace(0.0, 0.0830015349, 100, endpoint=True))
    x_2 = (np.linspace(0.1822287280, 0.2577623634, 100, endpoint=True))
    x_3 = (np.linspace(0.4093136748, 0.4538821041, 100, endpoint=True))
    x_4 = (np.linspace(0.6183967944, 0.6525117038, 100, endpoint=True))
    x_5 = (np.linspace(0.8233317983, 0.8518328645, 100, endpoint=True))
    X = np.concatenate((x_1, x_2, x_3, x_4, x_5))

    tem = np.sin(10.0*math.pi*X)
    Y = (1 - X ** 0.5 - X * tem) + 0.8

    sum_dist = 0
    for i in range(500):
        minDist = 1.0 * 10e20
        for j in range(len(f1)):
            d = math.sqrt((X[i] - f1[j])**2 + (Y[i] - f2[j])**2)
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

    toolbox.register("evaluate", evalZDT3)
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

    flag = 2
    d_metric = []
    for i in range(1):
        print i,"th iteration"
        if flag == 1:
            ea_w = moead_w(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
            paretoPoints, weight = ea_w.execute()
        else:
            if flag == 2:
                ea_ = moead_(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
                paretoPoints, weight = ea_.execute()
            else:
                if flag == 3:
                    ea_w2 = moead_w2(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
                    paretoPoints, weight = ea_w2.execute()
                else:
                    ea_awa = moead_awa(pop, toolbox, n_subproblem, CXPB, MUTPB, ngen=NGEN, halloffame=hof)
                    paretoPoints, weight = ea_awa.execute()

        pf = toolbox.map(toolbox.evaluate, paretoPoints)
        f1 = []
        f2 = []
        for item in pf:
            f1.append(item[0])
            f2.append(item[1])
        d_metric.append(countD_metrics(f1, f2))
        plt = plotResult(f1, f2)
        weight = np.array(weight)
        plt.scatter(weight[:, 0], weight[:, 1], color = 'r')
        plt.show()
    # d_mat = pd.DataFrame(d_metric)
    # d_mat.to_csv("zdt3_d_mat.csv", header=False, index=False)

if __name__ == "__main__":
    print "Program is running!"
    main()
