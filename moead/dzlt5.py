import numpy as np
import pandas as pd
import math


m = 2
M = 5
IND_SIZE = 10



def evalDZLT5(individual):
    ilist = individual
    func1 = ilist[0]
    gr = g(ilist)
    func2 = gr * (1 - (func1 / gr) ** 0.5 - func1/gr*math.sin(math.pi*10*func1))
    return func1, func2 + 0.8

def genFunc(idx, solution):
    result = 1 + func_g(solution)
    if idx == 0:
        for i in range(M - 1):
            result *= math.cos(genTheta(i, solution))
    else:
        if idx == M - 1:
            result *= math.sin(genTheta(idx, solution))
        else:
            for i in range(M - idx):
                result *= math.cos(genTheta(i, solution)) \
                          * math.sin(genTheta((M - idx - 1 + 1), solution))
    return result

def func_g(solution):
    g = 0
    for i in range(M, IND_SIZE + 1):
        g += (solution[i] - 0.5) ** 2
    return g

def genTheta(idx, solution):
    if idx <= m - 2:
        theta_ = math.pi / 2 * solution[idx]
    else:
        g = func_g(solution)
        theta_ = math.pi / (4 * (1 + g)) * (1 + 2 * g * solution[idx])

    return theta_

