# coding=utf-8
import random
import math
import numpy as np
import pandas as pd
from copy import deepcopy
import re

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


class moead_dzlt5(object):

    def __init__(self, population, toolbox, mu, cxpb, mutpb, ngen=0, maxEvaluations=0,
                 T=30, nr=2, delta=0.9, stats=None, halloffame=None, verbose=__debug__, dataDirectory="weights"):

        self.populationSize_ = int(0)

        self.co = 0

        # Reference the DEAP toolbox to the algorithm
        self.toolbox = toolbox

        # Pareto Otimal point
        self.paretoPoints = []

        # Store the population
        self.population = population

        # Individual array
        self.indArray_ = []

        # Discontinuity point in weight optimization
        self.dp = []

        fitnesses = self.toolbox.map(self.toolbox.evaluate,self.population)

        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
            self.indArray_.append(ind)

        # print "all individual", self.indArray_
        # print "0 fitness", self.indArray_[0].fitness.values

        self.populationSize_ = mu

        # Z vector (ideal vector)
        self.z = []

        # Lambda vectors (weight vector)
        self.lambda_ = []

        # Neighborhood size
        self.T_ = T

        # Neighbourhood
        self.neighbourhood_ = []

        self.n_objectives = len(self.population[0].fitness.values)

        # Stopping criterion
        self.ngen = ngen

    def execute(self):
        # print "Executing MOEA/D with weight optimization"

        # print "Population Size:", self.populationSize_

        # 2-D list of size populationSize * T_
        self.neighbourhood_ = []

        # List of size number of objectives. Contains best fitness.
        self.z_ = []

        # 2-D list  Of size populationSize_ x Number of objectives. Used for weight vectors.
        self.lambda_ = []

        # STEP 1. Initialization
        self.initUniformWeight()
        # print "lambda",self.lambda_
        self.initNeighbourhood()
        # print "all neighborhood", self.neighbourhood_
        self.initIdealPoint()
        # print "Ideal points", self.z_

        n = 0

        while n < 150:
            for i in xrange(len(self.population)):
                print "Gen = ",n ," Pop = ", i
                # STEP 2.1 Reproduction
                offSpring = self.reproduction(i)

                offSpring_fitness = self.toolbox.evaluate(offSpring)
                offSpring.fitness.values = offSpring_fitness

                # Step 2.2 Improvement:
                # Apply a problem-specific repair, which is not necessary in this problem
                # Step 2.3 Update of z_
                self.updateZ_point(offSpring)
                # Step 2.4 Update of Neighboring Solution
                self.updateNeigboringSolution(offSpring, i, n)
                # Step 2.5 Update of EP
                self.updateEP(offSpring)

            # Step 3.1 Optimize the weight vectors
            """
            if  n > 150 :
                for i in range(len(self.indArray_)):
                    print "Peroto Point:", self.toolbox.evaluate(self.indArray_[i])
                    print "old weight:",self.lambda_[i]
                    print "old neiborhood:", self.lambda_[self.neighbourhood_[i][0]]
                    self.updateNeiWeight(i)
                    self.updateAllNeighborhood()
                    print "new weight:", self.lambda_[i]
                    print "new neiborhood:", self.lambda_[self.neighbourhood_[i][0]]
            # Step 3.2 update neighboring
            """

            if n > 50 and n % 30 == 0:
                for i in range(len(self.indArray_)):
                    self.updateNeiWeight(i)
                    self.updateAllNeighborhood()

            n = n + 1

        # print self.indArray_
        w_df = pd.DataFrame(self.lambda_)
        # p_df = pd.DataFrame(self.indArray_[0])

        w_df.to_csv("weight_dzlt5.csv", header=False,index=False)
        # p_df.to_csv("population.csv",header=False,index=False)
        return self.paretoPoints, self.lambda_


    def initUniformWeight(self):
        # ||w_i||^2 = 1
        if self.n_objectives == 2:
            p = 90.0/(self.populationSize_-1)
            for i in xrange(self.populationSize_):
                theta_ = 1.0 * float(i) * p
                weight = [math.cos(math.radians(theta_)), math.sin(math.radians(theta_))]
                self.lambda_.append(weight)
        else:
            with open('5_objective_weights.txt', 'rt') as f:
                for line in f:
                    a = re.split("    |\n| ", line)
                    w = []
                    for i in range(1, self.n_objectives + 1):
                        w.append(float(a[i]))
                    self.lambda_.append(w)


    def initNeighbourhood(self):
        x = [None] * self.populationSize_    # Of type float
        idx = [None] * self.populationSize_  # Of type int

        for i in xrange(self.populationSize_):
            for j in xrange(self.populationSize_):
                x[j] = (self.distVector(self.lambda_[i], self.lambda_[j]))
                idx[j] = j
            self.minFastSort(x, idx, self.populationSize_, self.T_)
            self.neighbourhood_.append(idx[1:self.T_+1])
            # print i, "th's neigborhoods", self.neighbourhood_[i][0:self.T_]
            # System.arraycopy(idx, 0, neighbourhood_[i], 0, T_)

    def updateAllNeighborhood(self):
        self.neighbourhood_ = []
        self.initNeighbourhood()

    def distVector(self, vector1, vector2):
        dim = len(vector1)
        sum_ = 0
        for n in xrange(dim):
            sum_ += ((vector1[n] - vector2[n]) * (vector1[n] - vector2[n]))
        return math.sqrt(sum_)

    def minFastSort(self, x, idx, n, m):
        """
        x   : list of floats
        idx : list of integers (each an index)
        n   : integer
        m   : integer        
        """
        for i in xrange(m):
            for j in xrange(i + 1, n):
                if x[i] > x[j]:
                    temp = x[i]
                    x[i] = x[j]
                    x[j] = temp
                    id_ = idx[i]
                    idx[i] = idx[j]
                    idx[j] = id_

    """
    " initIdealPoint
    """

    def initIdealPoint(self):
        for i in range(self.n_objectives):
            self.z_.append(1e30)
        for i in range(len(self.population)):
            for j in range(self.n_objectives):
                if self.indArray_[i].fitness.values[j] < self.z_[j]:  # For Minimization problem"
                    self.z_[j] = self.indArray_[i].fitness.values[j]



    def reproduction(self, i_subproblem,):

        k_idx = random.randrange(self.T_)
        while True:
            l_idx = random.randrange(self.T_)
            if l_idx != k_idx:
                while True:
                    m_idx = random.randrange(self.T_)
                    if m_idx != l_idx and m_idx != k_idx:
                        break
                break

        # print "neighboring: ", self.neighbourhood_[i_subproblem]

        k_neib = self.neighbourhood_[i_subproblem][k_idx]
        k_ind = self.indArray_[k_neib]
        l_neib = self.neighbourhood_[i_subproblem][l_idx]
        l_ind = self.indArray_[l_neib]
        m_neib = self.neighbourhood_[i_subproblem][m_idx]
        m_ind = self.indArray_[m_neib]

        # Something very very very strange happens, so I use deepcopy to protect the "self.paretoPoints"
        tem_pf = deepcopy(self.paretoPoints)
        # 　Apply Crossover
        offSpring = self.toolbox.mate(k_ind, l_ind, m_ind, self.indArray_[i_subproblem])

        self.paretoPoints = tem_pf

        # Apply mutation
        offSpring = self.toolbox.mutate(offSpring)
        # print "offspring: ", offSpring
        return offSpring

    def updateZ_point(self, child_ind):
        child_fitness = self.toolbox.evaluate(child_ind)
        for i in range(self.n_objectives):
            if child_fitness[i] < self.z_[i]:
                self.z_[i] = child_fitness[i]




    def updateNeigboringSolution(self, child_vector, i_subproblem, gen):

        for idx in self.neighbourhood_[i_subproblem]:
            i_lambda = self.lambda_[idx]
            ind = self.indArray_[idx]
            gte_child = self.tchebysheff_func(i_lambda, child_vector, idx)
            gte_ind = self.tchebysheff_func(i_lambda, ind, idx)
            if gte_child < gte_ind:
                self.indArray_[idx] = child_vector
                self.indArray_[idx].fitness.value = self.toolbox.evaluate(child_vector)


    def updateNeiWeight(self, idx):
        print "updating weight!"
        delta_ = 0.001
        w_vector = deepcopy(self.lambda_)
        ind_ = self.indArray_[idx]
        fitness = self.toolbox.evaluate(ind_)
        nearest_nei = w_vector[self.neighbourhood_[idx][0]]
        nei_ind = self.indArray_[self.neighbourhood_[idx][0]]
        nei_fitness = self.toolbox.evaluate(nei_ind)
        # nearest_nei = self.toolbox.evaluate(self.indArray_[self.neighbourhood_[idx][0]])
        weight = w_vector[idx]
        new_w = []
        for i in range(self.n_objectives):
            print "F",i," :",fitness[i]
            new_w.append(fitness[i] + delta_ *(fitness[i] - nearest_nei[i]))
            print "N", i, " :", nearest_nei[i]


        self.lambda_[idx] = new_w

    def tchebysheff_func(self, lambda_, individual, i_subproblem):
        # g_te_i - theta * dist(w_i,W)
        lambda_ = list(lambda_)
        l = 1.0 # theta
        g_te = - 1.0
        # max { lambda_j * |f_i(x) - z_i| }
        for i in range(self.n_objectives):
            if lambda_[i] <= 0:
                lambda_[i] = 0.001
            tem = 1.0 * abs((individual.fitness.values[i] - self.z_[i])) / lambda_[i]
            if tem > g_te:
                g_te = tem
        dist = self.distVector(self.lambda_[i_subproblem], self.lambda_[self.neighbourhood_[i_subproblem][0]])
        result = g_te - l * dist
        return result

    def distWeight(self, i_subproblem):
        w_dist = []
        for i in range(len(self.lambda_)):
            if i == i_subproblem:
                continue
            w_dist.append(self.distVector(self.lambda_[i_subproblem], self.lambda_[i]))
        w_dist.sort()
        return w_dist[0]


    def updateEP(self, new_vector):

        tem_pp = self.paretoPoints
        new_vector.fitness.values = self.toolbox.evaluate(new_vector)
        # print "new point：", new_vector
        # print "new solution fitness: ", new_vector.fitness.values
        if new_vector.fitness.values[1] <= 1:
            self.co += 1

        if len(tem_pp) == 0:
            tem_pp.append(new_vector)
            self.paretoPoints = tem_pp
            return

        new_fitness = np.array(new_vector.fitness.values)
        paretoFront = np.array(self.toolbox.map(self.toolbox.evaluate, tem_pp))

        # Estimate if any current_PF is dominated new_solution
        flag_leq = new_fitness <= paretoFront
        result_leq = flag_leq[:, 0]
        for i in range(self.n_objectives-1):
            result_leq = result_leq & flag_leq[:,i+1]

        flag_l = new_fitness < paretoFront

        # using De Morgan's theorem to transfer a or b to !(!a and !b)
        result_l = ~flag_l[:, 0]
        for i in range(self.n_objectives-1):
            result_l = result_l & (~flag_l[:,i+1])
        result_l = ~result_l

        # the current_PF non-dominated by new_child_vector
        result = ~(result_l & result_leq)

        # delete the dominated PF
        tem_pf = np.array(self.paretoPoints)
        pf_list = tem_pf[result].tolist()
        tem_pf = self.toolbox.map(self.toolbox.evaluate, pf_list)
        # print "pf_list: ", tem_pf
        tem_pp = []

        # fit = self.toolbox.map(self.toolbox.evaluate, pf_list)
        for item in pf_list:
            tem_pp.append(item)

        # Estimate if new_solution is dominated any current_PF
        flag_new = new_fitness >= paretoFront
        # print "flag_new: ", flag_new
        result_new = flag_new[:, 0]
        for i in range(self.n_objectives - 1):
            result_new = result_new & flag_new[:, i+1]

        # Delete the dominated new_solution
        if not result_new.any():
            # print "add new solution!"
            tem_pp.append(new_vector)

        self.paretoPoints = tem_pp

        # print "Pareto optimal point: ", self.paretoPoint
        PF = self.toolbox.map(self.toolbox.evaluate, self.paretoPoints)
        # print "Pareto Front: ", PF
        # print "lambda: ", self.lambda_
        # print "ideal point: ", self.z_











