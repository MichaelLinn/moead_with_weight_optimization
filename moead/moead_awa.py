# coding=utf-8
import random
import math
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import re


class moead_awa(object):
    def __init__(self, population, toolbox, mu, cxpb, mutpb, ngen=0, maxEvaluations=0,
                 T=20, nr=2, delta=0.9, stats=None, halloffame=None, verbose=__debug__, dataDirectory="weights"):

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

        fitnesses = self.toolbox.map(self.toolbox.evaluate, self.population)

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
        while n < 300:
            for i in xrange(len(self.population)):
                print "Gen = ", n, " Pop = ", i
                """
                print "neighorhood:"
                for a in self.neighbourhood_[i]:
                    print self.indArray_[a]
                """
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
                # Step 2.5 adjust weight
                self.updateEP(offSpring)

            if n > 100 and n % 10 == 0:
                self.adjustWeight()
                self.updateAllNeighborhood()

            n = n + 1

        print self.indArray_
        # w_df = pd.DataFrame(self.lambda_)
        # p_df = pd.DataFrame(self.indArray_[0])

        # w_df.to_csv("weight_zdt3.csv", header=False,index=False)
        # p_df.to_csv("population.csv",header=False,index=False)
        return self.paretoPoints, self.lambda_

    def transferWeitghts(self):

        tem_ws = deepcopy(self.lambda_)
        lambda_ = []
        for i in range(len(tem_ws)):
            tem_w = np.array(tem_ws[i])
            tem_w += 0.0000001
            tem_w = 1 / tem_w
            sum_w = sum(tem_w)
            tem_w = tem_w / (sum_w)
            lambda_.append(tem_w.tolist())
        self.lambda_ = lambda_

    def initUniformWeight(self):
        # ||w_i||^2 = 1
        if self.n_objectives == 2:
            p = 90.0 / (self.populationSize_ - 1)
            for i in xrange(self.populationSize_):
                theta_ = 1.0 * float(i) * p
                weight = [math.cos(math.radians(theta_)), math.sin(math.radians(theta_))]
                self.lambda_.append(weight)
        else:
            with open('3_objective_weights.txt', 'rt') as f:
                for line in f:
                    a = re.split("    |\n| ", line)
                    w = [float(a[1]), float(a[2]), float(a[3])]
                    self.lambda_.append(w)
        self.transferWeitghts()

        """
        for n in xrange(self.populationSize_):
            a = float(1.0 * float(n) / (self.populationSize_ - 1))
            tem = [a,1-a]
            self.lambda_.append(tem)
        """

    def initNeighbourhood(self):
        x = [None] * self.populationSize_  # Of type float
        idx = [None] * self.populationSize_  # Of type int

        for i in xrange(self.populationSize_):
            for j in xrange(self.populationSize_):
                x[j] = (self.distVector(self.lambda_[i], self.lambda_[j]))
                idx[j] = j
            self.minFastSort(x, idx, self.populationSize_, self.T_)
            self.neighbourhood_.append(idx[0:self.T_])
            # print i, "th's neigborhoods", self.neighbourhood_[i][0:self.T_]
            # System.arraycopy(idx, 0, neighbourhood_[i], 0, T_)

    def calSparsityValue(self):
        self.sparsityVals = []
        w_value = 1.0
        for ith_sub in range(len(self.indArray_)):
            for j in range(self.T_):
                w_value *= self.distVector(self.lambda_[ith_sub], self.lambda_[self.neighbourhood_[ith_sub][j]])
            self.sparsityVals.append(w_value)

    def adjustWeight(self):
        self.calSparsityValue()
        sorted_idx = (np.argsort(np.array(self.sparsityVals))).tolist()
        num = len(sorted_idx) / 10
        add_ind = self.indArray_[sorted_idx[(len(sorted_idx) - 1)]]
        fitness = np.array(self.toolbox.evaluate(add_ind))
        z_point = np.array(self.z_)
        w = 1.0 / (fitness - z_point + 0.0000001)
        w_sum = sum(w)
        new_w = (w / w_sum).tolist()
        # update overcrowded individuals
        for i in range(num):
            self.indArray_[sorted_idx[i]] = add_ind
            self.lambda_[sorted_idx[i]] = new_w


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
            self.indArray_[i].fitness.values = self.toolbox.evaluate(self.indArray_[i])
            if self.indArray_[i].fitness.values[0] < self.z_[0]:  # For Minimization problem"
                self.z_[0] = self.indArray_[i].fitness.values[0]
            if self.indArray_[i].fitness.values[1] < self.z_[1]:
                self.z_[1] = self.indArray_[i].fitness.values[1]

    def reproduction(self, i_subproblem, ):

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
                # update weight
                """
                if gen % 30 == 0 and gen > 100:
                    self.updateNeighboringWeight(idx, child_vector)
                    self.updateAllNeighborhood()
                """
                """
                if gen > 50:
                    self.updateNeighboringWeight(idx, child_vector)
                    self.updateAllNeighborhood()
                """
    def updateNeighboringWeight(self, idx, new_solution):

        # print "updating Neighboring Weight!"
        w_vector = deepcopy(self.lambda_)
        neighbor_weight = w_vector[idx]
        # old_tr = []
        # old_tr.append(self.tchebysheff_func(neighbor_weight, new_solution, idx))

        n = 0
        while n < 10:
            # weight gradient
            w_gradient = []
            for i in range(self.n_objectives):
                # computing Partial Derivative of Weight
                w_gradient.append(self.computePartialDerivative(i, neighbor_weight, new_solution, idx))
                tem_g = np.array(w_gradient)
                neighbor_weight = (np.array(neighbor_weight) - 0.0002 * tem_g).tolist()
            n = n + 1

        # Regularize the                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            w_i in order to satisfying the constrain  w's l2 norm == 1 and w_i >= 0
        flag = 0
        for i in range(self.n_objectives):
            if neighbor_weight[i] < 0:
                neighbor_weight[i] = 0.00000001
                flag = 1

        sum_w = 1.0 * sum(np.array(neighbor_weight) ** 2)
        for i in range(self.n_objectives):
            if flag == 0:
                neighbor_weight[i] = (neighbor_weight[i] ** 2 / sum_w) ** 0.5

        self.lambda_[idx] = neighbor_weight
        # print "new weight:", neighbor_weight


    def tchebysheff_func(self, lambda_, individual, i_subproblem):
        # g_te_i - theta * dist(w_i,W)
        lambda_ = list(lambda_)
        l = 1.0  # theta
        g_te = - 1.0
        # max { lambda_j * |f_i(x) - z_i| }
        for i in range(self.n_objectives):
            tem = 1.0 * lambda_[i] * abs((individual.fitness.values[i] - self.z_[i]))
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
        for i in range(self.n_objectives - 1):
            result_leq = result_leq & flag_leq[:, i + 1]

        flag_l = new_fitness < paretoFront

        # using De Morgan's theorem to transfer a or b to !(!a and !b)
        result_l = ~flag_l[:, 0]
        for i in range(self.n_objectives - 1):
            result_l = result_l & (~flag_l[:, i + 1])
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
            result_new = result_new & flag_new[:, i + 1]

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

    def searchArmijoWeight(self, gradient, weight, new_solution, i_subproblem):
        # lambda_i
        # find a new weight_i based on Gradient Descent Method using Armijo
        gradient = np.array(gradient)
        # print "g: ", float(- np.mat(gradient) * np.mat(gradient).T)
        weight = np.array(weight)
        x = 0.0
        beta_ = 0.0
        rho_ = 0.0
        m = 1.0
        while beta_ == 0:
            beta_ = 0.1
        while rho_ == 0:
            rho_ = random.random() / 2

        while True:
            func_1 = self.tchebysheff_func((weight + (beta_ ** m) * (- gradient)), new_solution, i_subproblem)
            func_2 = self.tchebysheff_func(weight, new_solution, i_subproblem) + rho_ * (beta_ ** m) * float(
                - np.mat(gradient) * np.mat(gradient).T)

            if func_1 > func_2:
                m = m + 1
            else:
                new_weight = weight + (beta_ ** m * - gradient)
                print "step", beta_ ** m
                return new_weight.tolist()

    def computePartialDerivative(self, i_weight, weight_vector, new_solution, i_subproblem):
        # print "computing partial derivative!"
        delta_w = 10e-8
        weight_v = deepcopy(weight_vector)
        weight_left = deepcopy(weight_vector)
        weight_right = deepcopy(weight_vector)

        while True:
            weight_right[i_weight] = weight_v[i_weight] + delta_w
            weight_left[i_weight] = weight_v[i_weight] - delta_w
            func_1_right = self.tchebysheff_func(weight_right, new_solution, i_subproblem)
            func_1_left = self.tchebysheff_func(weight_left, new_solution, i_subproblem)
            func_2 = self.tchebysheff_func(weight_v, new_solution, i_subproblem)
            gradient_right = (func_1_right - func_2) / delta_w
            gradient_left = (func_2 - func_1_left) / delta_w
            if abs(gradient_left - gradient_right) < 10e-3:
                # Judging if the point is discontinuity or not
                return gradient_left
            else:
                self.dp.append(weight_vector)
                weight_v[i_weight] = weight_v[i_weight] + delta_w














