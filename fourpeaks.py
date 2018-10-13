import sys
sys.path.append("/Users/Quintus/OneDrive/Code/python/omscs/machine learning/assignment 2/ABAGAIL/ABAGAIL.jar")
import os
import time
import csv

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array



# N = 30                      # the number of bits
# T = N/10                      # threshold
# fill = [2] * N
# ranges = array('i', fill)
#
# ef = FourPeaksEvaluationFunction(T)
# odd = DiscreteUniformDistribution(ranges)
# nf = DiscreteChangeOneNeighbor(ranges)
# mf = DiscreteChangeOneMutation(ranges)
# cf = SingleCrossOver()
# df = DiscreteDependencyTree(.1, ranges)
# hcp = GenericHillClimbingProblem(ef, odd, nf)
# gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
# pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

# rhc = RandomizedHillClimbing(hcp)
# fit = FixedIterationTrainer(r hc, 200000)
# fit.train()
# print "RHC: " + str(ef.value(rhc.getOptimal()))
#
# sa = SimulatedAnnealing(1E11, .95, hcp)
# fit = FixedIterationTrainer(sa, 200000)
# fit.train()
# print "SA: " + str(ef.value(sa.getOptimal()))
#
# ga = StandardGeneticAlgorithm(200, 100, 10, gap)
# fit = FixedIterationTrainer(ga, 1000)
# fit.train()
# print "GA: " + str(ef.value(ga.getOptimal()))

# mimic = MIMIC(500, 50, pop)     # the number of samples, tokeeps
# fit = FixedIterationTrainer(mimic, 5000)
# fit.train()
# print "MIMIC: " + str(ef.value(mimic.getOptimal()))


def OptimedTrainer(ef, oa, optima):
    start = time.time()
    iters = 0
    cur_value = -1
    while cur_value < optima:
        oa.train()
        cur_value = ef.value(oa.getOptimal())
        iters += 1
        if iters > 1000000:  # 1000k
            print "bound of iterations reached!"
            break
        # print oa, iters, cur_value
    return iters, time.time()-start


# mimic = MIMIC(500, 20, pop)   # the number of samples, tokeeps(theta)
# n = OptimedTrainer(ef, mimic, optima=56)
# print n
# # higher the theta, faster the fitness goes up with the number of iterations
# # more samples in single iterations, faster converge

# ga = StandardGeneticAlgorithm(500, 250, 50, gap)
# n = OptimedTrainer(ef, ga, optima=56)
# print n

# sa = SimulatedAnnealing(1E11, .95, hcp)
# n = OptimedTrainer(ef, sa, optima=56)
# print n

# rhc = RandomizedHillClimbing(hcp)
# n = OptimedTrainer(ef, rhc, optima=56)
# print n

# ------------------------------------------------
start = time.time()

mul_res_it = []
mul_res_t = []
for i in range(10):
    print i
    res_it = []
    res_t = []
    for N in [30, 40, 50, 60, 70, 80]:
        print N
        T = N / 10
        opt_val = N + N-T-1
        # print N, T, opt_val
        fill = [2] * N
        ranges = array('i', fill)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        df = DiscreteDependencyTree(.1, ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

        rhc = RandomizedHillClimbing(hcp)
        sa = SimulatedAnnealing(1E11, .95, hcp)
        ga = StandardGeneticAlgorithm(500, 250, 20, gap)
        mimic = MIMIC(500, 20, pop)

        row_it = []
        row_t = []
        for oa in [rhc, sa, ga, mimic]:
            print oa
            n_iters, t = OptimedTrainer(ef, oa, optima=opt_val)
            row_it.append(n_iters)
            row_t.append(t)
        res_it.append(row_it)
        res_t.append(row_t)
    print 'n_iters: ', res_it
    print 'time: ', res_t
    mul_res_it.append(res_it)
    mul_res_t.append(res_t)
    print time.time() - start, ' seconds'

with open('fp_n_iters.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(mul_res_it)
with open('fp_time.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(mul_res_t)

print time.time()-start, ' seconds'  # 132 secs each

# [[10000001, 1236, 375, 77], [10000001, 2422, 472, 188], [10000001, 2405, 676, 357], [10000001, 4849, 1834, 522], [10000001, 10000001, 2740, 398], [10000001, 10000001, 5755, 739]]
#   rhc,      sa,   ga,  mimic
# [[10000001, 1236, 375, 77],
#  [10000001, 2422, 472, 188],
#  [10000001, 2405, 676, 357],
#  [10000001, 4849, 1834, 522],
#  [10000001, 10000001, 2740, 398],
#  [10000001, 10000001, 5755, 739]]


# ------------------------------------------------
# start = time.time()
#
# res_it = []
# res_t = []
# for N in [30, 40, 50, 60, 70, 80]:
#     print N
#     T = N / 10
#     opt_val = N + N - T - 1
#     # print N, T, opt_val
#     fill = [2] * N
#     ranges = array('i', fill)
#
#     ef = FourPeaksEvaluationFunction(T)
#     odd = DiscreteUniformDistribution(ranges)
#     nf = DiscreteChangeOneNeighbor(ranges)
#     mf = DiscreteChangeOneMutation(ranges)
#     cf = SingleCrossOver()
#     df = DiscreteDependencyTree(.1, ranges)
#     hcp = GenericHillClimbingProblem(ef, odd, nf)
#     gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
#     pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
#     rhc = RandomizedHillClimbing(hcp)
#     sa = SimulatedAnnealing(1E11, .95, hcp)
#     ga = StandardGeneticAlgorithm(500, 250, 20, gap)
#     mimic = MIMIC(500, 20, pop)
#
#     row_it = []
#     row_t = []
#     for oa in [rhc, sa, ga, mimic]:
#         print oa
#         n_iters, t = OptimedTrainer(ef, oa, optima=opt_val)
#         row_it.append(n_iters)
#         row_t.append(t)
#     res_it.append(row_it)
#     res_t.append(row_t)
# print 'n_iters: ', res_it
# print 'time: ', res_t
#
# print time.time()-start, ' seconds'  # 132 secs each

# n_iters:  [[1000001, 872, 134, 139], [1000001, 1956, 532, 234], [1000001, 2772, 917, 397], [1000001, 3525, 2143, 828], [1000001, 2989, 4945, 681], [1000001, 1000001, 3359, 746]]
# time:  [[2.2799999713897705, 0.03299999237060547, 0.20399999618530273, 1.3220000267028809], [1.6570000648498535, 0.004999876022338867, 0.2560000419616699, 1.7889997959136963], [1.253999948501587, 0.005000114440917969, 0.22600007057189941, 2.7339999675750732], [1.247999906539917, 0.009000062942504883, 0.4570000171661377, 6.275000095367432], [0.9750001430511475, 0.003999948501586914, 1.0290000438690186, 5.2209999561309814], [0.9860000610351562, 1.5899999141693115, 0.7580001354217529, 9.592000007629395]]

# n_iters:  [[1000001, 1256, 109, 51], [1000001, 1559, 399, 69], [1000001, 2984, 206, 348], [1000001, 3263, 3773, 276], [1000001, 4505, 6059, 482], [1000001, 6218, 14874, 541]]
# time:  [[2.361999988555908, 0.015999794006347656, 0.09800004959106445, 0.3769998550415039], [1.4570000171661377, 0.0019998550415039062, 0.16000008583068848, 0.4549999237060547], [1.1599998474121094, 0.006999969482421875, 0.10800004005432129, 2.544999837875366], [1.124000072479248, 0.01900005340576172, 0.9660000801086426, 2.1059999465942383], [1.6929998397827148, 0.005000114440917969, 1.371999979019165, 3.7269999980926514], [0.9650001525878906, 0.006000041961669922, 3.364000082015991, 4.896999835968018]]

# n_iters:  [[834, 1001, 69, 57], [1000001, 2094, 169, 79], [1000001, 2634, 779, 379], [1000001, 1000001, 6429, 525], [1000001, 4380, 3429, 559], [1000001, 5220, 13222, 440]]
# time:  [[0.04399991035461426, 0.017000198364257812, 0.07999992370605469, 0.5080001354217529], [1.9699997901916504, 0.006999969482421875, 0.05800008773803711, 0.7960000038146973], [1.3550000190734863, 0.003000020980834961, 0.28100013732910156, 2.429999828338623], [1.622999906539917, 2.2779998779296875, 2.2249999046325684, 5.490999937057495], [2.305000066757202, 0.009000062942504883, 1.808000087738037, 6.795999765396118], [1.065999984741211, 0.006000041961669922, 3.0460000038146973, 4.105000019073486]]
