import sys
sys.path.append("/Users/Quintus/OneDrive/Code/python/omscs/machine learning/assignment 2/ABAGAIL/ABAGAIL.jar")
import os
import time

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


"""
Commandline parameter(s):
   none
"""

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
# fit = FixedIterationTrainer(rhc, 200000)
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
    iters = 0
    cur_value = -1
    while cur_value < optima:
        oa.train()
        cur_value = ef.value(oa.getOptimal())
        iters += 1
        if iters > 10000000:
            print "bound of iterations reached!"
            break
        # print iters, cur_value
    return iters


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

start = time.time()

res = []
for N in [30, 40, 50, 60, 70, 80]:
    T = N / 10
    opt_val = N + N-T-1
    print N, T, opt_val
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
    ga = StandardGeneticAlgorithm(500, 250, 50, gap)
    mimic = MIMIC(500, 20, pop)

    row = []
    for oa in [rhc, sa, ga, mimic]:
        n_iters = OptimedTrainer(ef, oa, optima=opt_val)
        row.append(n_iters)
    res.append(row)
    print res
    print time.time()-start, ' seconds'

# [[10000001, 1236, 375, 77], [10000001, 2422, 472, 188], [10000001, 2405, 676, 357], [10000001, 4849, 1834, 522], [10000001, 10000001, 2740, 398], [10000001, 10000001, 5755, 739]]
#   rhc,      sa,   ga,  mimic
# [[10000001, 1236, 375, 77],
#  [10000001, 2422, 472, 188],
#  [10000001, 2405, 676, 357],
#  [10000001, 4849, 1834, 522],
#  [10000001, 10000001, 2740, 398],
#  [10000001, 10000001, 5755, 739]]
