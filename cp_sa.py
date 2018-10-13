import sys
sys.path.append("/Users/Quintus/OneDrive/Code/python/omscs/machine learning/assignment 2/ABAGAIL/ABAGAIL.jar")
import os
import csv
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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array


# ---------------------------------------------------------------
def FixedIterTrainer(ef, oa, n_iters):
    count = 0
    f_hist = []
    while count < n_iters:
        oa.train()
        if count in xrange(1, n_iters, 100):
            cur_value = ef.value(oa.getOptimal())
            f_hist.append(cur_value)
        count += 1
        # print count
    return f_hist


def write_hist_csv(hist, oa_name):
    with open('cp_'+oa_name+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(hist)


# ---------------------------------------------------------------
N = 60
T = N / 10
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


# ---------------------------------------------------------------
N_ITERS = 10001

# SA

# for cool_fac in [0.9, 0.92, 0.94, 0.96, 0.98]:
#     print cool_fac
#     fit_hist = []
#     time_hist = []
#     for i in xrange(30):
#         hcp = GenericHillClimbingProblem(ef, odd, nf)
#         sa = SimulatedAnnealing(1E12, cool_fac, hcp)
#
#         fh = FixedIterTrainer(ef, sa, N_ITERS)
#         fit_hist.append(fh)
#     write_hist_csv(fit_hist, 'fitness_sa_cf_'+str(cool_fac))
#


for power in range(1, 12, 2):
    fit_hist = []
    time_hist = []
    for i in xrange(30):
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(10**power, 0.94, hcp)

        fh = FixedIterTrainer(ef, sa, N_ITERS)
        fit_hist.append(fh)
    write_hist_csv(fit_hist, 'fitness_sa_t_'+str(power))

