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
    start = time.time()
    count = 0
    f_hist = []
    t_hist = []
    while count < n_iters:
        oa.train()
        if count in xrange(1, n_iters, 100):
            cur_value = ef.value(oa.getOptimal())
            f_hist.append(cur_value)
            t_hist.append(time.time() - start)
        count += 1
        # print count
    return f_hist, t_hist


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
N_ITERS = 100001

# RHC
fit_hist = []
time_hist = []
for i in xrange(30):
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)

    fh, th = FixedIterTrainer(ef, rhc, N_ITERS)
    fit_hist.append(fh)
    time_hist.append(th)
write_hist_csv(fit_hist, 'fitness_rhc')
write_hist_csv(time_hist, 'time_rhc')

# SA
fit_hist = []
time_hist = []
for i in xrange(30):
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    sa = SimulatedAnnealing(1E12, .95, hcp)

    fh, th = FixedIterTrainer(ef, sa, N_ITERS)
    fit_hist.append(fh)
    time_hist.append(th)
write_hist_csv(fit_hist, 'fitness_sa')
write_hist_csv(time_hist, 'time_sa')

# GA
fit_hist = []
time_hist = []
for i in xrange(30):
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    ga = StandardGeneticAlgorithm(200, 100, 20, gap)

    fh, th = FixedIterTrainer(ef, ga, N_ITERS)
    fit_hist.append(fh)
    time_hist.append(th)
write_hist_csv(fit_hist, 'fitness_ga')
write_hist_csv(time_hist, 'time_ga')

# MIMIC
start = time.time()
fit_hist = []
time_hist = []
for i in xrange(30):
    print i
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
    mimic = MIMIC(200, 20, pop)

    fh, th = FixedIterTrainer(ef, mimic, N_ITERS)
    fit_hist.append(fh)
    time_hist.append(th)
    print time.time() - start, 'seconds'

write_hist_csv(fit_hist, 'fitness_mimic')
write_hist_csv(time_hist, 'time_mimic')

# 6941 secs
