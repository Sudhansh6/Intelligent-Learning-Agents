import sys
import math
import numpy as np
import argparse
import warnings
sys.path.append('.')
from _impl import *

algos = ["epsilon-greedy-t1", 'ucb-t1', 'kl-ucb-t1',
         'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']

# Quick copypaste
# python bandit.py --instance ../instances/instances-task1/i-1.txt --algorithm epsilon-greedy-t1 --randomSeed 4 --epsilon 0.4 --horizon 10
def argparser():
    parser = argparse.ArgumentParser()
    # --instance in, where in is a path to the instance file.
    parser.add_argument("--instance", type=str, required=True)
    # --algorithm al, where al is one of epsilon-greedy-t1, ucb-t1, kl-ucb-t1, thompson-sampling-t1, ucb-t2, alg-t3, alg-t4.
    parser.add_argument("--algorithm", type=str, required=True)
    # --randomSeed rs, where rs is a non-negative integer.
    parser.add_argument("--randomSeed", type=int, required=True)
    # --epsilon ep, where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02.
    parser.add_argument("--epsilon", type=float, required=True)
    # --horizon hz, where hz is a non-negative integer.
    parser.add_argument("--horizon", type=int, required=True)
    # --scale c, where c is a positive real number. The parameter is only relevant for Task 2; for other tasks pass --scale 2.
    parser.add_argument("--scale", type = float, default = 2, required = True)
    # --threshold th, where th is a number in [0, 1]. The parameter is only relevant for Task 4; for other tasks pass --threshold 0.
    parser.add_argument("--threshold", type = float, default = 0, required = True)
    args = parser.parse_args()
    return args


args = argparser()

# Get the instance attributes
with open(args.instance) as f:
    ps = f.readlines()

# Filter the probabilities from the file
flag = True
try:
    L = np.array(ps, dtype=np.float16)
except:
    flag = False
    L = np.array([np.array(i.split(' '), dtype=np.float32) for i in ps])

# Get the algorithm from the _impl file
reg = "Error!"
Highs = 0
if args.algorithm == algos[0] and flag:
    reg = epsilon_greedy_t1(L, args.epsilon, args.horizon, args.randomSeed)
elif args.algorithm == algos[1] and flag:
    reg = ucb_t1(L, args.horizon, args.randomSeed, 2)
elif args.algorithm == algos[2] and flag:
    reg = kl_ucb_t1(L, args.horizon, args.randomSeed)
elif args.algorithm == algos[3] and flag:
    reg = thompson_sampling_t1(L, args.horizon, args.randomSeed)
elif(args.algorithm == algos[4]):
    reg = ucb_t1(L, args.horizon, args.randomSeed, args.scale)
elif(args.algorithm == algos[5]):
    reg = alg_t3(L, args.horizon, args.randomSeed)
elif(args.algorithm == algos[6]):
    header = L[0] > args.threshold
    L1 = np.dot(L[1:], header)
    reg = 0
    Highs = thompson_sampling_t1(L1, args.horizon, args.randomSeed, task_4=True)
else:
    print("Error")
    exit(1)

# instance, algorithm, random seed, epsilon, scale, threshold, horizon, REG, HIGHS 
output = f'{args.instance}, {args.algorithm}, {args.randomSeed}, {args.epsilon}, {args.scale},\
 {args.threshold}, {args.horizon}, {reg}, {Highs}'
print(output)