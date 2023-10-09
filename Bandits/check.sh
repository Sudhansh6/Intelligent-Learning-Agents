#!/bin/sh

i=1

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-2.txt --algorithm epsilon-greedy-t1 --randomSeed 0 --epsilon 0.61 --scale 2 --threshold 0 --horizon 82
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-2.txt --algorithm ucb-t1 --randomSeed 499 --epsilon 0.02 --scale 2 --threshold 0 --horizon 27
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-1.txt --algorithm kl-ucb-t1 --randomSeed 16 --epsilon 0.02 --scale 2 --threshold 0 --horizon 39
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-2.txt --algorithm thompson-sampling-t1 --randomSeed 0 --epsilon 0.02 --scale 2 --threshold 0 --horizon 200
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task2/i-3.txt --algorithm ucb-t2 --randomSeed 451 --epsilon 0.02 --scale 0.9 --threshold 0 --horizon 77
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-1.txt --algorithm epsilon-greedy-t1 --randomSeed 5982 --epsilon 0.037 --scale 2 --threshold 0 --horizon 20000
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task3/i-2.txt --algorithm alg-t3 --randomSeed 33 --epsilon 0.02 --scale 2 --threshold 0 --horizon 241
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task4/i-1.txt --algorithm alg-t4 --randomSeed 33 --epsilon 0.02 --scale 2 --threshold 0.16 --horizon 95
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task3/i-1.txt --algorithm alg-t3 --randomSeed 10 --epsilon 0.02 --scale 2 --threshold 0 --horizon 53
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task2/i-5.txt --algorithm ucb-t2 --randomSeed 49 --epsilon 0.02 --scale 18 --threshold 0 --horizon 96395
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task1/i-1.txt --algorithm kl-ucb-t1 --randomSeed 2 --epsilon 0.02 --scale 2 --threshold 0 --horizon 201
sleep 1
i=$((i + 1))

echo "Test $i"
python bandit.py --instance ../instances/instances-task4/i-2.txt --algorithm alg-t4 --randomSeed 100 --epsilon 0.02 --scale 2 --threshold 0.75 --horizon 201
sleep 1
i=$((i + 1))
