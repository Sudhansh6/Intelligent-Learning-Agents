import numpy as np
import argparse
import matplotlib.pyplot as plt 

'''
Implementation for the epsilon Greedy 3 algorithm
Params:
    L = List of probabilities of each bandit arm from the instance
    epsilon = The exploration factor
    T = Horizon
    seed = Random seed to be used by numpy for the experiment
    debug = Whether or not to give the complete list of regrets
Output:
    Regret (as a list if debug true)
'''
def epsilon_greedy_t1(L, epsilon, T, seed, debug = False):
    # Set the seed
    np.random.seed(seed)

    delta = 1e-8  # Initialising to a small value to avoid zeros
    scores = np.ones(len(L), np.float32) * delta
    tries = np.ones(len(L), np.float32) * delta

    # Stores Regret
    REG = 0

    # Simulation of bandit pulls
    if debug:
        regrets = np.zeros(T)

    for t in range(T):
        # Sample a uniform number in 0 to 1
        random = np.random.random_sample()
        means = scores / tries

        # With probability 1 - epsilon, choose the highest empirical mean arm
        if random > epsilon:
            arm = np.argmax(means)
        # Otherwise, sample a random arm
        else:
            arm = np.random.choice(range(len(L)))

        # Update trials and scores accordingly
        tries[arm] += 1
        scores[arm] += (np.random.random_sample() < L[arm])

        # Calculating regret
        REW = np.sum(scores)
        REG = (t + 1)*np.max(L) - REW
        if debug:
            regrets[t] = REG

    if debug:
        return regrets
    return REG

'''
Implementation for the UCB algorithm
Params:
    L = List of probabilities of each bandit arm from the instance
    T = Horizon
    seed = Random seed to be used by numpy for the experiment
    c = The scale factor inside the exploration factor of UCB expression
    debug = Whether or not to give the complete list of regrets
Output:
    Regret (as a list if debug true)
'''
def ucb_t1(L, T, seed, c, debug = False):
    # Set the random seed
    np.random.seed(seed)
    
    scores = np.zeros(len(L))
    tries = np.zeros(len(L))
    if debug:
        regrets = np.zeros(T)

    # Simulation of bandit pulls
    for t in range(T):
        # Pull all arms once before applying UCB bounds
        not_tried = np.where(tries == 0)[0]
        
        if len(not_tried) != 0:
            arm = not_tried[0]
        else:
        # Find ucb scores for each arm
            ucbs = (scores / tries) + np.sqrt(c * np.log(t) / tries)
            arm = np.argmax(ucbs)
        
        # Update accordingly
        tries[arm] += 1
        scores[arm] += (np.random.random_sample() < L[arm])
        
        # Calculating regret
        REW = np.sum(scores)
        REG = np.max(L) * (t + 1) - REW
        if debug:
            regrets[t] = REG

    if debug:
        return regrets
    return REG

def kl_solver(c, p, u, t):
    start = p
    end = 1
    mid = (start + end) / 2
    cons = np.log(t) +  c * np.log(np.log(t))
    rhs = cons / u
    idx = 0
    tol = 1e-3

    while abs(start - end) > tol:
        idx += 1
        if p == 0: 
            val = -np.log(1 - mid)
        elif p == 1:
            val = -np.log(mid)
        else:
            val = p * np.log(p / mid) + (1 - p) * np.log((1 - p) / (1 - mid))
        if val > rhs:
            end = mid
        else:
            start = mid
        mid = (start + end) / 2
    return mid

'''
Implementation for the KL-UCB
Params:
    L = List of probabilities of each bandit arm from the instance
    T = Horizon
    seed = Random seed to be used by numpy for the experiment
    debug = Whether or not to give the complete list of regrets
Output:
    Regret (as a list if debug true)
'''
def kl_ucb_t1(L, T, seed, debug = False):
    # Setting random seed
    np.random.seed(seed)

    scores = np.zeros(len(L))
    tries = np.zeros(len(L))

    # The scale factor in KL_UCB expression
    c = 3
    
    if debug:
        regrets = np.zeros(T)
    # Simulation of bandit pulls
    for t in range(T):
        # Pulling every arm once before applying kl-ucb
        not_tried = np.where(tries == 0)[0]
        
        if len(not_tried) != 0:
            arm = not_tried[0]
        else:
            # Calculating kl-ucb
            kl_ucb = np.zeros(len(L))
            starts = scores/tries
            # Performing binary search to calculate 'q' for each bandit arm
            for i in range(len(L)):
                kl_ucb[i] = kl_solver(c, starts[i], tries[i], t)

            # Get the max KL-UCB value
            arm = np.argmax(kl_ucb)

        # Update tries and scores accordingly
        tries[arm] += 1
        scores[arm] += (np.random.random_sample() < L[arm])
        
        # Calculating regret
        REW = np.sum(scores)
        REG = np.max(L) * (t + 1) - REW
        if debug:
            regrets[t] = REG

    if debug:
        return regrets
    return REG

'''
Implementation for the Thompson sampling algorithm
Params:
    L = List of probabilities of each bandit arm from the instance
    T = Horizon
    seed = Random seed to be used by numpy for the experiment
    debug = Whether or not to give the complete list of regrets
Output:
    Regret (as a list if debug true)
'''
def thompson_sampling_t1(L, T, seed, debug = False, task_4 = False):
    # Setting the seed
    np.random.seed(seed)

    scores = np.zeros(len(L))
    tries = np.zeros(len(L))

    if debug:
        regrets = np.zeros(T)
        highs = np.zeros(T)
    
    # Simulation of bandit pulls
    for t in range(T):
        # Generate samples from the beta distribution pertaining to each arm
        samples = np.random.beta(scores + 1, tries - scores + 1)
        # Get the maximum sample
        arm = np.argmax(samples)

        # Update accordingly
        tries[arm] += 1
        scores[arm] += (np.random.random_sample() < L[arm])
        
        # Calculating regret
        REW = np.sum(scores)
        REG = np.max(L) * (t + 1) - REW
        if debug:
            regrets[t] = REG
            highs[t] = REW

    if debug and task_4:
        return regrets, highs
    if debug:
        return regrets
    if task_4:
        return np.sum(scores)   
    return REG

'''
Implementation of KL_UCB for task 3
Params:
    L = List of expected reward of each bandit arm from the instance
    T = horizon
    seed = Random seed to be used by numpy for the experiment
    debug = Whether or not to give the complete list of regrets
Output:
    Regret (as a list if debug true)
'''
def alg_t3(L, T, seed, debug = False):
    # Setting the seed
    np.random.seed(seed)
    
    # Scaling factor in KL_UCB expression
    c = 3

    # Calculating the Maximum expected reward
    header = L[0]
    L1 = np.dot(L[1:], header)

    M = np.max(L1)
    
    scores = np.zeros(len(L1))
    tries = np.zeros(len(L1)) 
    categorical = np.zeros(np.shape(L[1:]))
    
    if debug:
        regrets = np.zeros(T)
    REW = 0

    # Simulation of bandit pulls
    for t in range(T):
        # Calculating samples
        samples = np.zeros(len(L1))
        for i in range(len(L1)):
            samples[i] = np.dot(header, np.random.dirichlet(categorical[i] + 1))
        # Get the max sample
        arm = np.argmax(samples)

        # Update accordingly
        tries[arm] += 1
        score = np.random.choice(range(len(header)), p = L[arm + 1])
        categorical[arm][score] += 1
        scores[arm] += header[score]

        # Calculating regret
        REW += header[score]
        REG = M * (t + 1) - REW
        if debug:
            regrets[t] = REG

    if debug:
        return regrets

    return REG

if __name__=='__main__':
    algos = ["epsilon-greedy-t1", 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1']
    parser = argparse.ArgumentParser()
    # --instance in, where in is a path to the instance file.
    parser.add_argument("--instance", type=str, required=True)
    # --randomSeed rs, where rs is a non-negative integer.
    parser.add_argument("--randomSeed", type=int, required=True)
    # --epsilon ep, where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02.
    parser.add_argument("--epsilon", type=float, required=True)
    # --horizon hz, where hz is a non-negative integer.
    parser.add_argument("--horizon", type=int, required=True)
    # --scale c, where c is a positive real number. The parameter is only relevant for Task 2; for other tasks pass --scale 2.
    parser.add_argument("--scale", type = float, default = 2)
    # --threshold th, where th is a number in [0, 1]. The parameter is only relevant for Task 4; for other tasks pass --threshold 0.
    parser.add_argument("--threshold", type = float, default = 0, required = True)
    args = parser.parse_args()
    # Get the instance attributes
    with open(args.instance) as f:
        ps = f.readlines()

    flag = True
    try:
        L = np.array(ps, dtype=np.float16)
    except:
        flag = False
        L = np.array([np.array(i.split(' '), dtype=np.float16) for i in ps])

    '''
         This is for task 1 check
    '''
    # Get the algorithm
    # smooth_factor = 100
    # reg = np.array([\
    # epsilon_greedy_t1(L, args.epsilon, args.horizon, args.randomSeed, True), \
    # ucb_t1(L, args.horizon, args.randomSeed, args.scale, True), \
    # kl_ucb_t1(L, args.horizon, args.randomSeed, True)]) #, \
    # # thompson_sampling_t1(L, args.epsilon, args.horizon, args.randomSeed, True)])
    # for i in range(smooth_factor):
    #     seed = i
    #     reg = reg + np.array([\
    #     epsilon_greedy_t1(L, args.epsilon, args.horizon, seed, True) , \
    #     ucb_t1(L, args.horizon, seed, args.scale, True), \
    #     kl_ucb_t1(L, args.horizon, seed, True)]) #, \
    #     # thompson_sampling_t1(L, args.horizon, seed, True)])
    # reg = reg/ (smooth_factor + 1)
    # plt.figure()
    # for k in range(3):
    #     plt.plot(range(args.horizon), reg[k])
    # plt.legend(algos)
    # plt.xlabel('Time')
    # plt.ylabel('Regret')
    # plt.title('For instance ' + args.instance)
    # plt.show()
    # plt.savefig("debug1.png")

    '''
        This is for task 2 check
    '''
    # scales = np.array(range(2, 31, 5))/100
    # seeds = range(10)
    # reg = np.zeros((len(scales), args.horizon))
    # for i in range(len(scales)):
    #     scale = scales[i]
    #     for seed in seeds:
    #         reg[i] += np.array(ucb_t1(L, args.horizon, seed, scale, True))
    #     reg[i] /= len(seeds)
    #     plt.plot(range(args.horizon), reg[i])
    # plt.legend(scales)
    # plt.xlabel('Time')
    # plt.ylabel('Regret')
    # plt.title('For instance ' + args.instance)
    # plt.savefig("debug2.png")
    
    '''
         This is task 3 check
    '''
    # Get the algorithm
    # smooth_factor = 50
    # t = alg_t3(L, args.horizon, args.randomSeed, True)
    # reg = np.array(t)
    # for i in range(smooth_factor):
    #     seed = i
    #     t = alg_t3(L, args.horizon, seed, True)
    #     reg = reg + np.array(t)
    # reg = reg/ (smooth_factor + 1)
    # plt.plot(range(args.horizon), reg)
    # plt.legend(['alg-t3'])
    # plt.xlabel('Time')
    # plt.ylabel('Regret')
    # plt.title('For instance ' + args.instance)
    # plt.show()
    # plt.savefig("debug3.png")

    '''
         This is task 4 check
    '''
    # # Get the algorithm
    # smooth_factor = 50

    # header = L[0] > args.threshold
    # L = np.dot(L[1:], header)
    # reg = thompson_sampling_t1(L, args.horizon, args.randomSeed)
    # t = thompson_sampling_t1(L, args.horizon, args.randomSeed, True)
    # reg = np.array(t)
    # for i in range(smooth_factor):
    #     seed = i
    #     t = thompson_sampling_t1(L, args.horizon, seed, True)
    #     reg = reg + np.array(t)
    # reg = reg/ (smooth_factor + 1)
    # plt.figure()
    # plt.plot(range(args.horizon), reg)
    # plt.legend(['alg-t4'])
    # plt.xlabel('Time')
    # plt.ylabel('Regret')
    # plt.title('For instance ' + args.instance)
    # plt.show()
    # plt.savefig("debug4.png")
