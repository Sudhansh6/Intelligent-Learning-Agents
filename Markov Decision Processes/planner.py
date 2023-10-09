import pulp, argparse
import numpy as np

'''
    Function to parse arguments
'''
def argparser():
    parser = argparse.ArgumentParser()
    # --mdp in, where in is a path to the instance file.
    parser.add_argument("--mdp", type=str, required=True)
    # --algorithm al
    parser.add_argument("--algorithm", type=str, default = 'vi')
    args = parser.parse_args()

    return args.mdp, args.algorithm

'''
    Function to get Q values from V
'''
def traverse(mdp, V):
    res = np.sum(mdp['T'][..., 1]*\
        ((mdp)['T'][..., 0] + mdp['gamma']*V), axis = 2)
    return res 

'''
    Implementation of Value Iteration
'''
def vi(mdp):   
    tol = 1e-11
    # Bstar operator: Returns B*(V) and the corresponding policy
    def BStar(V):
        res = np.zeros_like(V)
        Q = traverse(mdp, V)
        
        res = np.max(Q, axis = 1)
        return res

    VOld = np.ones(mdp['S'])
    V = np.zeros_like(VOld)
    pi = np.zeros_like(V, dtype = int)
    # Iterate till convergence
    while np.linalg.norm(V - VOld) > tol:
        VOld = V
        V = BStar(V)

    # Get pi from V
    Q = traverse(mdp, V)
    pi = np.argmax(Q, axis = 1)

    return V, pi

'''
    Implementation of Linear Programming
'''
def lp(mdp):
    lpFormulation = pulp.LpProblem("V", pulp.LpMinimize)
    
    # Create Variables
    variables, cost = np.zeros(mdp['S'], dtype = pulp.LpVariable), 0
    for s in range(mdp['S']):
        var = pulp.LpVariable('V' + str(s))
        variables[s] = var
        cost += var

    # Formulate Equations
    for s in range(mdp['S']):
        for a in range(mdp['A']):
            P = mdp['T'][s, a, ..., 1]
            R = mdp['T'][s, a, ..., 0]
            term = variables[s] >= pulp.lpSum(P*(R + mdp['gamma']*variables))
            lpFormulation += term

    lpFormulation += cost
    lpFormulation.writeLP("LP.lp")

    # Solve the equations to get the policy
    res = lpFormulation.solve(pulp.PULP_CBC_CMD(msg=0))
    V = np.zeros(mdp['S'], dtype = float)
    for v in lpFormulation.variables():
        V[int(v.name[1:])] = v.varValue

    # Get Policy from value function
    Q = traverse(mdp, V)
    pi = np.argmax(Q, axis = 1)

    return V, pi

'''
    Implementation of Howard Policy Iteration
'''
def hpi(mdp):  
    # Function to obtain Value from Policy
    def getV(pi):
        # Solve the Bellman Optimality Equations
        A, B = np.zeros((mdp['S'], mdp['S'])), np.zeros(mdp['S'])
        for s in range(mdp['S']):
            A[s, s] = 1
            P = mdp['T'][s, pi[s], ..., 1]
            R = mdp['T'][s, pi[s], ..., 0]
            A[s, ...] -= mdp['gamma']*P 
            B[s] = np.sum(P*R)
        return np.linalg.solve(A, B)

    pi, V = np.zeros(mdp['S'], int), None

    flag = True
    itr = 0
    # Iterate till convergence
    while flag:
        itr += 1
        flag = False
        V = getV(pi)
        Q = traverse(mdp, V)
        for s in range(mdp['S']):
            # Get an improvable action
            for a in range(mdp['A']):
                if pi[s] != a and Q[s, a] - V[s] > 1e-6:
                    pi[s] = a
                    flag = True
                    break
    return V, pi

'''
    Main driver code
'''
def main():
    instance, algo = argparser()
    with open(instance) as f:
        fileContent = f.readlines()
    mdp = {'S' : "", 'A' : "", 'type' : "", 'T' : [], 'end' : [], 'gamma' : ""}
    
    # Parse the MDP file
    for line in fileContent:
        words = line[:-1].split()
        if words[0] == 'numStates':
            mdp['S'] = int(words[1])

        elif words[0] == 'numActions':
            mdp['A'] = int(words[1])
            mdp['T'] = np.zeros([mdp['S'], mdp['A'], mdp['S'], 2])

        elif words[0] == 'end':
            mdp['end'] = [int(ep) for ep in words[1:]]

        elif words[0] == 'transition':
            if len(words) != 6:
                print("Error while parsing transitions")
                exit(1)
            # 1 is rew, 2 is p
            mdp['T'][int(words[1]), int(words[2]), int(words[3]), 0] = float(words[4])
            mdp['T'][int(words[1]), int(words[2]), int(words[3]), 1] = float(words[5])
        
        elif words[0] == 'discount':
            mdp['gamma'] = float(words[1])

        elif words[0] == 'mdptype':
            mdp['type'] = words[1]

    # Preliminary check
    if mdp['gamma'] == 1 and mdp['type'] == 'continuous':
        print("Gamma cannot be 1 for continuous MDPs")
        exit(1)

    # Check the algorithm
    V, pi = [], []

    if algo == 'hpi':
        V, pi = hpi(mdp)
    elif algo == 'lp':
        V, pi = lp(mdp)
    elif algo == 'vi':
        V, pi = vi(mdp)

    for i in range(len(V)):
        print('{:.6f}'.format(round(V[i], 6)) + "\t" + str(int(pi[i])))

if __name__ == '__main__':
    main()
