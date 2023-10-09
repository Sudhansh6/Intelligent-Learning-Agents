import numpy as np 
import argparse

'''
    Function to parse arguments
'''
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--states", type=str, required=True)
    args = parser.parse_args()

    return args.policy, args.states

'''
    Checks if the game has terminated
     - Return -1 if not terminated
     - 0 if draw
     - 1 if player 1 won
     - 2 if player 2 won
'''
def checkTerminal(state):
    for i in range(3):
        # Check rows
        p2 = np.sum([state[3*i + j] == '2' for j in range(3)])
        p1 = np.sum([state[3*i + j] == '1' for j in range(3)])
        if p1 == 3:
            return 2
        if p2 == 3:
            return 1
        # Check columns
        p2 = np.sum([state[i + 3*j] == '2' for j in range(3)])
        p1 = np.sum([state[i + 3*j] == '1' for j in range(3)])
        if p1 == 3:
            return 2
        if p2 == 3:
            return 1
        # Check diagonals
        p2 = np.sum([state[3*j + j] == '2' for j in range(3)])
        p1 = np.sum([state[3*j + j] == '1' for j in range(3)])
        if p1 == 3:
            return 2
        if p2 == 3:
            return 1
        p2 = np.sum([state[3*(2 - j) + j] == '2' for j in range(3)])
        p1 = np.sum([state[3*(2 - j) + j] == '1' for j in range(3)])
        if p1 == 3:
            return 2
        if p2 == 3:
            return 1
    if sum([state[i] == '0' for i in range(9)]) == 0:
        return 0
    return -1

'''
    Main driver code
'''
def main():
    policyFile, statesFile = argparser()
    with open(statesFile) as f:
        fileContent = f.readlines()
    states = [line[:-1] for line in fileContent]
    
    with open(policyFile) as f:
        fileContent = f.readlines()
    
    policy = {}
    if fileContent[0] == "1\n":
        player = 2
        opp = 1
    else:
        player = 1
        opp = 2

    for line in fileContent[1:]:
        words = line[:-1].split()
        policy[words[0]] = [float(p) for p in words[1:]]
    
    numStates = len(states)
    numActions = 9
    print('numStates', numStates + 1) # one for terminal state win
    print('numActions', numActions)

    # Find end states
    end = [numStates] 

    print('end', *end)

    for i in range(numStates):
        # Take player's action
        for j in range(numActions):
            state = states[i]
            P = [0, 0]

            # Check if action is valid
            if state[j] != '0':
                continue

            # Get new state
            state = state[:j] + str(player) + state[j + 1:]
            
            # Check if state is terminal
            terminal = checkTerminal(state)
            if terminal != -1:
                print('transition', i, j, numStates, 0, 1)
                continue

            # Take opponent's action
            for oppAction in range(9):
                if state[oppAction] != '0':
                    continue

                # Get new state and probability
                p = policy[state][oppAction]
                newState = state[:oppAction] + str(opp) + state[oppAction + 1:]

                # Check if the state is terminal
                terminal = checkTerminal(newState)
                if terminal != -1:
                    P[int(terminal == player)] += p
                    continue    

                # Get index of new state
                nidx = states.index(newState)
                if p == 0.0:
                    continue
                print('transition', i, j, nidx, 0, p)
            
            # Print the terminal state transitions
            for rew in range(2):
                if P[rew] == 0.0:
                    continue
                print('transition', i, j, numStates,\
                    rew  , P[rew])
    print('mdptype', 'episodic')
    print('discount', '1')

if __name__ == '__main__':
    main()
