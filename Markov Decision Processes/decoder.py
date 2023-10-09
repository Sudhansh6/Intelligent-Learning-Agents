import numpy as np 
import argparse

'''
    Function to parse arguments
'''
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-policy", type=str, required=True)
    parser.add_argument("--states", type=str, required=True)
    # Assuming argument is given as an integer
    parser.add_argument("--player-id", type=int, required=True)

    args = parser.parse_args()
    return args.value_policy, args.states, args.player_id

'''
    Main driver loop
'''
def main():
    vp, statesFile, player = argparser()
    # Print the player in the first line
    print(player)
    with open(statesFile) as f:
        fileContent = f.readlines()
    states = [line[:-1] for line in fileContent]
     
    with open(vp) as f:
        fileContent = f.readlines()
    numStates = len(states)
    numActions = 9

    # Check the action for each state and make the entry 1
    for i in range(numStates):
        words = fileContent[i][:-1].split()
        T = np.zeros(numActions)
        state = states[i]

        # If an invalid action, change to a valid action
        if state[int(words[1])] != '0':
            for a in range(9):
                if state[a] == '0':
                    T[int(a)] = 1
                    continue
        else:
            T[int(words[1])] = 1

        print(states[i], *T)

if __name__ == '__main__':
    main()
