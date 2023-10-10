'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.env = gym.make('MountainCar-v0')
         # Custom attributes
        self.d_granularity = 8
        self.v_granularity = 8
        self.d_granule = (self.env.observation_space.high[0] - self.env.observation_space.low[0])/self.d_granularity
        self.v_granule = (self.env.observation_space.high[1] - self.env.observation_space.low[1])/self.v_granularity

        self.num_tilings = 8
        self.d_tile_width = self.d_granule
        self.v_tile_width = self.v_granule
        self.d_tiles = self.d_granularity + 1
        self.v_tiles = self.v_granularity + 1
        self.offset = 1/self.num_tilings

        # Predefined attributes
        self.epsilon_T1 = 0.02
        self.epsilon_T2 = 0.03
        self.learning_rate_T1 = 1e-2
        self.learning_rate_T2 = 1e-2
        self.weights_T1 = np.zeros(((self.d_granularity + 1)*(self.v_granularity + 1), 3))
        self.weights_T2 = np.zeros((self.num_tilings*self.d_tiles*self.v_tiles, 3))
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]    

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''
    def get_table_features(self, obs):
        # Find the discretized state value
        a = int((obs[0] - self.lower_bounds[0])/self.d_granule)
        b = int((obs[1] - self.lower_bounds[1])/self.v_granule)
        # Set the corresponding entry to 1
        entry = np.zeros(((self.d_granularity + 1)*(self.v_granularity + 1), 1))
        entry[a*(self.d_granularity + 1) + b] = 1
        return entry

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''
    def get_better_features(self, obs):
        # State has dimension of total number of tiles
        state = np.zeros((self.num_tilings*self.d_tiles*self.v_tiles, 1))
        # Find the tiles overlapping with the input
        d_array = np.array([range(self.num_tilings)])*self.offset*self.d_tile_width
        v_array = np.array([range(self.num_tilings)])*self.offset*self.v_tile_width
        a = (((obs[0] - self.lower_bounds[0] - d_array)/self.d_tile_width).astype('int') + 1)*self.d_tiles
        b = ((obs[1] - self.lower_bounds[1] - v_array)/self.v_tile_width).astype('int') + 1
        c = np.array([range(self.num_tilings)])*self.d_tiles*self.v_tiles
        # Set the corresponding tiles in the state to 1
        state[a + b + c] = 1
        return state

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''
    def choose_action(self, state, weights, epsilon):
        # Get action using epsilon greedy
        if np.random.rand() < epsilon:
            return np.random.choice(range(3))
        # Get max Q(s, a)
        best_action = np.argmax(np.matmul(weights.T, state))
        return best_action

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''
    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        # Get new and old Q values
        new_q = np.matmul(weights[:, new_action:new_action + 1].T, new_state)
        old_q = np.matmul(weights[:, action:action + 1].T, state)  
        # Find the target based on SARSA rule
        target = reward + self.discount*new_q
        # Update the corresponding weights
        weights[:, action:action  + 1] += learning_rate*float(target - old_q)*state
        return weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
            # print(f"Epoch: {e:05}, Reward = {-t}", end = '\r')     
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))
