## Markov Decision Processes
Markov Decision Processes (MDPs) are mathematical frameworks used in the field of reinforcement learning and decision-making. They model situations where an agent interacts with an environment, taking actions to maximize a cumulative reward over time
This brief documentation provides an overview of MDPs and optimal policy algorithms motivated on the anti tic-tac-toe game.

**Definition**: An MDP is a tuple (S, A, P, R), where:

**S**: Set of states representing possible situations or configurations.
**A**: Set of actions representing possible decisions or choices the agent can make.
**P**: Transition probability function, which specifies the probability of transitioning from one state to another when an action is taken.
**R**: Reward function, which defines the immediate reward the agent receives upon transitioning from one state to another due to an action.


### Anti tic-tac-toe
Anti Tic-Tac-Toe is a fun and strategic variation of the classic Tic-Tac-Toe game. In this version of the game, the objective is reversed: players aim to avoid forming a line of three of their own symbols, rather than trying to achieve a winning line. 

The optimal policy is found using three methods
### Value Iteration
Value Iteration is an iterative algorithm used to find the optimal value function and, consequently, the optimal policy. It works by iteratively improving the estimated values of states until they converge to their true values.

**Steps**:
- Initialize the value function arbitrarily.
- Iteratively update the value function using the Bellman equation until convergence.
- Extract the optimal policy based on the computed value function.

### Howard's policy iteration
Howard's Policy Iteration is an improvement over standard Policy Iteration, making it more efficient by using a truncated policy evaluation step.

**Steps**:
- Initialization: Start with an initial policy.
- Policy Evaluation: Unlike standard Policy Iteration, Howard's method uses truncated policy evaluation, which involves performing a fixed number of iterations of the Bellman equation to estimate the value function under the current policy.
- Policy Improvement: Improve the policy by selecting actions that maximize the expected value based on the estimated value function.
- Convergence: Repeat the policy evaluation and improvement steps until the policy no longer changes significantly.

### Linear Programming
Linear Programming is a mathematical optimization technique used for finding the best possible outcome in a linear relationship, subject to linear constraints. It has applications in various fields, including operations research, economics, and engineering.

**Key Components**:
- Objective Function: A linear equation that needs to be maximized or minimized. It typically represents a quantity of interest, such as profit or cost.
- Decision Variables: Variables that the LP problem seeks to determine. These variables often represent the quantities to be decided upon.
- Constraints: Linear inequalities that restrict the possible values of decision variables. Constraints define the feasible region.
- Feasible Region: The set of values for decision variables that satisfy all constraints.
- Optimal Solution: The combination of decision variables that maximizes or minimizes the objective function while staying within the feasible region.
 
**Solving an LP Problem**
- Formulate the Problem: Define the objective function and constraints based on the real-world problem.
- Convert to Standard Form: Standardize the LP problem by converting inequalities to equations, introducing slack or surplus variables.
- Solve: Use LP solvers (e.g., Simplex method or interior-point methods) to find the optimal solution.
- Interpret Results: Analyze the solution to interpret the values of decision variables and the optimized objective function value.

## Results
