"""
Value Iteration 

Central data structures:
* Reward table: A dictionary with the keys of 'source state' + 'action' + 'target state'
    * The value is obtained from the immediate reward
* Transitions table: A dictionary that keeps counters of the experienced transitions
    * Key is composite 'state' + 'action'
    * Value is another dictionary that maps the target state into a count of times that it has been seen
    * If in state 0, action 1 is executed 10 times, and it leads to state4 3 times, and state5 7 times
        * Results in key of (0, 1) with value of {4: 3, 5: 7}
    * This table will be used to estimate the probabilities of transitions
* value table: A dictionary that maps a state into the calculated value of this state

Overview:
In the loop, play 100 random steps and populate Reward and Transitions tables.
After those 100 steps, perform a value iteration loop to update value table.
Then play several full episodes with the learned value, to reach an average 0.8 score.
During test episodes, also update reward and transition tables to get as much data as possible.
"""
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
    """
    Gather random experience and update reward/transition tables.
    Instead of waiting for the full episodes, just do N steps.  
    Cross-entropy can only learn on full epsiodes
    """
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
    """
    Calculates value of the action from the state using transition , reward and values tables.
    This value is used to select the best action in the current state, and to calculate the new value of the state
    on value iteration.

    1. Extract transition counters from given state and action from transition table.  
       Sum all counters to obtain the total count of times we've executed the action from the state
       This value is used to go from individual counter to probability
    2. Iterate every target state that the action creates, and calculate its contribution into
       the total action value using the Bellman equation.  This contribute comprises of
       the immediate reward plus discounted value for the target state.
       We multiply this sum to the probability fo this transition and add the result to the 
       final action value.
    """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.-values[tgt_state])
        return action_value
    