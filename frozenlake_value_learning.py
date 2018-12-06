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
        on value iteration. Bellman equation.

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
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value
    
    def select_action(self, state):
        """
        Uses calc_action_value to make a decision about best action to take.
        Iterates over all actions in the environment and calculates value for every action.
        The action with the largest value wins and is returned as the one to take
        The function play_n_random_steps() introduces the necessary exploration. 
        So, the agent will behave greedily in regard to the value approximation.
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
        Uses select_action function to find best action to take and plays one full episode.
        This function is used to pay test eipsodes, during which we dont want to mess up the
        current state of the main environment used to gather random data.
        Loops over states accumulating reward for one episode.
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        Value iteration implementation.
        Loop over all states in the environment, then for every state we calculate the values
        for the states reachable from it, obtaining candidates for the value of the state.
        Then update the value of the current state with the max value of the action available 
        for the state
        """
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)\
                for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()