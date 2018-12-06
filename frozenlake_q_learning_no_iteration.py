"""
Tabular Q Learning
Iterating over every state in the state space can be very expensive/impossible.
So, just update the values of states with the ones seen in the environment.

1. Start with an empty table for Q(s, a), mapping states to values of actions.
2. Obtain tuple (s, a, r, s'). s' being the new state.  Decide which action to take here.
3. Update Q(s,a) using Bellman equation
4. Repeat from Step 2 if necessary after checking for convergence.

Instead of replacing old Q values with new ones, use 'blending' with a weighted average 
between the old and new values.  Weighted by the learning rate alpha.
Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma * maxQ(s', a'))
"""
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    """
    Much simpler than before, now that there is no need to track history of rewards
    and transition counters, just the value. This is can be a big help in larger environments. 
    """
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        """
        Obtains next transition from environment.
        Sample random action and return tuple of old state, action, reward, and the new state.
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        """
        Takes state and takes best action (one with largest value in table).
        Used in play_episode test method to evaluate policy with current value table.
        And in value_update which performs value update to get the value of the next state
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        """
        Update values table using one step from the environment.
        Calculate Bellman approximation for state s and action a by summing 
        the immediate reward with the discounted value of the next state.
        The last line is the blending calculation which results in the new approximation of s and 
        a that takes the new and old state/action pairs.
        """
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA

    def play_episode(self, env):
        """
        Plays one full episode using the test environment. 
        This method doesn't alter the value table, only uses it to find the best action to take. 
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

# Initialize test environment, agent, and summary writer.
# In the loop, do one step in the env and perform a value update using hte obtained data.
# Then test current policy by playing several test episodes. 
# If a good reward is obtained, stop training.
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

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
