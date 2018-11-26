# Frozen Lake

A grid of size 4x4 with movement of UDLR.

Agent tries to make it from top left to bottom right cell.

Holes in the grid cells. If the agent falls in a hole, episode ends
and reward is 0.  If agent reaches the destination, get reward of 1.0.

Slippery, so 33% chance of moving to the correct cell, 33% for slipping
left and 33% chance of slipping right *IN RELATION TO* the intended movement

## Why Value Iteration is so much faster and accurate than Cross-Entropy

(For this problem)

The cross-entropy solution required several hours and achieved around a 60% success rate.  But value iteration does it in a few seconds but gets a 80% success rate.

The stochastic outcome of our actions and the length of the episodes make it hard for the cross-entropy method to understand what was done right in the episode and which step was a mistake.

The value iteration works with individual values of state (or action) and incorporates the probabilistic outcome of actions naturally, by estimating probability and calculating the expected value. So, it's much simpler for the value iteration and requires much less data from the environment (which is called sample efficiency in RL).

Value iteration doesn't require full episodes to start learning.  However
since FrozenLake only gives a reward for successfully completing an episode,
there needs to be at least one successful episode for the algorithm to learn 
from the appropriate value table.  In TensorBoard, you can see that the reward
graph stays low until the first successful episode, after which it converges 
quickly.