# Frozen Lake

A grid of size 4x4 with movement of UDLR.

Agent tries to make it from top left to bottom right cell.

Holes in the grid cells. If the agent falls in a hole, episode ends
and reward is 0.  If agent reaches the destination, get reward of 1.0.

Slippery, so 33% chance of moving to the correct cell, 33% for slipping
left and 33% chance of slipping right *IN RELATION TO* the intended movement
