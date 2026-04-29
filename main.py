import pickle
import numpy as np
import matplotlib.pyplot as plt


### states are vectors with 6 dimensions:
# x-y position of the "mobile robot"
# x-y position of goal1
# x-y position of goal2
# [x_robot, y_robot, x_goal1, y_goal1, x_goal2, y_goal2]

### actions are vectors with 2 dimensions:
# [delta_x_robot, delta_y_robot]

# dynamics: (state, action) -> next_state
def dynamics(state, action):
	# the two goals are static
	next_state = np.copy(state)
	# move the robot by the action
	next_state[0:2] += action
	return next_state

# get an action that goes directly to the closest goal
def get_action(state):
	goal = state[2:4]
	if np.linalg.norm(state[0:2] - state[4:6]) < np.linalg.norm(state[0:2] - state[2:4]):
		goal = state[4:6]
	action_vec = goal - state[0:2]
	if np.linalg.norm(action_vec) > 1.0:
		action_vec /= np.linalg.norm(action_vec)
	return action_vec

# add a (state, action) pair to the dataset
def add_to_dataset(dataset, state, action):
	dataset.append(state.tolist() + action.tolist())

# visualize a (state, action) pair
def plot_state_action(state, action):
	plt.plot(state[0], state[1], 'bo')
	plt.plot(state[2], state[3], 'ro')
	plt.plot(state[4], state[5], 'go')
	plt.plot([state[0], state[0]+action[0]], [state[1], state[1]+action[1]], 'b-')
	plt.axis("equal")
	plt.savefig("state_action.png")


### task description
# the mobile robot is moving in an x-y
# we want to train this robot to move to the closest goal
# the goal locations are randomized, and could be different at each test run
# optimize the dataset of (state, action) pairs to teach this task

dataset = []
### first pass solution (you will want to improve on this...)
for _ in range(10):
	# pick a state uniformly at random
	state = np.random.uniform(-10, 10, 6)
	# calculate action towards closest goal
	action = get_action(state)
	# add the state-action pair to the dataset
	add_to_dataset(dataset, state, action)
	# plot the state-action pair
	plot_state_action(state, action)

pickle.dump(dataset, open("dataset.pkl", "wb"))
print("dataset has this many state-action pairs:", len(dataset))