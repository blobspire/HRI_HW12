import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg") # Headless plotting for autoresearch optimization
import matplotlib.pyplot as plt


### states are vectors with 4 dimensions:
# x-y position of the mobile robot
# x-y position of the goal
# [x_robot, y_robot, x_goal, y_goal]

### actions are vectors with 2 dimensions:
# [delta_x_robot, delta_y_robot]

### task description
# the mobile robot is moving in an x-y plane
# we want to train this robot to move to the goal
# the goal locations are randomized, and could be different at each test run
# optimize the dataset of (state, action) pairs to teach this task

# dynamics: (state, action) -> next_state
def dynamics(state, action):
	# the goal is static
	next_state = np.copy(state)
	# move the robot by the action
	next_state[0:2] += action
	return next_state

# get an action that goes directly to the goal
def get_action(state):
	action_vec = state[2:4] - state[0:2]
	if np.linalg.norm(action_vec) > 1.0:
		action_vec /= np.linalg.norm(action_vec)
	return action_vec

# add a (state, action) pair to the dataset
def add_to_dataset(dataset, state, action):
	dataset.append(state.tolist() + action.tolist())

# visualize all (state, action) pairs in one dataset
def plot_state_actions(states, filename):
	fig, ax = plt.subplots()
	for idx, state in enumerate(states):
		action = get_action(state)
		robot_label = "robot" if idx == 0 else None
		goal_label = "goal" if idx == 0 else None
		action_label = "action" if idx == 0 else None

		ax.plot(state[0], state[1], 'bo', label=robot_label)	# robot position
		ax.plot(state[2], state[3], 'ro', label=goal_label)	# goal position
		# plot the vector for the action
		ax.plot(
			[state[0], state[0]+action[0]],
			[state[1], state[1]+action[1]],
			'b-',
			label=action_label,
		)

	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(-10.5, 10.5)
	ax.set_ylim(-10.5, 10.5)
	ax.grid(True)
	ax.legend()
	fig.savefig(filename)
	plt.close(fig)

states_5 = np.array([
	[-6.601, 0.0, 6.601, 0.0],
	[6.601, 0.0, -6.601, 0.0],
	[0.0, -6.601, 0.0, 6.601],
	[0.0, 6.601, 0.0, -6.601],
	[1.247, 5.136, 0.989, 4.718],
])

states_10 = np.array([
	[-8.0, -8.0, -7.1, -8.0],
	[8.0, 8.0, 7.1, 8.0],
	[-8.0, 8.0, -8.0, 7.1],
	[8.0, -8.0, 8.0, -7.1],
	[0.0, 0.0, 0.7, 0.7],
	[0.0, 0.0, -0.7, 0.7],
	[0.0, 0.0, 0.7, -0.7],
	[0.0, 0.0, -0.7, -0.7],
	[5.0, -5.0, 5.9, -5.0],
	[-5.0, 5.0, -5.0, 4.1],
])

states_20 = np.array([
	[0.336, -8.542, -0.394, -8.869],
	[8.487, -4.579, 8.058, -4.322],
	[0.909, 1.280, 1.707, 1.229],
	[7.517, 8.824, 7.143, 9.155],
	[-6.196, -0.577, -7.880, -1.655],
	[-8.129, -6.734, -6.637, -5.402],
	[1.372, -1.066, 0.907, -0.882],
	[0.488, 6.005, 2.297, 6.858],
	[-0.520, 6.143, -0.586, 8.142],
	[-0.349, 6.972, -0.396, 4.972],
	[8.091, -7.451, 8.555, -6.800],
	[0.499, 0.439, -0.313, -0.144],
	[-3.027, -1.022, -2.036, -0.894],
	[-6.188, -5.066, -6.228, -5.865],
	[6.746, 8.178, 6.557, 8.113],
	[1.532, -2.027, -0.001, -0.742],
	[8.797, 4.234, 5.303, 7.810],
	[4.738, 5.932, 3.960, 5.303],
	[-8.702, -5.162, -8.877, -5.065],
	[-6.774, -7.149, -7.680, -8.931],
])

datasets = {
	5: states_5,
	10: states_10,
	20: states_20,
}

# build a dataset from one of the optimized state lists
def get_dataset(states=states_20, filename="dataset_20.pkl", plot_filename="state_action_20.png"):
	dataset = []

	for state in states:
		# calculate action towards closest goal
		action = get_action(state)
		# add the state-action pair to the dataset
		add_to_dataset(dataset, state, action)

	pickle.dump(dataset, open(filename, "wb"))
	plot_state_actions(states, plot_filename)
	print(filename, "has this many state-action pairs:", len(dataset))

def get_all_datasets():
	for size, states in datasets.items():
		get_dataset(
			states,
			"dataset_" + str(size) + ".pkl",
			"state_action_" + str(size) + ".png",
		)

if __name__ == "__main__":
	get_all_datasets()
