from time import sleep
import numpy as np
import random
import gym


class Taxi():
	def __init__(self):
		super(Taxi, self).__init__()
		# import the taxi environment
		self.env = gym.make("Taxi-v3").env
		# reset environment to a random state
		self.env.reset() 
		# animation frames
		self.frames = []
		# q table itialization
		self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

	def state_space(self):
		print("Action Space {}".format(self.env.action_space))
		print("State Space {}".format(self.env.observation_space))

	def print_frames(self):
		for i, frame in enumerate(self.frames):
			print(frame.get("frame"))
			print(f"Timestep: {i + 1}")
			print(f"State: {frame['state']}")
			print(f"Action: {frame['action']}")
			print(f"Reward: {frame['reward']}")
			sleep(0.2)

	def random_solving(self):
		# set environment initial state
		self.env.s = 328  

		epochs = 0
		penalties, reward = 0, 0

		self.frames = []

		done = False

		while not done:
			action = self.env.action_space.sample()
			state, reward, done, info = self.env.step(action)

			if reward == -10:
				penalties += 1
			
			# Put each rendered frame into dict for animation
			self.frames.append({
				'frame': self.env.render(mode='ansi'),
				'state': state,
				'action': action,
				'reward': reward
				}
			)

			epochs += 1
			
		print("Timesteps taken: {}".format(epochs))
		print("Penalties incurred: {}".format(penalties))

	def train(self):
		alpha = 0.1
		gamma = 0.6
		epsilon = 0.1

		for i in range(1, 100001):
			state = self.env.reset()

			epochs, penalties, reward, = 0, 0, 0
			done = False
			
			while not done:
				if random.uniform(0, 1) < epsilon:
					action = self.env.action_space.sample() # Explore action space
				else:
					action = np.argmax(self.q_table[state]) # Exploit learned values

				next_state, reward, done, info = self.env.step(action) 
				
				old_value = self.q_table[state, action]
				next_max = np.max(self.q_table[next_state])
				
				new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
				self.q_table[state, action] = new_value

				if reward == -10:
					penalties += 1

				state = next_state
				epochs += 1
				
			if i % 100 == 0:
				print(f"Episode: {i}")

		print("Training finished.\n")

	def test(self):
		total_epochs, total_penalties = 0, 0
		episodes = 100

		for _ in range(episodes):
			state = self.env.reset()
			epochs, penalties, reward = 0, 0, 0
			
			done = False
			self.frames = []
			
			while not done:
				action = np.argmax(self.q_table[state])
				state, reward, done, info = self.env.step(action)

				if reward == -10:
					penalties += 1

				epochs += 1

				# Put each rendered frame into dict for animation
				self.frames.append({
					'frame': self.env.render(mode='ansi'),
					'state': state,
					'action': action,
					'reward': reward
					}
				)

			total_penalties += penalties
			total_epochs += epochs

		print(f"Results after {episodes} episodes:")
		print(f"Average timesteps per episode: {total_epochs / episodes}")
		print(f"Average penalties per episode: {total_penalties / episodes}")