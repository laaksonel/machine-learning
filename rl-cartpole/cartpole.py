import math
import gym
import numpy as np

class CartPoleQAgent():
    def __init__(self, bins=(3, 3, 6, 6), episodes=500, min_learning_rate=0.1, discount=1.0, decay=25, min_epsilon=0.1):
        self.bins = bins
        self.episodes = episodes
        self.min_learning_rate = min_learning_rate
        self.min_epsilon = min_epsilon

        # Discount determines the importance of future reward
        # When discount = 0, model considers only immediate rewards
        # discount = 1, model will consider future rewards from next state
        self.discount = discount

        # Multipliier for
        self.decay = decay

        self.env = gym.make('CartPole-v0')

        self.Q_table = np.zeros(self.bins +  (self.env.action_space.n,))

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                             math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                             -math.radians(50) / 1.]

        self.steps = np.zeros(self.episodes)

    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            # Normalize the value between lower and upper bounds (0 - 1)
            scale = ((obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]))

            # Select the bin based on the scale
            # For example 3 bins = [0, 1, 2]
            # Scale 0.5 * (3 - 1) = 1, observation belongs to bin 1
            new_obs = int(round((self.bins[i] - 1) * scale))

            # If the calculated bin index is larger than the highest bin index,
            # select the highest bin index instead (cap the value)
            new_obs = min(self.bins[i] - 1, max(0, new_obs))

            discretized.append(new_obs)
        return tuple(discretized)

    def get_action(self, state):
        # The epsilon value decreases during the training
        # After series of episodes we should start to exploit our Q-table values (previous experiences)
        # instead of taking random actions ("exploration action")
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += (self.min_learning_rate *
                                        (reward
                                         + self.discount * np.max(self.Q_table[new_state])
                                         - self.Q_table[state][action]))

    def get_epsilon(self, t):
        """Gets value for epsilon. It declines as we advance in episodes."""
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        # Adapt the learning curve during the episodes
        # New information in the later stages is valued lower than in the beginning of training
         return max(self.min_learning_rate, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        for e in range(self.episodes):
            self.learning_rate = self.get_learning_rate(e)

            current_state = self.discretize_state(self.env.reset())
            self.epsilon = self.get_epsilon(e)

            done = False
            while not done:
                # Select next action from the current state
                action = self.get_action(current_state)

                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state

    def run(self):
        self.env = gym.wrappers.Monitor(self.env,'cartpole')
        current_state = self.discretize_state(self.env.reset())
        done = False
        while not done:
            self.env.render()
            action = self.get_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(obs)
            current_state = new_state

if __name__ == "__main__":
    agent = CartPoleQAgent()
    print("Training agent...")
    agent.train()
    print("Training done, run agent...")
    agent.run()
    print("Done")
