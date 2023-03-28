from collections import deque
import numpy as np
import random
import pickle


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, n_step):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.buffer = deque(maxlen=self.max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)

        states = np.zeros((batch_size, *self.state_dim))
        actions = np.zeros((batch_size, self.action_dim))
        rewards = np.zeros(batch_size)
        next_states = np.zeros((batch_size, *self.state_dim))
        dones = np.zeros(batch_size)

        for i, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done

        return states, actions, rewards, next_states, dones

    def sample_n_step(self, batch_size):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)

        states = np.zeros((batch_size, *self.state_dim))
        actions = np.zeros((batch_size, self.action_dim))
        rewards = np.zeros((batch_size, self.n_step))
        next_states = np.zeros((batch_size, *self.state_dim))
        dones = np.zeros((batch_size, self.n_step))

        for i, sample in enumerate(samples):
            state, action, reward, next_state, done = sample
            for j in range(self.n_step):
                if j == 0:
                    states[i] = state
                    actions[i] = action
                    rewards[i, j] = reward
                    next_states[i] = next_state
                    dones[i, j] = done
                else:
                    state, action, reward, next_state, done = self.buffer[
                        (self.buffer.index(sample) + j) % len(self.buffer)
                    ]
                    rewards[i, j] = reward
                    dones[i, j] = done
        return states, actions, rewards, next_states, dones

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer = pickle.load(f)
