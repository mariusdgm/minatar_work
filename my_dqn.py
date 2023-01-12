import torch
import random
import numpy as np
import copy 
from collections import deque, Counter

import torch.autograd as autograd
from torch import optim

import seaborn as sns
import matplotlib.pyplot as plt


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class AgentDQN:
    def __init__(self, train_model, target_model, optimizer) -> None:

        self.epsilon_by_frame = self._get_decay_function(start=1.0, end=0.01, decay=100)
        self.n_experiments = 0
        self.n_frames = 0

        self.exploration = True
        self.gamma = 0.99  # discount rate
        self.replay_buffer = ReplayBuffer(100000)

        self.train_model = train_model
        self.target_model = target_model

        self.optimizer = optimizer

        self.prev_state = None

    def _get_exp_decay_function(self, start, end, decay):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(self, start, end, decay_in):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay_in)
            decay_in (int): how many steps to reach the end value
        """
        return lambda x:lambda x: max(end, min(start, start - (start - end) * (x / decay_in)))

    def get_state(self, game):
        # # the state will be a matrix of the game state
        # # 1 for snake head
        # # 2 for snake body
        # # 3 for food

        # state = np.zeros(
        #     (game.w // game.block_size, game.h // game.block_size), dtype=np.float32
        # )

        # head = game.snake[0]
        # state[
        #     int(head.x // game.block_size) - 1, int(head.y // game.block_size) - 1
        # ] = 1
        # for point in game.snake[1:]:
        #     state[
        #         int(point.x // game.block_size) - 1, int(point.y // game.block_size) - 1
        #     ] = 2
        # state[
        #     int(game.food.x // game.block_size) - 1,
        #     int(game.food.y // game.block_size) - 1,
        # ] = 4

        # state /= 4  # normalize

        # # return as [1, 1, 32, 32] toch tensor
        # # state = torch.from_numpy(state)
        # # state = torch.reshape(state, (1, 1, state.shape[0], state.shape[1]))

        # # build sequence using previous state and update prev
        # state_sequence = np.asarray([self.prev_state, state])
        # self.prev_state = state

        # # reshape for lininar network
        # if state_sequence[0] is not None:
        #     state_sequence = np.stack(state_sequence)
        #     state_sequence = state_sequence.reshape(32*32*2)

        # return state_sequence

        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        state = np.asarray(state)
        state = state.astype(np.float32)

        return state

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward) 
        done = torch.FloatTensor(done)

        q_values = self.train_model(state)
        q_value = q_values.gather(1, action).squeeze(1) 

        next_q_values = self.target_model(next_state).detach()
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - done) 

        expected_q_value_data = expected_q_value
        expected_q_value_data = torch.unsqueeze(expected_q_value_data, 1) 

        loss = (q_value - expected_q_value_data).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, reward, done, q_value, expected_q_value

    def get_action(self, state):
        eps = self.epsilon_by_frame(self.n_experiments)

        # get either a random move for exploration or an expected move from the model
        if random.random() > eps and state[0] is not None:
            # print("Shape of tensor before squeeze: ", state.shape)
            # print(state)

            state = torch.from_numpy(state)
            state = torch.unsqueeze(state, 0)

            # print("Shape of tensor: ", state.shape)
           
            q_value = self.train_model.forward(state)
            action = q_value.max(1)[1].data[0]
            action = action.item()
        else:
            action = random.randrange(self.train_model.num_actions)
            # action = random.randrange(env.action_space.n)

        # transform single value to vector
        action_vec = [0, 0, 0]
        action_vec[action] = 1

        return action_vec, action