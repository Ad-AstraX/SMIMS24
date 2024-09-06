from typing import final

import torch
import random
import numpy as np
from collections import deque

from NeuralNetwork import NeuralNetwork
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.05


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = []  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        #self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.nn=NeuralNetwork(11,6,3,LR)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory+=[[state, action, reward, next_state, done]]  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        print(len(self.memory))
        for i in range(len(self.memory)-1,-1,-1):
            #if self.memory[i][4]:#game over
                reward=self.memory[i][2]
                self.nn.train(self.memory[i][0],reward,None)
                for j in range(i-1,-1,-1):
                    print(j)
                    if reward!=0:
                        self.nn.train(self.memory[j][0], reward, np.argmax(self.memory[j][1]))

        '''for j in range(i-1,-1,-1):
                    #print("a",self.memory[j][1])
                    self.nn.train(self.memory[j][0],reward,np.argmax(self.memory[j][1]))
                continue'''
            #else:
                #break


        """if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)"""

    def train_short_memory(self, state, reward, state_new):
        try:
            self.nn.train_short(state,reward,state_new)
            a=0
        finally:
            a=0


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        #print("state",state)
        self.epsilon = 40 - self.n_games
        final_move = [0, 0, 0]

        k=random.randint(0, 200)

        if k < self.epsilon:
            move=np.zeros(3,dtype = float)
            r=random.randint(0, 3)
            if r==3:
                r=0
            move[r] =0.99
            #print("random",r)
            #final_move[move] = 1
        else:
            #Hier
            out=self.nn.query(np.array(state))
            #state0 = torch.tensor(state, dtype=torch.float)
            #prediction = self.model(state0)
            #move = torch.argmax(prediction).item()
            move=np.argmax(out).item()
            final_move[move] = 1
            move=out

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        #print("move",final_move,np.argmax(final_move))

        # perform move and get new state
        reward, done, score = game.play_step(np.argmax(final_move))
        #print(reward,done,score)
        state_new = agent.get_state(game)
        #print("stateN",state_new)

        # train short memory
        agent.train_short_memory(state_old, reward,state_new)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            #print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()