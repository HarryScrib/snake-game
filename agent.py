import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
MAX_MEMORY = 100_000 # store 100,000 items in this memory
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (value smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY) # auto removes from left - popleft()
        self.model = Linear_QNet(11, 256, 3) # input, hidden, output
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Load existing model if it exists
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded existing model with learned knowledge")
            print(f"Game counter restarting from 1 (cosmetic only)")
            print(f"Neural network retains all previous training")
        else:
            print("Starting fresh training - no existing model found")
    
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
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
        ]
        
        # convert list to numpy array and set data type to int (also converts bools to 0 or 1)
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        # transpose list of tuples: [(s1,a1,r1,ns1,d1), (s2,a2,r2,ns2,d2)] 
        # -> (states, actions, rewards, next_states, game_overs)
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # epsilon-greedy strategy: balance exploration vs exploitation during training
        # as games increase, epsilon decreases (less random exploration, more learned behaviour)
        self.epsilon = max(5, 200 - self.n_games * 0.1) # more games we have, smaller epsilon becomes
        # much slower decay, maintains 2.5% exploration
        
        # old schedule:

        # game 1: 79/200 = 39.5% exploration
        # game 40: 40/200 = 20% exploration
        # game 80+: 0% exploration (stops learning new strategies)

        # new schedule:

        # game 1: 199.9/200 = 99.95% exploration
        # game 1000: 100/200 = 50% exploration
        # game 1950: 5/200 = 2.5% exploration
        # game 2000+: 5/200 = 2.5% exploration (maintains some exploration forever)
      
        # initialise action array: [straight, right, left] - only one will be set to 1
        final_move = [0,0,0]
     
        # exploration: take random action if random number < epsilon threshold
        if random.randint(0, 200) < self.epsilon:
            # choose random direction (0=straight, 1=right, 2=left)
            move = random.randint(0, 2)
            # set the chosen action to 1 (one-hot encoding)
            final_move[move] = 1
        else:
            # exploitation: use trained model to predict best action
            # convert state to tensor format expected by the neural network
            state0 = torch.tensor(state, dtype=torch.float)
            # get Q-values for all possible actions from the model
            prediction = self.model(state0)
            # find the action with highest Q-value (best predicted reward)
            move = torch.argmax(prediction).item()
            # set the chosen action to 1 in one-hot encoding
            final_move[move] = 1
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent() # create an Agent object
    game = SnakeGameAI() # create a SnakeGameAI object
    
    try:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, game_over, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

            # remember
            agent.remember(state_old, final_move, reward, state_new, game_over)

            if game_over:
                # train long memory (experience replay), and plot result
                game.reset()
                agent.n_games += 1
                
                # only train long memory every 5th game
                if agent.n_games % 5 == 0:
                    agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
    except KeyboardInterrupt:
        # Ctrl + c
        print(f"\nTraining stopped after {agent.n_games} games")
        plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()