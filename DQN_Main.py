from agent import DQN_Agent
import gym
from utils import *
import numpy as np
import torch
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1')


###### SEED  ######
seed = 10 
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
##################


########## *** PARAMETERS *** ##########
hidden_size1 = 128 # Size of first Hidden Layer
hidden_size2 = 128 # Size of second Hidden Layer
num_of_hd_layers = 2 # 1 if we want 1 Hidden Layer, 2 if we want 2 Hidden Layers
learning_rate = 0.02 # Learning rate
memory_size = 50000 # Memory size
batch_size = 32 # Batch size
gamma = 0.99 # Discounting factor
num_episodes = 200 # Number of training episodes
egreedy_decay = 500 # Parameter for probability to pick random actions. 
                     # If we have a larger number of training episodes set it at a higher value (500, 1000, 5000)
eval_episodes = 5 # Number of evaluation episodes
number_of_elements = 5000 # Number of elements in memory
file2save = "dqn_agent" # Name we want to save
directory = "saves" # Directory where we want to save
########################################

agent = DQN_Agent(env, hidden_size1, hidden_size2, num_of_hd_layers, learning_rate, memory_size, batch_size, gamma) # Create agent

####### TRAIN AGENT #######
fill_memory(env, number_of_elements, agent) # Add elements in memory
train(agent, env, num_episodes, batch_size, egreedy_decay, file2save, directory) # Train agent


######### EVALUATE AGENT #########
agent.load(file2save, directory) # Load best avg saved 
evaluate_agent(agent, env, eval_episodes, render=True) # Evaluate agent


"""If we want to train multiple agents with different parameters and compare their results,
we can store 'avg_rewards' in some variable and plot them"""
# agent1 = DQN_Agent(env, 120, 64, 1, 0.01, 25000, 32, 0.99)
# agent2 = DQN_Agent(env, 150, 150, 2, 0.02, 25000, 32, 0.99)

# agent1_average = train(agent1, env, 25, 32, 500, 'agent1', 'saves')
# agent2_average = train(agent2, env, 25, 32, 500, 'agent2', 'saves')

# plt.plot(agent1_average, label = "agent1")
# plt.plot(agent2_average, label = "agent2")
# plt.title("Compare Agents")
# plt.legend(loc="upper left")

