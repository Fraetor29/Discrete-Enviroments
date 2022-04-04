import torch
import torch.optim as optim
import torch.nn as nn
from networks import *
from utils import Memory
import random


class DQN_Agent():
    def __init__(self, env, hidden_size1, hidden_size2, num_of_hd_layers, learning_rate, \
                 memory_size, batch_size, gamma):
        
        self.num_of_inputs = env.observation_space.shape[0] # Select number of inputs from the enviroment
        self.num_of_outputs = env.action_space.n # Select number of posible actions from the enviroment
        self.gamma = gamma
        
        if num_of_hd_layers == 1:
            self.dqn_network = NeuralNetwork1(self.num_of_inputs, hidden_size1, self.num_of_outputs) # Create  "dqn_network" object
        elif num_of_hd_layers == 2:
            self.dqn_network = NeuralNetwork2(self.num_of_inputs, hidden_size1, hidden_size2, self.num_of_outputs) # Create  "dqn_network" object
        else:
            print ("This configuration supports only 1 or 2 Hidden Layers. Please select num_of_hd_layers = [1 or 2] ")
        
        self.loss_function = nn.MSELoss() # A built in PyTorch function that measures the mean squared error
                                          #between each element in the input x and target y
                                          
        self.optimizer = optim.Adam(self.dqn_network.parameters(), learning_rate)
        
        self.memory = Memory(memory_size)
        
        
    def select_action(self, state, epsilon, env):
        random_egreedy = torch.rand(1)[0] # torch.rand(1) = tensor([0.15])  // torch.rand(1)[0] = tensor(0.15)
                                          
        if random_egreedy > epsilon: # This comparation represents probability to take a random aciton 
            with torch.no_grad(): # Context-manager that disabled gradient calculation
                state = torch.Tensor(state) # Make "state" a tensor, because Neurala Network works with tensor
                action = self.dqn_network(state) # Probabilies to select certain "actions" -> the outputs of Neural Networks
                action = torch.max(action,0)[1] # Select the action with the highest probability; torch.max will return values[0]
                                                # and indices[1] of that values. Since we work with discrete actions, we need the
                                                # the indicies of values with the highest probability, thus we add [1] at the end
                action = action.item()  # Returns the value of this tensor as a standard Python number                           
        else:
            action = env.action_space.sample() # Select random action from the enviroment
                
        return action
        
    def update(self, batch_size):
        
        if (len(self.memory) < batch_size): # We will wait until our memory buffer has at least "batch_size" position
            return
        
        state, action, new_state, reward, done = self.memory.sample(batch_size) # Import samples from memory
        # If the network learned only from consecutive samples of experience as they occurred 
        # sequentially in the environment, the samples would be highly correlated and would 
        # therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.

        # Convert samples to Tensors
        action = torch.LongTensor(action) 
        state = torch.FloatTensor(state)
        new_state = torch.FloatTensor(new_state)
        reward = torch.FloatTensor(reward)
        done = torch.Tensor(done)   
        
        
        new_state_value = self.dqn_network(new_state).detach()
        max_new_state_value = torch.max(new_state_value, 1)[0]
        Q_val = reward + (1-done) * self.gamma * max_new_state_value
        
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])
        
        predicted_value = self.dqn_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_function(predicted_value, Q_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, filename, directory):
        torch.save(self.dqn_network.state_dict(), '%s/%s.pth' % (directory, filename))
        
    def load(self, filename, directory):
        self.dqn_network.load_state_dict(torch.load('%s/%s.pth' % (directory, filename)))
        