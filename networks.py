import torch
import torch.nn as nn
import torch.nn.functional as F 


class NeuralNetwork1(nn.Module):
    """Neural Network with 1 Hidden Layer"""
    def __init__(self, num_of_inputs, hidden_size1, num_of_outputs):
        super (NeuralNetwork1, self).__init__()
        self.linear1 = nn.Linear(num_of_inputs, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, num_of_outputs)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x

class NeuralNetwork2(nn.Module):
    """Neural Network with 2 Hidden Layers"""
    def __init__(self, num_of_inputs, hidden_size1, hidden_size2, num_of_outputs):
        super (NeuralNetwork2, self).__init__()
        self.linear1 = nn.Linear(num_of_inputs, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, num_of_outputs)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        
        return x    




