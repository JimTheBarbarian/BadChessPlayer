import torch 
import torch.nn as nn 


class MLP(nn.Module):
    def __init__(self, observation_space, hidden_dim, action_space):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(observation_space, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_space)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

    