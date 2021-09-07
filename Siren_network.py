import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from siren_pytorch import SirenNet


class DQNNet_siren(nn.Module):
    """   
    Class of the SIREN neural network - This is to define the network architecture for the main Q network and the target Network
    
    """
    def __init__(self, input_size, output_size, w0, lr=1e-3):
        
        """     
        input_size: the size of the input to the neural network -> corresponds to the state of the relay position (coordinates of the respective grid cell).
        output_size: the size of the output which is going to be 1 since the Q network is a function approximator of the Q function  (state-action value function)
        w0: parameter for the initialization of the first dense layer of the SIREN 
        lr: learning rate
        
        
        """
        super(DQNNet_siren, self).__init__()
        self.siren = SirenNet(dim_in = input_size, dim_hidden = 300, dim_out = output_size, num_layers = 3, final_activation = nn.Identity(),w0_initial = w0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.siren(x)
        return x