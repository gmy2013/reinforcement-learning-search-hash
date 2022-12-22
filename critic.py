import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 init_baseline, device=None):
        super().__init__()

        self.batch_size      = batch_size
        self.max_length      = max_length
        self.input_dimension = input_dimension
        # Network config
        self.input_embed     = hidden_dim
        self.num_neurons     = hidden_dim
        self.device     = device

        # Baseline setup
        self.init_baseline = 0

        layer0 = nn.Linear(in_features=self.input_dimension,
                           out_features=self.num_neurons).to(self.device)
        torch.nn.init.xavier_uniform_(layer0.weight)
        self.h0_layer = nn.Sequential(layer0, nn.ReLU()).to(self.device)

        self.layer1 = nn.Linear(in_features=self.num_neurons,
                                out_features=1).to(self.device)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer1.bias.data = torch.Tensor([self.init_baseline]).to(self.device)

    def forward(self, encoder_output, t, y):
        #print ("encoder output is", encoder_output.shape)
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        critic_input = torch.cat((encoder_output.detach(),t,y),1)
        frame = torch.mean(critic_input, dim=1)
 
        # ffn 1
        h0 = self.h0_layer(frame)
        # ffn 2
        h1 = self.layer1(h0)
        self.predictions = torch.squeeze(h1)
        #self.predictions = torch.mean(self.predictions,0,keepdim=True)
        #print ("critic output is", self.predictions.shape)