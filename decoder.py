import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr

class SingleLayerDecoder(nn.Module):

    def __init__(self, feature_size, batch_size, max_length, input_dimension, input_embed,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = input_dimension
        self.input_embed = input_embed    # dimension of embedding space (actor)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.device = device
        self.is_training = is_train
        self.feature_size = feature_size
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self._wl = nn.Parameter(torch.Tensor(*(self.input_embed, decoder_hidden_dim)).to(self.device))

        self.t_l = nn.Parameter(torch.Tensor(*(self.input_embed, decoder_hidden_dim)).to(self.device))

        self.y_l = nn.Parameter(torch.Tensor(*(self.input_embed, decoder_hidden_dim)).to(self.device))

        self.logit_l = nn.Parameter(torch.Tensor(*(self.feature_size+2, self.feature_size)).to(self.device))
        #self._wr = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._u = nn.Parameter(torch.Tensor(*(decoder_hidden_dim, 1)).to(self.device))
        self._l = self._l = nn.Parameter(torch.Tensor(1).to(self.device), requires_grad=True)

        #nn.Parameter(torch.cat(((10)*torch.ones(10,),(10)*torch.ones(10,)),0).to(self.device), requires_grad=True)

        self.bias = torch.nn.Parameter((-10)*torch.rand((100,1)).to(device), requires_grad=True)

        self.act_2 =  nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._wl)
        nn.init.xavier_uniform_(self._u)
        nn.init.xavier_uniform_(self.t_l)
        nn.init.xavier_uniform_(self.y_l)
        nn.init.xavier_uniform_(self.logit_l)
        #nn.init.xavier_uniform_(self._u)
        # is None:  # Randomly initialize the learnable bias
          #  bias_initial_value = torch.randn([1]).numpy()[0]
        #elif self.use_bias_constant:  # Constant bias
       #     bias_initial_value = self.bias_initial_value
        #else:  # Learnable bias with initial value
        #    bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, 0)

    def forward(self, encoder_output,t,y):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        W_l = self._wl
        U = self._u

        #print (W_l.shape,encoder_output.shape)
        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)
        #print (encoder_output.type())
        dot_t = torch.einsum('ijk, kl->ijl', t, self.t_l)
        dot_y = torch.einsum('ijk, kl->ijl', y, self.y_l)

        dot_sum = torch.cat((dot_l,dot_t,dot_y),1)

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = self.activation(dot_sum)
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(dot_sum)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = dot_sum
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        #print ("final_sum is", final_sum.shape)
        #final_sum = final_sum.squeeze(2)
        #print ("U.view(self.decoder_hidden_dim) is", U.view(self.decoder_hidden_dim,1).shape)
        #print ("final_sum is", final_sum.shape)
        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijk, k->ij', final_sum, U.squeeze(1))  # Readability

        logits = self.act_2(logits)

        logits = torch.einsum('ij, jk -> ik', logits, self.logit_l)
        #logits = logits + self.bias
        #print ("logits shaps is", logits.shape)
        self.logit_bias = self._l

        #if self.use_bias:  # Bias to control sparsity/density
        #logits 


        logits = logits  + self.logit_bias
        #print (logits.shape)
        #logits = logits #+ self.bias
        
        prob = distr.Bernoulli(logits=logits)  
    
        sampled_arr = prob.sample()  
        sampled_arr.requires_grad=True
        kk =  nn.Sigmoid()
        #print (kk(logits))
        #print (sampled_arr.shape)
        return sampled_arr, logits, prob.entropy()