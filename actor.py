import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder import TransformerEncoder
from decoder import SingleLayerDecoder
from critic import Critic


class Actor(object):
    _logger = logging.getLogger(__name__)

    def __init__(self,
                 hidden_dim,
                 max_length,
                 num_heads,
                 num_stacks,
                 residual,
                 decoder_activation,
                 decoder_hidden_dim,
                 use_bias,
                 use_bias_constant,
                 bias_initial_value,
                 batch_size,
                 input_dimension,
                 lr1_start,
                 lr1_decay_step,
                 lr1_decay_rate,
                 alpha,
                 init_baseline,
                 device,
                 feature_size,
                 is_train=True
            ):

        self.input_dimension = input_dimension
        self.hidden_dim      = hidden_dim
        self.batch_size      = batch_size
        self.max_length      = max_length
        self.num_heads       = num_heads
        self.num_stacks      = num_stacks
        self.residual        = residual
        self.alpha           = alpha
        self.init_baseline   = init_baseline
        self.lr1_start       = lr1_start
        self.lr1_decay_rate  = lr1_decay_rate
        self.lr1_decay_step  = lr1_decay_step
        self.decoder_activation = decoder_activation
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_bias          = use_bias
        self.use_bias_constant = use_bias_constant
        self.bias_initial_value = bias_initial_value
        self.is_train        = is_train
        self.device          = device
        self.feature_size = feature_size
        # Reward config
        self.avg_baseline = torch.tensor([self.init_baseline],
                                         device=self.device)  # moving baseline for Reinforce

        # Training config (actor)
        self.global_step = torch.Tensor([0])  # global step

        # Training config (critic)
        self.global_step2 = torch.Tensor([0])  # global step
        self.lr2_start = self.lr1_start  # initial learning rate
        self.lr2_decay_rate = self.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = self.lr1_decay_step  # learning rate decay step

        # encoder
        #if self.encoder_type == 'TransformerEncoder':
        self.encoder = TransformerEncoder(batch_size=self.batch_size,
                                            max_length=self.max_length,
                                            input_dimension=self.input_dimension,
                                            hidden_dim=self.hidden_dim,
                                            num_heads=self.num_heads,
                                            num_stacks=self.num_stacks,
                                            is_train=self.is_train,
                                            device=self.device)
        
        self.encoder = self.encoder.to(self.device)
        # decoder
        
        self.decoder = SingleLayerDecoder(
            feature_size = self.feature_size,
            batch_size=self.batch_size,
            max_length=self.max_length,
            input_dimension=self.input_dimension,
            input_embed=self.hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            decoder_activation=self.decoder_activation,
            use_bias=self.use_bias,
            bias_initial_value=self.bias_initial_value,
            use_bias_constant=self.use_bias_constant,
            is_train=self.is_train,
            device=self.device)
        self.decoder = self.decoder.to(self.device)

        # critic
        self.critic = Critic(batch_size=self.batch_size,
                             max_length=self.max_length,
                             input_dimension=self.input_dimension,
                             hidden_dim=self.hidden_dim,
                             init_baseline=self.init_baseline,
                             device=self.device)
        self.critic = self.critic.to(self.device)
        # Optimizer
        self.opt1 = torch.optim.Adam([
                        {'params': self.encoder.parameters()},
                        {'params': self.decoder.parameters()},
                        {'params': self.critic.parameters()}
                    ], lr=self.lr1_start, betas=(0.9, 0.99), eps=0.0000001)

        self.opt2 = torch.optim.Adam(self.critic.parameters(),lr=0.0001, betas=(0.9, 0.99), eps=0.0000001)

        self.lr1_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt1, gamma=pow(self.lr1_decay_rate, 1/self.lr1_decay_step))
        
        self.lr2_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt2, gamma=pow(self.lr1_decay_rate, 1/self.lr1_decay_step))
        self.batch_num = 0
        self.optimal_mask = None
        self.criterion = nn.MSELoss()
        self.critic = Critic(batch_size=self.batch_size,
                        max_length=self.max_length,
                        input_dimension=self.input_dimension,
                        hidden_dim=self.hidden_dim,
                        init_baseline=self.init_baseline,
                        device=self.device)

    def build_permutation(self, inputs, inputs_t, inputs_y):

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = inputs
        self.t_input = inputs_t
        self.y_input = inputs_y
        # encoder
        self.encoder_output = self.encoder(self.input_)

        # decoder
        self.mask, self.mask_scores, self.entropy = self.decoder(self.encoder_output,inputs_t,inputs_y)
        #print (self.mask.shape)
       # mask_for_reward = self.mask.mean()

        logits_for_rewards = self.mask_scores

        entropy_for_rewards = self.entropy
        

        self.test_scores = torch.sigmoid(logits_for_rewards)[:2]

        log_probss = F.binary_cross_entropy_with_logits(input=logits_for_rewards, 
                                                        target=self.mask, 
                                                        reduction='none')
        #print (log_probss.shape)
        self.log_softmax = log_probss.mean(1)#torch.mean(log_probss, axis=0)
        self.entropy_regularization = entropy_for_rewards.mean(1)#torch.mean(entropy_for_rewards, axis=0)
        self.criterion = nn.MSELoss()
        self.build_critic()

        self.critic_exp_mvg_avg = torch.zeros(1).to(self.device)
        self.beta = 0.9

    def build_critic(self):
        # Critic predicts reward (parametric baseline for REINFORCE)
        self.critic(self.encoder_output,self.t_input,self.y_input)

    def build_reward(self, reward_):

        self.reward = reward_#.squeeze(1)

        self.build_optim()

    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        # Update baseline
        reward_mean = self.reward
        self.reward_batch = reward_mean
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean
        self.avg_baseline = self.avg_baseline.to(self.device)



        # Discounted reward
        self.reward_baseline = (self.reward).detach()  # [Batch size, 1]
        #self.reward_baseline = self.reward_baseline.squeeze(1)
        # Loss



        #print (self.log_softmax.shape)
        #print (self.entropy_regularization.shape)
        #print (self.reward_baseline.shape)
        #print (self.reward.shape)
        #print (self.critic.predictions.shape) #torch.mean(self.reward_baseline * self.log_softmax, 0)
        #self.loss1 = ()

        #critic_prediction = self.critic.predictions.detach()

        #self.critic_exp_mvg_avg = self.critic_exp_mvg_avg.detach()
        self.loss_critic = self.criterion(self.critic.predictions,self.reward.float())
        #- self.critic_exp_mvg_avg #-self.avg_baseline
        self.loss_actor = ((self.reward -self.avg_baseline).detach())*self.log_softmax #self.criterion(self.reward - self.avg_baseline, self.critic.predictions)
        self.loss_actor = self.loss_actor.mean(0)+((-1)*self.entropy_regularization.mean(0)) #(-10)*self.lr1_scheduler.get_last_lr()[0]
        
        if self.batch_num <= 500:
            self.reward_constraint = 0.
        else:
            #print (self.mask_scores.shape)
            #print (self.optimal_mask.shape)
            #print((self.optimal_mask.reshape(1,-1).detach().repeat(self.batch_size,1)).shape)
            self.reward_constraint = 0.#F.binary_cross_entropy_with_logits(input=self.mask_scores, target=self.optimal_mask.reshape(1,-1).detach().repeat(self.batch_size,1))#nn.BCELoss(self.mask_scores,self.optimal_mask.detach().repeat(self.batch_size,1))

        self.loss_actor = self.loss_actor+0.1*self.reward_constraint
        # Minimize step
        self.opt1.zero_grad()
        
        self.loss_actor.backward()
        #self.loss_critic.backward()

        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) +
            list(self.critic.parameters())
            , max_norm=1., norm_type=2)

        self.opt1.step()
        
        self.lr1_scheduler.step()
        


        return 