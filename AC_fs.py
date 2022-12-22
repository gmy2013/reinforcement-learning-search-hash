import os
import logging
from tqdm import tqdm
import platform
import torch
import numpy as np
from torch.utils.data import DataLoader
import time


from actor import Actor

from GLM_machine_learner import GLM
from familes import Gaussian, Bernoulli, Poisson, Exponential
from AIPW_eff import AIPW_eff


from BaseLearner import BaseLearner

from data_sim import GeneratingSimulatingData

import multiprocessing as mp
from multiprocessing import Pool, Manager,Process

mp.set_start_method('forkserver', force=True)

def propessing(batch_input, device):
    if ('ndarray' in str(type(batch_input))):
        batch_input = torch.from_numpy(batch_input)
    batch_input = batch_input.float().to(device)
    return batch_input

feature_size = 20

import gc








def GetRewardShallow(queue,data_list,n_jobs,index,num,mask):
    #print (mask.shape)
    
    start = time.time()
    t,y,x = data_list[0],data_list[1],data_list[2]
    resss = 0
    # = None
    index_list = []
    reward_list = []
    for index_task in range(index*num, index*num+num):
        mask_per_index = mask[index_task,:]
        sum_mask = np.sum(mask_per_index)
        reward_curr = None
        if (sum_mask==0.):
            #print ("Overlall")
            reward_curr = 20.0
            #queue.put([index_task,reward_curr])
            index_list.append(int(index_task))
            reward_list.append(reward_curr)
        else:
            nonzero_entries = np.nonzero(mask_per_index)
            covariate = x[:,nonzero_entries]
            covariate = np.squeeze(covariate,1)
            if (len(covariate.shape)==1):
                covariate = covariate[:,np.newaxis]
            data_list_ours = [t,y,covariate]
            aipw = AIPW_eff("AIPW", data_list_ours)
            aipw.exposure_model(GLM(family=Bernoulli()))
            aipw.outcome_model(GLM(family=Gaussian()))
            ate = aipw.fit()
            reward_curr = aipw.OverallMaskObjective(nonzero_entries)
            reward_curr = reward_curr*0.01
            index_list.append(int(index_task))
            reward_list.append(reward_curr)
            #reward.append(reward_curr)
            data_list_ours = []
            #print ("Overlall")
            #gc.collect()


    queue.put([index_list,reward_list])
    #print ("Process Done!")
    end = time.time()
    reward_list = []
    index_list = []
    return #resss


class RL(BaseLearner):

    

    def __init__(self,
                 hidden_dim=512, 
                 num_heads=16, 
                 num_stacks=2, 
                 residual=False,  
                 decoder_activation='tanh', 
                 decoder_hidden_dim=512, 
                 use_bias=False, 
                 use_bias_constant=False, 
                 bias_initial_value=False, 
                 batch_size=64, 
                 input_dimension=512, 
                 normalize=False, 
                 transpose=False, 
                 score_type='BIC', 
                 reg_type='LR', 
                 lambda_iter_num=1000, 
                 lambda_flag_default=True, 
                 score_bd_tight=False, 
                 lambda1_update=1.0, 
                 lambda2_update=10, 
                 score_lower=0.0, 
                 score_upper=0.0, 
                 lambda2_lower=-1.0, 
                 lambda2_upper=-1.0, 
                 seed=8, 
                 nb_epoch=30000,
                 lr1_start=0.001,
                 lr1_decay_step=5000, 
                 lr1_decay_rate=0.96, 
                 alpha=0.99, 
                 init_baseline=-1.0, 
                 temperature=3.0, 
                 C=10.0, 
                 l1_graph_reg=0.0, 
                 inference_mode=True, 
                 verbose=False, 
                 device_type='gpu', 
                 device_ids=4):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.residual = residual
        self.decoder_activation = decoder_activation
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_bias = use_bias
        self.use_bias_constant = use_bias_constant
        self.bias_initial_value = bias_initial_value
        self.batch_size = batch_size
        self.input_dimension = input_dimension
        self.normalize = normalize
        self.transpose = transpose
        self.score_type = score_type
        self.reg_type = reg_type
        self.lambda_iter_num = lambda_iter_num
        self.lambda_flag_default = lambda_flag_default
        self.score_bd_tight = score_bd_tight
        self.lambda1_update = lambda1_update
        self.lambda2_update = lambda2_update
        self.score_lower = score_lower
        self.score_upper = score_upper
        self.lambda2_lower = lambda2_lower
        self.lambda2_upper = lambda2_upper
        self.seed = seed
        self.nb_epoch = nb_epoch
        self.lr1_start = lr1_start
        self.lr1_decay_step = lr1_decay_step
        self.lr1_decay_rate = lr1_decay_rate
        self.alpha = alpha
        self.init_baseline = init_baseline
        self.temperature = temperature
        self.C = C
        self.l1_graph_reg = l1_graph_reg
        self.inference_mode = inference_mode
        self.verbose = verbose
        self.device_type = device_type
        self.device_ids = device_ids
        self.words = []
        with open('method_name_5w.txt', 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                self.words.append(line.strip())

       
        os.environ['CUDA_VISIBLE_DEVICES'] = "4"
        device = torch.device('cuda')


        self.GT_mask = np.concatenate((np.zeros((int(feature_size/2),)),np.ones((int(feature_size/2),))),0)
        #self.GT_mask = np.tile(self.GT_mask.reshape(1,-1), (self.batch_size,1))
        #print (self.GT_mask.shape)
        self.GT_mask_ = np.concatenate((np.ones((int(feature_size/2),)),np.zeros((int(feature_size/2),))),0)
        self.device = device

    def learn(self, columns=None, dag=None, **kwargs):

        global feature_size
        # Where to generate the data
        data_list = GeneratingSimulatingData(sample_size= 20, feature_size=feature_size, ratio_post_t=0.1, ratio_pre_t=0.1, ratio_post_y=0.2, ratio_pre_y=0.3)
        print('data list ########################')
        print(data_list)
        self.T = data_list[0]
        self.inputdata = data_list[2]
        self.Y = data_list[1]
        self.data_list = data_list
        

        total_datasize = self.inputdata.shape[0]
        seq = np.random.randint(total_datasize, size=(total_datasize))
        #print('Seq ###############################')
        #print(seq)
        self.T = self.T[seq]
        self.inputdata = self.inputdata[seq,:]
        self.Y = self.Y[seq]
        training_size = int(total_datasize*0.8)
        testing_size = int(total_datasize*0.2)
        self.data_list = [self.T[:training_size],self.Y[:training_size],self.inputdata[:training_size,:]] ## training
        self.test_data_list = [self.T[training_size:],self.Y[training_size:],self.inputdata[training_size:,:]] ## training
        #print ("hehehehehehehe",self.test_data_list[2].shape)



        

        self.datasize = self.inputdata.shape[0]
        self.max_length = self.inputdata.shape[1]

        causal_matrix = self._rl(self.inputdata)
        self.causal_matrix = causal_matrix







    def GetRewardShallowInit(self,mask):
        #print(mask.shape)
        nonzero_entries = np.nonzero(mask)
        #nonzero_entries = np.squeeze(nonzero_entries)
        #print (nonzero_entries.shape)
        covariate = self.inputdata[:,nonzero_entries]
        #print (covariate.shape)
        covariate = np.squeeze(covariate,1)
        data_list_ours = [self.T,self.Y,covariate]
        aipw = AIPW_eff("AIPW", data_list_ours)
        aipw.exposure_model(GLM(family=Bernoulli()))
        aipw.outcome_model(GLM(family=Gaussian()))
        ate = aipw.fit()
        reward_curr = aipw.OverallMaskObjective(nonzero_entries)
        #reward.append(reward_curr)
        #d#el aipw,
        data_list_ours = []
        return reward_curr




    # calculate the reward based on the current mask
    def SequentialReward(self,mask):
        batch_size = mask.shape[0]
        '''
        #np_mask = mask.
        reward_mask = np.where(mask==self.GT_mask,2*np.ones(mask.shape),(-1)*np.zeros(mask.shape))
        reward_mask = np.sum(reward_mask,1)
        '''
        reward = []

        import time
        st = time.time
        for index in range(batch_size):  # per batch
            cur_mask = mask[index, :]
            hash_vocab = {}
            positions = []
            #print(cur_mask)
            for i in range(len(cur_mask)): # current mask
                if cur_mask[i] == 1:
                    positions.append(i)
            #print(positions)
            for word in self.words:
                hashed_string = ""
                for i in positions:
                    if i >= len(word):
                        continue
                    hashed_string += word[i]
                hashed_string += str(len(word))
                if hash_vocab.get(hashed_string) == None:
                    hash_vocab[hashed_string] = 0
                else:
                    hash_vocab[hashed_string] += 1
            #print(hash_vocab)
            collide = 0
            for key in hash_vocab.keys():
                collide += (hash_vocab[key])
            #print(reward)
            reward.append(collide + 100*len(positions))

        '''
        for index in range(batch_size):
            maskk = mask[index,:]
            sum_mask = np.sum(maskk)
            nonzero_entries = np.nonzero(maskk)[0]
            reward_curr = None
            if (sum_mask==0.):
                #print ("Overlall")
                reward_curr = 20.0
            else:
                covariate = self.inputdata[:,nonzero_entries]
                #covariate = np.squeeze(covariate,1)
                if (len(covariate.shape)==1):
                    covariate = covariate[:,np.newaxis]
                data_list_ours = [self.T,self.Y,covariate]
                aipw = AIPW_eff("AIPW", data_list_ours)
                aipw.exposure_model(GLM(family=Bernoulli()))
                aipw.outcome_model(GLM(family=Gaussian()))
                ate = aipw.fit()
                #reward_curr = aipw.OverallMaskObjective(nonzero_entries)*0.01
                #print (self.test_data_list[0].shape)
                #print ()
                reward_curr = 0.01*aipw.OverallEvaMaskObjective(mask,[self.test_data_list[0],self.test_data_list[1],self.test_data_list[2][:,nonzero_entries]])
                #reward.append(reward_curr)
                data_list_ours = []
                #print ("Overlall")
            reward.append(reward_curr)
            print(reward)
        '''
        return -np.array(reward) #-reward_mask 

    #


    


    
    def _rl(self, X):
        # Reproducibility
        #set_seed(self.seed)

        print('Python version is {}'.format(platform.python_version()))

        # input data


        # set penalty weights
        score_type = self.score_type
        reg_type = self.reg_type
        global feature_size


        # actor
        actor = Actor(hidden_dim=self.hidden_dim,
                      max_length=self.max_length,
                      num_heads=self.num_heads,
                      num_stacks=self.num_stacks,
                      residual=self.residual,
                      decoder_activation=self.decoder_activation,
                      decoder_hidden_dim=self.decoder_hidden_dim,
                      use_bias=self.use_bias,
                      use_bias_constant=self.use_bias_constant,
                      bias_initial_value=self.bias_initial_value,
                      batch_size=self.batch_size,
                      input_dimension=self.input_dimension,
                      lr1_start=self.lr1_start,
                      lr1_decay_step=self.lr1_decay_step,
                      lr1_decay_rate=self.lr1_decay_rate,
                      alpha=self.alpha,
                      init_baseline=self.init_baseline,
                      device=self.device,
                      feature_size = feature_size)
        self.actor = actor
        #actor = actor.to(device)
        print('Finished creating training dataset and reward class')

        # Initialize useful variables
        rewards_avg_baseline = []
        rewards_batches = []
        reward_max_per_batch = []

        probsss = []
        self.optimal_mask = None

        print('Starting training.')
        
        #print (feature_size/2)
        mask_initital = np.concatenate((np.zeros((int(feature_size/2),)),np.ones((int(feature_size/2),))),0)
        #mask_initital = mask_initital.unsqueeze(0).repeat(128,1)
        start = time.time()
        #print ("The initial optimal reward is", np.mean(self.GetRewardShallowInit(mask_initital)))
        end = time.time()
        #print('Initial Task runs %0.2f seconds.' % (end - start))
        mask_all = np.concatenate((np.ones((int(feature_size/2),)),np.ones((int(feature_size/2),))),0)
        #mask_initital = mask_initital.unsqueeze(0).repeat(128,1)
        print ("The all reward is", np.mean(self.GetRewardShallowInit(mask_all)))
        max_reward_all = -float('inf')
        max_mask = None
        cnt = 0
        for i in tqdm(range(1, self.nb_epoch + 1)):
            cnt = cnt + 1
            if self.verbose:
                print('Start training for {}-th epoch'.format(i))

            actor.batch_num = cnt
            input_batch = []
            input_t_batch = []
            input_y_batch = []
            #start = time.time()
            for _ in range(self.batch_size):
                seq = np.random.randint(self.datasize, size=(self.input_dimension))
                input_ = self.inputdata[seq].T
                input_batch.append(input_)
                input_t_ = self.T[seq].T
                input_y_ = self.Y[seq].T
                input_t_batch.append(input_t_)
                input_y_batch.append(input_y_)
            #end = time.time()
            #print ("The data load time is {}".format(start-end))
            inputs = torch.from_numpy(np.array(input_batch)).to(self.device)
            input_t_batch = torch.from_numpy(np.array(input_t_batch)).to(self.device).float()
            input_y_batch = torch.from_numpy(np.array(input_y_batch)).to(self.device).float()
            # Test tensor shape
            if i == 1:
                print('Shape of actor.input: {}, {} and {}'.format(inputs.shape,input_t_batch.shape,input_y_batch.shape))

            # actor perform a permutation and select the task
            actor.build_permutation(inputs, input_t_batch, input_y_batch)
            #graphs_feed = actor.graphs_

            #reward_feed = callreward.cal_rewards(graphs_feed.cpu().detach().numpy(), lambda1, lambda2)  # np.array
            # get the mask from the actor
            np_mask = actor.mask.detach().cpu().numpy()
            #reward_feed = self.ParallelShallowReward(np_mask)
            # get the reward feedback
            reward_feed = self.SequentialReward(np_mask)
            #print (reward_feed)
            actor.build_reward(reward_ = torch.from_numpy(np.array(reward_feed)).to(self.device))
            reward_ = np.max(reward_feed)
            index_min = np.argmax(reward_feed, axis=0)
            # max reward, max reward per batch
            #max_reward_batch = float('inf')


            
            if reward_ > max_reward_all:
                max_reward_all = reward_
                
                max_mask = actor.mask[index_min,:]
                actor.optimal_mask = max_mask

                    
            #max_reward_batch = -max_reward_batch


            


            if self.verbose:
                print('Finish calculating reward for current batch of graph')

            score_test, probs, reward_batch, reward_avg_baseline = \
                    actor.test_scores, actor.log_softmax, actor.reward_batch, actor.avg_baseline

            if self.verbose:
                print('Finish updating actor and critic network using reward calculated')
            

            #rewards_avg_baseline.append(reward_avg_baseline)


            probsss.append(probs)

            # logging
            reward_for_pr = reward_#.detach().cpu().numpy()[0][0]
            max_reward_all_pr = max_reward_all#.detach().cpu().numpy()[0][0]
            #print (reward_for_pr.shape)0
            #print ("Reward constraint is", cnt,actor.batch_num,actor.reward_constraint)
            if i == 1 or i % 2 == 0:
                #loss1_pr = actor.loss1.detach().cpu().numpy()
                loss_actor = actor.loss_actor.detach().cpu().numpy()
                loss_critic = actor.loss_critic.detach().cpu().numpy()
                print('[iter {}] reward_batch: {},  max_reward_batch: {}'.format(i,
                            reward_for_pr, max_reward_all_pr))
                print ("Losses are, ", loss_actor, loss_critic)
                print ("Reward constraint is", actor.reward_constraint)
                print ("Optimalest mask is ")
                print(max_mask)

        logging.info('Training COMPLETED !')

        self.max_mask = max_mask.detach().cpu().numpy()

        return self.max_mask

    def metric_mask(self):
        global feature_size
        res_mask = self.max_mask
        cnt_vec = np.where(self.GT_mask==res_mask,np.ones(res_mask.shape),np.zeros(res_mask.shape))
        #print (np.sum(cnt_vec))
        #print (cnt_vec.shape)
        vec_acc = np.sum(cnt_vec)/(feature_size*1.0)
        return vec_acc


    

    def metric_ate(self):
        mask = self.max_mask

        nonzero_entries = np.nonzero(mask)
        covariate = self.inputdata[:,nonzero_entries]
        covariate = np.squeeze(covariate,1)
        data_list_ours = [self.T,self.Y,covariate]
        aipw = AIPW_eff("AIPW", data_list_ours)
        aipw.exposure_model(GLM(family=Bernoulli()))
        aipw.outcome_model(GLM(family=Gaussian()))
        ate = aipw.fit()
        del aipw,data_list_ours
        return ate

if __name__=='__main__':
    total_mask_ratio = 0.0
    total_ate = 0.0
    epoches = 1
    for index in range(epoches):
        model = RL()
        model.learn()
        #print(n.causal_matrix)
        mask_tmp = model.metric_mask()
        ate_tmp  = model.metric_ate()
        total_mask_ratio = total_mask_ratio+ mask_tmp
        total_ate = total_ate + ate_tmp
        del model

    print ("final result is", total_mask_ratio/epoches, total_ate/epoches)