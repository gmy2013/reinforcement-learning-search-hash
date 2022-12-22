import numpy as np
from numpy import genfromtxt
import pandas as pd
def ShiftMatrix(data):
    n = data.shape[0]
    m = data.shape[1]
    copy = data.copy()
    copy_1 = data.copy()
    copy[:,0] = data[:,m-1]
    copy[:,1:m] = data[:,0:m-1]

    #copy[:,m-1] = data[:,0]
    #copy[:,0:m-1] = data[:,1:m]
    res = data#np.power(data,2)#copy * data#np.exp(data)#copy * data#np.power(data,3)#
    #print (np.max(res),np.min(res))
    #print (np.max(data),np.min(data))
    return res
def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def GeneratingSimulatingData(sample_size, feature_size, ratio_post_t, ratio_pre_t, ratio_post_y, ratio_pre_y):
    m = sample_size
    n = feature_size
    n_pot = int(n*ratio_post_t) ## number of post_treatment variables
    n_pet = int(n*ratio_pre_t) ## number of pre_treatment variables
    n_poy = int(n*ratio_post_y) ## number of post_outcome variables
    n_pey = int(n*ratio_pre_y) ## number of pre_outcome variables
    n_confounders = n-(n_pot+n_poy+n_pet+n_pey)
    print ("The size of pre_treatment, post_treatment, pre_outcome, post_outcome and confounders are {}, {}, {} , {} and {}".format(n_pet,n_pot,n_pey, n_poy, n_confounders))

    #x_pot = np.random.normal(0, 1, (m, n_pot))
    x_pet = np.random.normal(0, 1, (m, n_pet))
    #x_poy = np.random.normal(0, 1, (m, n_poy))
    x_pey = np.random.normal(0, 1, (m, n_pey))    
    x_con = np.random.normal(0, 1, (m, n_confounders)) 

    ### generating treatment array


    nom_h = max(((n_pet+n_pot+n_confounders) / 20.0), 1.0)
    nom_c = max(((n_pet+n_pot+n_confounders) / 20.0), 1.0)

    #print (nom_d)
    treatment_coeff_harm = np.ones((n_pet,1))
    treatment_coeff_conf = np.ones((n_confounders,1))
    treatment_coeff_harm = treatment_coeff_harm/nom_h
    treatment_coeff_conf = treatment_coeff_conf/nom_c
    treatment_coeff = np.concatenate((treatment_coeff_harm,treatment_coeff_conf),0)
    #treatment_coeff = np.random.uniform(-0.2,0.2,(n_pet+n_pot+n_confounders,1))
    #print (treatment_coeff)
    #treatment_coeff = treatment_coeff/(1.0*np.sqrt(n/5))
    inner_product = np.matmul(np.concatenate((x_pet,x_con),1),treatment_coeff)
    inner_product = 1+np.exp(-inner_product)
    inner_product = np.reciprocal(inner_product)
    #inner_product = np.clip(inner_product,0.1,0.9)
    print (np.min(inner_product),np.max(inner_product))
    #inner_product = np.
    treatment = np.random.binomial(1, inner_product)
    print ("The shape of treatment array is {}".format(treatment.shape))



    treatment_post_co = np.random.rand(m, n_pot)
    x_pot = treatment*treatment_post_co+np.random.normal(0, 1, (m, n_pot))


    corre_Y_pey =  4*np.ones((n_pey,1))#np.random.uniform(-2.5,2.5,(n_pey,1))####
    corre_Y_con = (-2) *np.ones((n_confounders,1))#np.random.uniform(-2.5,2.5,(n_confounders,1))####

    x_pey_non = ShiftMatrix(x_pey)
    x_con_non = ShiftMatrix(x_con)


    outcome = np.matmul(x_pey,corre_Y_pey) + np.matmul(x_con,corre_Y_con) + treatment + np.random.normal(0, 2, (m, 1))
    outcome_post_co = 0.05#np.random.uniform(-0.05,0.05,(m, n_poy))
    x_poy = outcome*outcome_post_co+np.random.normal(0, 1, (m, n_poy))

    
    
    print ("The shape of outcome array is {}".format(outcome.shape))

    covariate = np.concatenate((x_pot,x_pet,x_con,x_poy,x_pey),1)
    print ("The shape of covariate array is {}".format(covariate.shape))    

    print (np.max(x_pot),np.min(x_pot))
    print (np.max(x_poy),np.min(x_poy))
    return [treatment,outcome,covariate]

