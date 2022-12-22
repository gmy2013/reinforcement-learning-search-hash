import numpy as np

# Utilities only meant for the doubly-robust branch

def softmax(x):
    """ softmax function """
    
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x


def logistic_sigmoid(x):
    return(1/(1 + np.exp(-x)))

    
def aipw_calculator(method, y, a, py_a, py_n, pa1, pa0, difference=True, weights=None, splits=None, continuous=False, coffe=None):
    """Function to calculate AIPW estimates. Called by AIPTW, SingleCrossfitAIPTW, and DoubleCrossfitAIPTW
    """
    #print ("sss")
    # Point estimate calculation
    #print (pa1)
    if method == "AIPW":
        y1 = np.where(a == 1, (y - py_a*(1 - pa1)) / pa1, py_a)
        y0 = np.where(a == 0, (y - py_n*(1 - pa0)) / pa0, py_n)
        return np.nanmean(y1 - y0)
    elif method == "IPW":
        Wbeta_t = (a - pa1) / (pa1 * (1 - pa1))
        IPW_var_arr = y*Wbeta_t
        return np.mean(IPW_var_arr)
    elif method == "REG":
        return coffe
    elif method == "DIR":
        y1 = y[a == 1]
        y0 = y[a == 0]
        Di_res = y1.mean()-y0.mean()
        return Di_res
    #####################IPW estimator####################
    #y1 = np.where(a == 1, y / pa1, 0)
    #y0 = np.where(a == 0, 0, y / (1-pa1))


    #IPW_res = np.mean(IPW_var_arr)
    #res = np.
    #####################IPW estimator####################

    #####################Reg estimator####################
    #y1 = py_a#np.where(a == 1, py_a, 0)
    #y0 = py_n#np.where(a == 0, , 0)
    #####################Reg estimator####################

    #####################Direct estimator####################
    #y1 = y[a == 1]#np.where(a == 1, y, np.zeros(y.shape))
    #y0 = y[a == 0]#np.where(a == 0, y, np.zeros(y.shape))
    #Di_res = y1.mean()-y0.mean()
    #####################Direct estimator####################

    # Warning system if values are out of range
    if not continuous:
        if np.mean(y1) > 1 or np.mean(y1) < 0:
            warnings.warn("The estimated probability for all-exposed is out of the bounds (less than zero or greater "
                          "than 1). This may indicate positivity issues resulting from extreme weights, too small of a "
                          "sample size, or too flexible of models. Try setting the optional `bound` argument. If using "
                          "DoubleCrossfitAIPTW, try SingleCrossfitAIPTW or the TMLE estimators instead.", UserWarning)
        if np.mean(y0) > 1 or np.mean(y0) < 0:
            warnings.warn("The estimated probability for none-exposed is out of the bounds (less than zero or greater "
                          "than 1). This may indicate positivity issues resulting from extreme weights, too small of a "
                          "sample size, or too flexible of models. Try setting the optional `bound` argument. If using "
                          "DoubleCrossfitAIPTW, try SingleCrossfitAIPTW or the TMLE estimators instead.", UserWarning)
    # Calculating ACE as a difference
    if difference:
        if weights is None:
            #print (y1-y0)
            
            estimate = np.mean(IPW_var_arr)#Di_res#np.nanmean(y1 - y0) ###
            if splits is None:
                var = np.mean(IPW_var_arr)#Di_res#np.nanmean(y1 - y0)#np.nanvar((y1 - y0) - estimate, ddof=1) / y.shape[0]##
            else:
                var_rd = []
                for i in set(splits):
                    y1s = y1[splits == i]
                    y0s = y0[splits == i]
                    var_rd.append(np.var((y1s - y0s) - estimate, ddof=1))
                var = np.mean(var_rd) / y.shape[0]
        else:
            estimate = DescrStatsW(y1, weights=weights).mean - DescrStatsW(y0, weights=weights).mean
            var = np.nan

    # Calculating ACE as a ratio
    else:
        if weights is None:
            estimate = np.nanmean(y1) / np.nanmean(y0)
            if estimate < 0:
                warnings.warn("lower than 0", UserWarning)
            py_o = a*py_a + (1-a)*py_n
            ic = ((a*(y-py_o)) / (np.mean(py_a)*pa1) + (py_a - np.mean(py_a)) -
                  ((1-a)*(y-py_o)) / (np.mean(py_n)*pa0) + (py_n - np.mean(py_n)))
            var = np.nanvar(ic, ddof=1) / y.shape[0]
        else:
            estimate = DescrStatsW(y1, weights=weights).mean / DescrStatsW(y0, weights=weights).mean
            var = np.nan

    return estimate, var

def tmle_unit_bounds(y, mini, maxi, bound):
    # bounding for continuous outcomes
    v = (y - mini) / (maxi - mini)
    v = np.where(np.less(v, bound), bound, v)
    v = np.where(np.greater(v, 1-bound), 1-bound, v)
    return v


def tmle_unit_unbound(ystar, mini, maxi):
    # unbounding of bounded continuous outcomes
    return ystar*(maxi - mini) + mini
