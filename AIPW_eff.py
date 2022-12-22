import copy
import warnings
#import patsy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
import math


from utils import aipw_calculator,softmax,logistic_sigmoid
from custom_learner import (exposure_learner,outcome_learner)

class AIPW_eff:
    r"""Augmented inverse probability of treatment weight estimator. This implementation calculates AIPTW for a
    time-fixed exposure and a single time-point outcome. `AIPTW` supports correcting for informative censoring (missing
    outcome data) through inverse probability of censoring/missingness weights.

    AIPTW is a doubly robust estimator, with a desirable property. Both of the the g-formula and IPTW require that
    our parametric regression models are correctly specified. Instead, AIPTW allows us to have two 'chances' at getting
    the model correct. If either our outcome-model or treatment-model is correctly specified, then our estimate
    will be unbiased. This property does not hold for the variance (i.e. the variance will not be doubly robust)

    The augment-inverse probability weight estimator is calculated from the following formula

    .. math::

        \widehat{DR}(a) = \frac{YA}{\widehat{\Pr}(A=a|L)} - \frac{\hat{Y}^a*(A-\widehat{\Pr}(A=a|L)}{
        \widehat{\Pr}(A=a|L)}

    The risk difference and risk ratio are calculated using the following formulas, respectively

    .. math::

        \widehat{RD} = \widehat{DR}(a=1) - \widehat{DR}(a=0)

    .. math::

        \widehat{RR} = \frac{\widehat{DR}(a=1)}{\widehat{DR}(a=0)}

    Confidence intervals for the risk difference come from the influence curve. Confidence intervals for the risk ratio
    are less straight-forward. To get confidence intervals for the risk ratio, a bootstrap procedure should be used.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame object containing all variables of interest
    exposure : str
        Column name of the exposure variable. Currently only binary is supported
    outcome : str
        Column name of the outcome variable. Currently only binary is supported
    weights : str, optional
        Column name of weights. Weights allow for items like sampling weights to be used to estimate effects
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05, returning the 95% CL

    Examples
    --------
    Set up the environment and the data set

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.doublyrobust import AIPTW
    >>> df = load_sample_data(timevary=False).drop(columns=['cd4_wk45'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    Estimate the base AIPTW model

    >>> aipw = AIPTW(df, exposure='art', outcome='dead')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    Estimate AIPTW accounting for missing outcome data

    >>> aipw = AIPTW(df, exposure='art', outcome='dead')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    AIPTW for continuous outcomes

    >>> df = load_sample_data(timevary=False).drop(columns=['dead'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    >>> aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    >>> aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
    >>> ymodel = 'art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model(ymodel)
    >>> aipw.outcome_model(ymodel, continuous_distribution='poisson')
    >>> aipw.fit()
    >>> aipw.summary()

    References
    ----------
    Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust
    estimation of causal effects. American Journal of Epidemiology, 173(7), 761-767.

    Lunceford JK, Davidian M. (2004). Stratification and weighting via the propensity score in estimation of causal
    treatment effects: a comparative study. Statistics in medicine, 23(19), 2937-2960.
    """

    def __init__(self, method ,data_list, weights=None, alpha=0.05):
        self.exposure = data_list[0]
        self.outcome = data_list[1]
        #self.outcome = np.squeeze(self.outcome,1)
        self.covariate = data_list[2]
        #self.covariate = self.covariate[:,50:]
        self.masked_covariate = None
        self._missing_indicator = '__missing_indicator__'
        self.method = method
        '''
        self.df, self._miss_flag, self._continuous_outcome = check_input_data(data=df,
                                                                              exposure=exposure,
                                                                              outcome=outcome,
                                                                              estimator="AIPTW",
                                                                              drop_censoring=False,
                                                                              drop_missing=True,
                                                                              binary_exposure_only=True)
        '''
        self._weight_ = weights
        self.alpha = alpha

        self.risk_difference = None
        self.risk_ratio = None
        self.risk_difference_ci = None
        self.risk_ratio_ci = None
        self.risk_difference_se = None
        self.risk_ratio_se = None

        self.average_treatment_effect = None
        self.average_treatment_effect_ci = None
        self.average_treatment_effect_se = None

        self._continuous_type = None
        self._fit_exposure_ = False
        self._exp_model_custom = False
        self._fit_outcome_ = False
        self._out_model_custom = False
        self._fit_missing_ = False
        self._miss_model_custom = False
        self._exp_model = None
        self._out_model = None
        self._predicted_y_ = None
        self.sample_z = np.random.normal(0,1,(100,1))


    def positivity(self, decimal=3):
        """Use this to assess whether positivity is a valid assumption for the exposure model / calculated IPTW. If
        there are extreme outliers, this may indicate problems with the calculated weights. To reduce extreme weights,
        the `bound` argument can be specified in `exposure_model()`

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        -------
        None
            Prints the positivity results to the console but does not return any objects
        """
        pos_exposure = self.exposure.copy()
        pos = np.where(pos_exposure == 1, 1 / self.prosensity_score_A1, 1 / self.prosensity_score_A0)
        print('======================================================================')
        print('                      Weight Positivity Diagnostics')
        print('======================================================================')
        print('If the mean of the weights is far from either the min or max, this may\n '
              'indicate the model is incorrect or positivity is violated')
        print('Average weight should be 2')
        print('----------------------------------------------------------------------')
        print('Mean weight:           ', round(np.mean(pos), decimal))
        #print('Standard Deviation:    ', round(pos[1], decimal))
        #print('Minimum weight:        ', round(pos[2], decimal))
        #print('Maximum weight:        ', round(pos[3], decimal))
        print('======================================================================\n')
        return np.mean(pos)

    def exposure_model(self, custom_model, bound=False, print_results=True):
        r"""Specify the propensity score / inverse probability weight model. Model used to predict the exposure via a
        logistic regression model. This model estimates

        .. math::

            \widehat{\Pr}(A=1|L) = logit^{-1}(\widehat{\beta_0} + \widehat{\beta} L)

        Parameters
        ---------- 
        model : str
            Independent variables to predict the exposure. For example, 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. SciKit-Learn style models supported as custom models. In the
            background, AIPTW will fit the custom model and generate the predicted probablities
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        #self.__mweight = model
        #self._exp_model = self.exposure + ' ~ ' + model

        #if custom_model is None:
        #    d, n, iptw = iptw_calculator(df=self.df, treatment=self.exposure, model_denom=model, model_numer='1',
        ##                                 weight=self._weight_, stabilized=False, standardize='population',
        #                                 bound=None, print_results=print_results)
        #else:
        self._exp_model_custom = True
        #data = self.covariate
        #data = patsy.dmatrix(model + ' - 1', self.df)
        intercept = np.ones(self.exposure.shape)
        d, score_model, fm = exposure_learner(xdata=np.concatenate((intercept,self.covariate),1),
                                        ydata=self.exposure,
                                        ml_model=copy.deepcopy(custom_model),
                                        print_results=False)

        g1w = d
        g0w = 1 - d

        #print(g1w.shape)
        # Applying bounds AFTER extracting g1 and g0
        if bound:
            g1w = probability_bounds(g1w, bounds=bound)
            g0w = probability_bounds(g0w, bounds=bound)
        self.score_fm = fm
        self.prosensity_score_A1 = g1w.reshape(-1,1)
        self.prosensity_score_A0 = g0w.reshape(-1,1)
        self.score_model = score_model
        self._predicted_A_ = self.prosensity_score_A1 * self.exposure + self.prosensity_score_A0 * (1 - self.exposure)
        self._predicted_A_ = self._predicted_A_.reshape(-1,1)
        #print (np.max(self._predicted_A_),np.min(self._predicted_A_))
        #print (self.exposure.shape)
        #print (g1w.shape)
        #print (self._predicted_A_.shape)
        self._fit_exposure_ = True


    def ProxGradient(self,Lambda,step,gradient):
        truncated_value = np.ones(gradient.shape)*Lambda*step
        zero = np.zeros(gradient.shape)
        result = np.maximum(gradient-truncated_value,zero)- np.maximum(-gradient-truncated_value,zero)
        #print ("asdadassadad",np.max(result))
        return result
    
    def _predict_Y(self,mask):
        #mask_new = mask.T
        intercept = np.ones(self.outcome.shape)
        covariate = self.covariate
        fm = self.outcome_fm
        Ya0 = fm.predict(np.concatenate((intercept,np.zeros(self.exposure.shape),covariate),1))
        Ya1 = fm.predict(np.concatenate((intercept,np.ones(self.exposure.shape),covariate),1))
        Ya0 = Ya0.reshape(-1,1)
        Ya1 = Ya1.reshape(-1,1)
        res = self.exposure*Ya1+(1-self.exposure)*Ya0
        #print ("WWWWWWWWWWWWW",np.max(res))

        return res

    def _predict_Y_gt(self,mask):
        intercept = np.ones(self.outcome.shape)
        nonzero_entries = np.nonzero(np.squeeze(mask))[0]
        covariate = self.covariate[:,nonzero_entries]
        fm = self.outcome_model.fit(X=np.concatenate((intercept,self.exposure,covariate),1),y=np.squeeze(self.outcome))
        Ya0 = fm.predict(np.concatenate((intercept,np.zeros(self.exposure.shape),covariate),1))
        Ya1 = fm.predict(np.concatenate((intercept,np.ones(self.exposure.shape),covariate),1))
        Ya0 = Ya0.reshape(-1,1)
        Ya1 = Ya1.reshape(-1,1)
        res = self.exposure*Ya1+(1-self.exposure)*Ya0
        return res

    def _predict_A(self,mask):
        #
        intercept = np.ones(self.outcome.shape)
        covariate = self.covariate
        A1 = 1+np.exp(-np.dot(np.concatenate((intercept,covariate),1), self.score_model.coef_))
        A1 = A1.reshape(-1,1)
        A0 = 1+1/(np.exp(-np.dot(np.concatenate((intercept,covariate),1), self.score_model.coef_)))
        A0 = A0.reshape(-1,1)
        res = self.exposure*A1+(1-self.exposure)*A0
        return res
    
    def _predict_A_gt(self,mask):
        intercept = np.ones(self.outcome.shape)
        nonzero_entries = np.nonzero(np.squeeze(mask))[0]
        covariate = self.covariate[:,nonzero_entries]
        fm = self.score_model.fit(X=np.concatenate((intercept,covariate),1), y=np.squeeze(self.exposure))
        A1 = 1/fm.predict(np.concatenate((intercept,covariate),1))#1+np.exp(-np.dot(np.concatenate((intercept,covariate),1), self.score_model.coef_))
        A1 = A1.reshape(-1,1)
        A0 = 1/(1-fm.predict(np.concatenate((intercept,covariate),1)))
        A0 = A0.reshape(-1,1)
        res = self.exposure*A1+(1-self.exposure)*A0
        #print ("WWWWWWWWWWWWW",np.max(res))
        return res

    def OverallMaskObjective(self,mask):
        #print (np.max(mask))
        residual = self.outcome-self._predict_Y(mask)
        inverse_term = self._predict_A(mask)
        #print (np.max(np.abs(inverse_term)))
        #print (np.max(inverse_term))
        overall_obj = np.mean(np.power(residual*inverse_term,2))
        return overall_obj

    def OverallEvaMaskObjective(self,mask,test_data_list):
         
        t,y,x= test_data_list
        #print (t.shape,x.shape)
        intercept = np.ones(t.shape)
        #print (t.shape,x.shape,y.shape)
        predicted_y = self.outcome_fm.predict(np.concatenate((intercept,t,x),1))
        predicted_score = self.score_fm.predict(np.concatenate((intercept,x),1))
        predicted_y = predicted_y.reshape(-1,1)
        predicted_score = predicted_score.reshape(-1,1)

        residual = y-predicted_y#self.outcome-self._predict_Y(mask)
        inverse_term = predicted_score*t+(1-predicted_score)*(1-t)
        #print (np.max(np.abs(inverse_term)))
        #print (np.max(inverse_term))
        overall_obj = np.mean(np.power(residual/inverse_term,2))
        return overall_obj

    def OverallEvaMaskATE(self,test_data_list):
        t,y,x= test_data_list
        intercept = np.ones(t.shape)

        predicted_y_1 = self.outcome_fm.predict(np.concatenate((intercept,np.ones(t.shape),x),1))
        predicted_y_0 = self.outcome_fm.predict(np.concatenate((intercept,np.zeros(t.shape),x),1))
        predicted_y_1 = predicted_y_1.reshape(-1,1)
        predicted_y_0 = predicted_y_0.reshape(-1,1)

        predicted_score = self.score_fm.predict(np.concatenate((intercept,x),1))
        predicted_score = predicted_score.reshape(-1,1)
        diff_est = aipw_calculator(method=self.method,y=y, a=t,
                                        py_a=predicted_y_1, py_n=predicted_y_0,
                                        pa1=predicted_score, pa0=1-predicted_score,
                                        difference=True, weights=None,
                                        splits=None, continuous=True)
        return diff_est

    def GTObjective(self,mask):
        residual = self.outcome-self._predict_Y_gt(mask)
        inverse_term = self._predict_A_gt(mask)
        overall_obj = np.mean(np.power(residual*inverse_term,2))
        return overall_obj




    def outcome_model(self, custom_model, continuous_distribution='gaussian', print_results=True):
        self._out_model_custom = True
        intercept = np.ones(self.exposure.shape)
        data = np.concatenate((intercept,self.exposure,self.covariate),1)
        adata = np.concatenate((intercept,np.ones(self.exposure.shape),self.covariate),1) 
        ndata = np.concatenate((intercept,np.zeros(self.exposure.shape),self.covariate),1) 


        qa1w, qa0w, fm_model, fm = outcome_learner(xdata=data,
                                    ydata=self.outcome,
                                    all_a=adata, none_a=ndata,
                                    ml_model=copy.deepcopy(custom_model),
                                    #continuous=self.3,
                                    print_results=False)
        #print (qa1w.shape)
        self.outcome_fm = fm
        self.outcome_model_A1 = qa1w.reshape(-1,1)
        self.outcome_model_A0 = qa0w.reshape(-1,1)
        self.outcome_model = fm_model
        self._predicted_y_ = self.outcome_model_A1 * self.exposure + self.outcome_model_A0 * (1 - self.exposure)
        self._predicted_y_ = self._predicted_y_.reshape(-1,1)
        #print (self._predicted_y_.shape)
        self._fit_outcome_ = True

    

    def UpdateOutcomeModel(self, mask):
        #
        #print (mask_prob)
        #nonzero_entries = np.nonzero(np.squeeze(mask_prob))[0]
        #covariate = self.covariate[:,nonzero_entries]
        #print (covariate.shape)
        covariate = self.covariate#*mask_new
        intercept = np.ones(self.exposure.shape)
        data = np.concatenate((intercept,self.exposure,covariate),1)
        adata = np.concatenate((intercept,np.ones(self.exposure.shape),covariate),1) 
        ndata = np.concatenate((intercept,np.zeros(self.exposure.shape),covariate),1) 
        qa1w, qa0w, fm_model, _ = outcome_learner(xdata=data,
                                    ydata=self.outcome,
                                    all_a=adata, none_a=ndata,
                                    ml_model=self.outcome_model,
                                    print_results=False)
        #print (qa1w.shape)
        self.outcome_model_A1 = qa1w.reshape(-1,1)
        self.outcome_model_A0 = qa0w.reshape(-1,1)
        self.outcome_model = fm_model
        self._predicted_y_ = self.outcome_model_A1 * self.exposure + self.outcome_model_A0 * (1 - self.exposure)
        self._predicted_y_ = self._predicted_y_.reshape(-1,1)
        #print (self._predicted_y_.shape)
        self._fit_outcome_ = True
    

    def FittingModelFirst(self,mask):
        step_out = 1 # 2 for independent case
        step_score = 1 # 2 for independent case
        penalty_reg = 1.0
        for index in range(200):
            #print (self.OverallMaskObjective(mask,1.0))
            self.outcome_model_coffe = self.OptimizingOutcome(mask,step_out,penalty_reg,False)
            step_out = step_out * 0.9
            self.exposure_model_coffe = self.OptimizingScore(mask,step_score,penalty_reg,False)
            #step_score = step_score * 0.9

        print ("woody loss is ", np.mean(np.power(self.outcome-np.concatenate((np.ones(self.outcome.shape),self.exposure,self.covariate),1)@self.outcome_model_coffe,2)))

        #print (self.OverallMaskObjective(mask,penalty_reg,True))
        #print ("Pretrain Done!")
        return

    def OptimizingOutcome(self,mask,step,penalty_reg,flag):
        self.outcome_model_coffe  = self.outcome_model_coffe - step*self.GradientForRegression(mask,penalty_reg,flag)
        return self.outcome_model_coffe


    def OptimizingScore(self,mask,step,penalty_reg,flag):
        self.exposure_model_coffe  = self.exposure_model_coffe - step*self.GradientForScore(mask,penalty_reg,flag)
        return self.exposure_model_coffe

    def UpdateExposureModel(self, mask):
        #mask_prob = np.random.binomial(1,mask)
        #nonzero_entries = np.nonzero(np.squeeze(mask_prob))[0]
        #covariate = self.covariate[:,nonzero_entries]
        #print (covariate.shape)
        #nonzero_entries = np.nonzero(np.squeeze(mask))[0]
        #covariate = self.covariate[:,nonzero_entries]
        #mask_new = mask.T
        covariate = self.covariate#*mask_new
        intercept = np.ones(self.exposure.shape)
        d, score_model, _ = exposure_learner(xdata=np.concatenate((intercept,covariate),1),
                                        ydata=self.exposure,
                                        ml_model=self.score_model,
                                        print_results=False)
        g1w = d
        g0w = 1 - d


        self.prosensity_score_A1 = g1w.reshape(-1,1)
        self.prosensity_score_A0 = g0w.reshape(-1,1)
        self.score_model = score_model
        self._predicted_A_ = self.prosensity_score_A1 * self.exposure + self.prosensity_score_A0 * (1 - self.exposure)
        self._predicted_A_ = self._predicted_A_.reshape(-1,1)
        #print (self._predicted_A_.shape)
        self._fit_exposure_ = True

    def GenerateMask(self,weight,bias):
        mask_res = logistic_sigmoid(np.matmul(weight,self.sample_z)+bias)
        mask_res = mask_res.reshape(-1,1)
        return mask_res

    def fit(self):
        if (self._fit_exposure_ is False) or (self._fit_outcome_ is False):
            raise ValueError('The exposure and outcome models must be specified before the doubly robust estimate can '
                             'be generated')
        #self.positivity()
        total_step = 20
        Lambda = 50.0

        

        
        a_obs = self.exposure
        y_obs = self.outcome
        py_a1 = self.outcome_model_A1
        py_a0 = self.outcome_model_A0

        if self._fit_missing_:
            ps_g1 = np.asarray(self.df['_g1_'] * self.df['_ipmw_a1_'])
            ps_g0 = np.asarray(self.df['_g0_'] * self.df['_ipmw_a0_'])
        else:
            ps_g1 = self.prosensity_score_A1
            ps_g0 = self.prosensity_score_A0




        #if self._weight_ is None:
        #    w = None
       # else:
        #    w = self.df[self._weight_]
        
        

        '''

        self.exposure_model_coffe = np.zeros((self.covariate.shape[1]+1,1))
        self.outcome_model_coffe = np.zeros((self.covariate.shape[1]+2,1))
        self.mask = np.ones((100,1))
        self.FittingModelFirst(self.mask)
        ones_vec = np.ones(self.exposure.shape)
        covariate_final = self.covariate
        self.outcome_model_A1 = np.concatenate((ones_vec,np.ones(self.exposure.shape),covariate_final),1)@self.outcome_model_coffe
        self.outcome_model_A0 = np.concatenate((ones_vec,np.zeros(self.exposure.shape),covariate_final),1)@self.outcome_model_coffe

        a_obs = self.exposure
        y_obs = self.outcome
        py_a1 = self.outcome_model_A1
        py_a0 = self.outcome_model_A0


        self.prosensity_score_A1 = logistic_sigmoid(np.concatenate((ones_vec,covariate_final),1)@self.exposure_model_coffe)
        self.prosensity_score_A0 = 1-self.prosensity_score_A1


        if self._fit_missing_:
            ps_g1 = np.asarray(self.df['_g1_'] * self.df['_ipmw_a1_'])
            ps_g0 = np.asarray(self.df['_g0_'] * self.df['_ipmw_a0_'])
        else:
            ps_g1 = self.prosensity_score_A1
            ps_g0 = self.prosensity_score_A0
        '''
       
        
        diff_est = aipw_calculator(method=self.method,y=y_obs, a=a_obs,
                                             py_a=py_a1, py_n=py_a0,
                                             pa1=ps_g1, pa0=ps_g0,
                                             difference=True, weights=None,
                                             splits=None, continuous=True)



        # Generating estimates for the risk difference and risk ratio
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)


        self.average_treatment_effect = diff_est
        #self.average_treatment_effect_se = np.sqrt(diff_var)
        #self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * np.sqrt(diff_var),
        #                                    self.average_treatment_effect + zalpha * np.sqrt(diff_var)]
        #print ("The result of the {} estimator is: bias {} and var {}".format(self.method, self.average_treatment_effect,self.average_treatment_effect_se))
        return self.average_treatment_effect#,self.average_treatment_effect_se



