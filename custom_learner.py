import numpy as np

def exposure_learner(xdata, ydata, ml_model, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of being
    treated (i.e. Pr(A=1 | L))
    """
    # Trying to fit the Machine Learning model
    #print (xdata.shape,ydata.shape)
    ydata = np.squeeze(ydata,1)
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
        #fm = sm.GLM(ydata, xdata, family=sm.families.family.Binomial()).fit()
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summary'):  # SuPyLearner has a nice summarize function
        print('==============================================================================')
        print('Propensity Score Model')
        fm.summary()
        print('==============================================================================')

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(xdata)
        # this allows support for pygam LogisticGAM, which only returns only 1 probability
        if g.ndim == 1:
            return g
        else:
            return g[:, 1]
    elif hasattr(fm, 'predict'):
        return fm.predict(xdata),ml_model,fm
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def outcome_learner(xdata, ydata, all_a, none_a, ml_model, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of outcome
    (i.e. Pr(Y=1 | A=1, L) and Pr(Y=1 | A=0, L)).
    """
    ydata = np.squeeze(ydata,1)

    # Trying to fit Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summary'):  # Nice summarize option from SuPyLearner
        print('==============================================================================')
        print('Outcome Model')
        fm.summary()
        print('==============================================================================')

    # Generating predictions
    #if continuous:
    if hasattr(fm, 'predict'):
        #print (all_a.shape)
        #print (none_a.shape)
        qa1 = fm.predict(all_a)
        qa0 = fm.predict(none_a)
        return qa1, qa0, ml_model, fm
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
    '''
    else:
        if hasattr(fm, 'predict_proba'):
            qa1 = fm.predict_proba(all_a)
            qa0 = fm.predict_proba(none_a)
            if (qa1.ndim == 1) and (qa0.ndim == 1):
                return qa1, qa0
            else:
                return qa1[:, 1], qa0[:, 1]
        elif hasattr(fm, 'predict'):
            qa1 = fm.predict(all_a)
            qa0 = fm.predict(none_a)
            return qa1, qa0
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
    '''