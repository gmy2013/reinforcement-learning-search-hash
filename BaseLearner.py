import numpy as np
import pandas as pd
from collections.abc import Iterable
from pandas import Index, RangeIndex


class BaseLearner(object):

    def __init__(self):

        self._causal_matrix = None

    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value