import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import torch


class MinMaxScaler:

    def __init__(self, feature_range=(0, 1), interval=(263.15, 393.15)):
        """Scales inputs to a feature range based on an interval. Can be used for any feature if an
        unscaled dataset is used.
        It is important that the interval is a global interval and not determined by the training data 
        to ensure comparability between different plants/turbines needed for the computation of the 
        dimensionless physics model of the PCRNN. 
        In case of using the scaled paper dataset with unscaled Gearbox bearing temperatures:
        The interval has to be (263.15, 393.15), which are approximate values for temperatures in 
        Kelvin that are physical meaningful boundaries for the ambient temperature as well as the 
        bearing temperature. 
        """
        self.feature_range = feature_range
        self.minmax = interval
        
    def transform(self, x):
        x_scaled = self._minmax_scaling(x)
        return x_scaled
    
    def inverse_transform(self, x_scaled, relative_value=False):
        range_min, range_max = self.feature_range
        min_val, max_val = self.minmax
        if relative_value:
            x = ((x_scaled - range_min) / (range_max - range_min)) * (max_val - min_val)
        else:
            x = ((x_scaled - range_min) / (range_max - range_min)) * (max_val - min_val) + min_val
        return x

    def _minmax_scaling(self, x):
        range_min, range_max = self.feature_range
        min_val, max_val = self.minmax
        x_std = (x - min_val) / (max_val - min_val)
        x_scaled = x_std * (range_max - range_min) + range_min
        return x_scaled
