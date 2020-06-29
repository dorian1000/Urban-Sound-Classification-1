# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:59:08 2020

@author: windows10-pc
"""


import pandas as pd
import numpy as np

t = pd.read_csv('../data/train_short.csv')

classes = t.Class.values

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

lab = LabelEncoder()
encodedClasses = lab.fit_transform(np.unique(classes))
labelClasses = np.unique(classes)
codes = dict(zip(encodedClasses, labelClasses))
transformed = lab.fit_transform(classes)
transformed = np_utils.to_categorical(transformed)
