#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datetime import datetime
import numpy as np
import numpy as numpy
import pandas as pd
import pylab
import calendar
from scipy import stats
import seaborn as sns
from sklearn import model_selection
from scipy.stats import kendalltau
import warnings
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def label_encode(df1,df2):
    for c in df2.columns: # Converts all categorical values to numerical values via Label Encoding
        df2[c]=df2[c].fillna(-1)
        if df2[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(df2[c].values))
            df2[c] = lbl.transform(list(df2[c].values))
    train_df = df1.merge(df2, how='left', on='parcelid') #Merged the old dataset with the converted properties set
    return train_df

#### Preprocessing
def preprocess(train1,test1):
    # StandardScaler
    sc = StandardScaler()
    train1 = sc.fit_transform(train1)
    test1 = sc.transform(test1)

