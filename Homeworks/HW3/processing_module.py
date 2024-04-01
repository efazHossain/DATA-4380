#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        
    Returns:
        DataFrame: DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        strategy (str): Imputation strategy. Options: 'mean', 'median', 'most_frequent'.
        
    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    imputer = SimpleImputer(strategy=strategy)
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_filled

def standardize_features(df):
    """
    Standardize features in a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        
    Returns:
        DataFrame: DataFrame with standardized features.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# You can add more functions for data preprocessing and cleaning as needed

def remove_columns(df, columns):
    """
    Remove specified columns from a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        columns (list): List of column names to be removed.
        
    Returns:
        DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=columns)

def encode_categorical(df, columns):
    """
    Encode categorical columns in a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        columns (list): List of column names to be encoded.
        
    Returns:
        DataFrame: DataFrame with categorical columns encoded.
    """
    label_encoders = {}
    for col in columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df

def scale_min_max(df):
    """
    Scale features to a specified range (0 to 1) using Min-Max scaling.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        
    Returns:
        DataFrame: DataFrame with features scaled to the range 0 to 1.
    """
    return (df - df.min()) / (df.max() - df.min())

def log_transform(df, columns):
    """
    Apply log transformation to specified columns in a DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        columns (list): List of column names to be log-transformed.
        
    Returns:
        DataFrame: DataFrame with specified columns log-transformed.
    """
    for col in columns:
        df[col] = np.log1p(df[col])
    return df

