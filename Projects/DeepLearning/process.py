#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# In[2]:


def encode_categorical(X):
    encoder = OneHotEncoder(drop="first", sparse=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
    X_encoded.columns = encoder.get_feature_names(categorical_cols)
    X.drop(columns=categorical_cols, inplace=True)
    X = pd.concat([X, X_encoded], axis=1)
    return X


# In[3]:


def handle_missing_values(data):
    # Replace missing values with median for numerical columns
    imputer = SimpleImputer(strategy="median")
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    
    # Replace missing values with most frequent value for categorical columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    
    return data


# In[4]:


def scale_features(X):
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X


# In[5]:


def preprocess_data(data):
    # Handling missing values
    data = handle_missing_values(data)
    
    # Splitting features and target variable
    X = data.drop(columns=["target_column"])
    y = data["target_column"]
    
    # Scaling numerical features
    X = scale_features(X)
    
    # Handling categorical variables
    X = encode_categorical(X)
    
    return X, y

