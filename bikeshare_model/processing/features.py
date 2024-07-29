from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError('variable must be a string.')

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        # self.wkday_null_index = X[X[self.variable].isnull() == True].index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        self.wkday_null_index = X[X[self.variable].isnull() == True].index
        # X['dteday'] = pd.to_datetime(X['dteday'])
        X.loc[self.wkday_null_index, self.variable] = X.loc[self.wkday_null_index, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        X = X.drop(columns= ['dteday'], axis= 1)

        return X
    
class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError('variable must be a string.')
        
        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.fill_value = X[self.variable].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X[self.variable] = X[self.variable].fillna(self.fill_value)

        return X
    
class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable: str, mappings: dict):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError('variable must be a string.')
        if not isinstance(mappings, dict):
            raise ValueError('mappings must be a dictionary.')

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings).astype(int)

        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable: str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError('variable must be a string.')

        self.variable = variable
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        self.q1 = X[self.variable].quantile(0.25)
        self.q3 = X[self.variable].quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.upper_bound = self.q3 + 1.5 * self.iqr
        self.lower_bound = self.q1 - 1.5 * self.iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variable] = np.where(X[self.variable] > self.upper_bound, self.upper_bound, X[self.variable])
        X[self.variable] = np.where(X[self.variable] < self.lower_bound, self.lower_bound, X[self.variable])
        # X[self.variable] = X[self.variable].clip(lower=self.lower_bound, upper=self.upper_bound)
        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """
    def __init__(self, variable:str):
        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")
        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output= False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out([self.variable])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded_weekdays = self.encoder.transform(X[[self.variable]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_weekdays
        # drop 'weekday' column after encoding
        X.drop(self.variable, axis=1, inplace=True)  
        return X
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    '''Drop unused columns from the dataframe'''

    def __init__(self, cols: list):
        # YOUR CODE HERE
        if not isinstance(cols, list):
            raise ValueError('cols must be a list.')

        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X = X.drop(self.cols, axis=1)

        return X