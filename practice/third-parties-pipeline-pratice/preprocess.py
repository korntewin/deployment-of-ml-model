import enum
from re import T
from numpy.core.defchararray import title
from numpy.core.fromnumeric import var
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np
import re

class ReplaceQmarkToNan(TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        df = X.replace('?', np.nan)
        return df


class ConvertManyCabinsToOneCabin(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def get_first_cabin(self, row):

        if row is np.nan:
            return np.nan

        return row.split()[0]

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.variables[0]] = df[self.variables[0]].apply(self.get_first_cabin)
        return df


class NameToTitle(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables

    def name_to_tile(self, row):
        name = row
        if re.search('Mrs', name):
            return 'Mrs'
        elif re.search('Mr', name):
            return 'Mr'
        elif re.search('Miss', name):
            return 'Miss'
        elif re.search('Master', name):
            return 'Master'
        else:
            return 'Other'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['title'] = df[self.variables].apply(self.name_to_tile)
        return df


class DropUnecessaryColumn(TransformerMixin):

    def __init__(self, variables=None):
        if isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = [variables]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        return df.drop(columns=self.variables)


class ReduceCardinality(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['cabin'] = df['cabin'].apply(
            lambda x: np.nan if x is np.nan else ' '.join(re.findall('[a-zA-Z]+', str(x))))

        return df


class RareLabelEncoder(TransformerMixin):
    def __init__(self, variables=None, threshold=0.05):
        self.variables = variables
        self.threshold = threshold
        self.rare_labels = {}
        return None

    def fit(self, X, y=None):
        df = X.copy()
        
        for var in self.variables:
            self.rare_labels[var] = df.groupby(var)[var].count()[
                df.groupby(var)[var].count()/len(df) < self.threshold
            ].index

        return self

    def transform(self, X):
        df = X.copy()

        for var in self.variables:
            df[var] = np.where(df[var].isin(self.rare_labels[var]), 'Rare', df[var])

        return df


class FillNumVars(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, variables_w_na=None):
        self.variables = variables
        self.variables_w_na = variables_w_na
        self.imputer = SimpleImputer(strategy='median')
        
    def fit(self, X, y=None):
        df = X.copy()

        self.imputer.fit(df[self.variables])
        return self

    def transform(self, X):
        df = X.copy()

        for var in self.variables_w_na:
            df[var+'_na'] = np.where(df[var].isnull(), 1, 0)

        df[self.variables] = self.imputer.transform(df[self.variables])

        return df


class FillCatVars(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, variables_w_na=None, fill_value=None):
        self.variables = variables
        self.variables_w_na = variables_w_na
        self.fill_value = fill_value
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for var in self.variables_w_na:
            df[var+'_na'] = np.where(df[var].isnull(), 1, 0)

        df[self.variables] = df[self.variables].fillna(self.fill_value)

        return df


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        self.ordinal_dics = {}


    def fit(self, X, y=None):
        data = pd.concat([X, y], axis=1)
        data.columns = list(X.columns) + ['target']
        
        for var in self.variables:
            label = data.groupby(var)['target'].count().sort_values(ascending=True).index
            self.ordinal_dics[var] = {k:i for i, k in enumerate(label)}

        return self

    def transform(self, X):
        df = X.copy()

        for var in self.variables:
            df[var] = df[var].map(self.ordinal_dics[var])

        return df


class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.variables] = PowerTransformer().fit_transform(df[self.variables])

        return df


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = None

    def fit(self, X, y=None):
        df = X.copy()
        self.scaler.fit(df)
        self.features=df.columns
        return self

    def transform(self, X):
        df = X.copy()
        df[self.features] = self.scaler.transform(df)
        return df

    
if __name__ == '__main__':
    replace_tf = ReplaceQmarkToNan()
    firstcab_tf = ConvertManyCabinsToOneCabin('cabin')
    title_tf = NameToTitle('name')
    drop_tf = DropUnecessaryColumn(['name','ticket', 'boat', 'body','home.dest'])
    reduce_cardinality_tf = ReduceCardinality('cabin')
    rmrare_tf = RareLabelEncoder(['cabin', 'title'], 0.05)
    fill_num_tf = FillNumVars(['age', 'fare'], ['age', 'fare'])
    fill_cat_tf = FillCatVars(['cabin', 'embarked'], ['cabin', 'embarked'], fill_value="Missing")
    ord_enc = OrdinalEncoder(['cabin'])

    train = pd.read_csv('train.csv')
    train = replace_tf.fit_transform(train)
    train = firstcab_tf.fit_transform(train)
    train = title_tf.fit_transform(train)
    train = drop_tf.fit_transform(train)
    train = reduce_cardinality_tf.fit_transform(train)
    train = rmrare_tf.fit_transform(train)
    train = fill_num_tf.fit_transform(train)
    train = fill_cat_tf.fit_transform(train)
    train = ord_enc.fit_transform(train.drop(columns=['survived']), train['survived'])
    print(train['cabin'])

