import preprocess as pp
import config as cf
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, Parallel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform
from sklearn.preprocessing import PowerTransformer, StandardScaler

prep_pipeline = Pipeline([
    ('replace_w_nan', pp.ReplaceQmarkToNan()),
    ('conv_to_one_cabin', pp.ConvertManyCabinsToOneCabin(cf.CABIN_COLUMN)),
    ('get_title', pp.NameToTitle(cf.NAME_COLUMN)),
    ('drop_features', pp.DropUnecessaryColumn(cf.DROP_COLUMN)),
    ('reduce_cab_cardinality', pp.ReduceCardinality(cf.CABIN_COLUMN)),
    ('fill_cat_vars', pp.FillCatVars(cf.CAT_COLUMN, cf.CAT_COLUMN_W_NA, 'Missing')),
    ('remove_rare_label', pp.RareLabelEncoder(cf.CAT_COLUMN)),
    ('fill_num_vars', pp.FillNumVars(cf.NUM_COLUMN, cf.NUM_COLUMN_W_NA)),
    ('log_transformer', pp.LogTransformer(cf.CONT_NUM_COLUMN)),
    ('ordinal_encoder', pp.OrdinalEncoder(cf.CAT_COLUMN)),
    ('scaler', pp.Scaler())
])

estimator_cv = RandomizedSearchCV(LogisticRegression(), cf.PARAM_GRIDS, n_iter=cf.N_ITER)


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop(cf.TARGET, axis=1), train[cf.TARGET],
        test_size=0.2, random_state=42
        )

    prep_pipeline.fit(pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))
    X_train_prep = prep_pipeline.transform(X_train)
