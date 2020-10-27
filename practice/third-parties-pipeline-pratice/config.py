from scipy.stats import uniform
import numpy as np

TRAIN_DATA_FN = 'train.csv'
MODEL_NAME = 'full_pipe_model'

CABIN_COLUMN = 'cabin'
NAME_COLUMN = 'name'
DROP_COLUMN = ['name','ticket', 'boat', 'body','home.dest']

CAT_COLUMN = ['sex', 'cabin', 'embarked', 'title']
NUM_COLUMN = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CONT_NUM_COLUMN = ['age', 'fare']

CAT_COLUMN_W_NA = ['cabin', 'embarked']
NUM_COLUMN_W_NA = ['age', 'fare']

PARAM_GRIDS = {'C':uniform(0.0001, 10)}
N_ITER = 50

TARGET = ['survived']