from sklearn.pipeline import Pipeline
import pipeline as pipe
import pandas as pd
import config as cf
import numpy as np
import joblib

from sklearn.model_selection import cross_val_score, train_test_split

if __name__ == '__main__':

    train = pd.read_csv(cf.TRAIN_DATA_FN)

    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop(cf.TARGET, axis=1), train[cf.TARGET[0]],
        test_size=0.2, random_state=42
        )

    pipe.prep_pipeline.fit(pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))

    # preprocessing input
    prep_train = pipe.prep_pipeline.transform(X_train) 
    
    # estimate
    np.random.seed(42)
    pipe.estimator_cv.fit(prep_train, y_train)

    # save prep pipeline and estimator
    full_pipeline = Pipeline([
        ('prep_pipeline', pipe.prep_pipeline),
        ('estimator', pipe.estimator_cv.best_estimator_)
    ])

    print(f'best estimator: {pipe.estimator_cv.best_estimator_}')
    print(f'full pipe score: {full_pipeline.score(X_train, y_train)}')

    joblib.dump(full_pipeline, cf.MODEL_NAME)
    