import joblib
import config as cf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def predict(input_data):
    model = joblib.load(cf.MODEL_NAME)
    return model.predict(input_data)

def predict_proba(input_data):
    model = joblib.load(cf.MODEL_NAME)
    return model.predict_proba(input_data)

if __name__ == '__main__':
    
    train = pd.read_csv(cf.TRAIN_DATA_FN)

    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop(cf.TARGET, axis=1), train[cf.TARGET[0]],
        test_size=0.2, random_state=42
        )
    
    # load model
    print(predict_proba(X_valid))