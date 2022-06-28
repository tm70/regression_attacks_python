import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from regattackTensorflow import *

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, mean_squared_error

if __name__ == "__main__":
    with tf.Session() as sess:
        #tf.disable_eager_execution()
        
        '''
        # generate generic regression data
        n_samples, n_features = 1000, 30
        rng = np.random.RandomState(0)
        X, y = make_regression(n_samples, n_features, noise=0.5, random_state=rng)
        '''
        
        # prepare red wine quality dataset
        data = pd.read_csv("wineQualityReds.csv")
        X = data.drop(columns=["Unnamed: 0", "quality"]).to_numpy()
        y = data["quality"].to_numpy()
        n_samples, n_features = X.shape
        
        '''
        # prepare US medical insurance dataset
        data = pd.read_csv("insurance.csv")
        X = data.drop(columns=["charges", "sex", "smoker", "region"])
        one_hot = pd.get_dummies(data[["sex", "smoker"]]).drop(columns=["sex_male", "smoker_no"])
        X = pd.concat([X, one_hot], axis=1).to_numpy()
        y = data["charges"].to_numpy()
        n_samples, n_features = X.shape
        '''
        
        # split into train/test sets
        rng = np.random.RandomState(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=rng)
        
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))
        
        # train model
        model = Sequential([Dense(20, activation="relu"),
                            Dense(1)])
        model.compile(loss=rmse, optimizer=SGD(), metrics=[rmse])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70)
        
        # test attack on a test data point
        inputs = [X_test[0]]
        targets = [0]
        #attack = regCW(sess, model, n_features, constant=0.01)
        #adv = attack.attack(inputs, targets)
        
        adv = attackFGSM(model, MeanSquaredError(),
            tf.convert_to_tensor(X_test[0].reshape(1,-1)),
            tf.convert_to_tensor(y_test[0]),
            epsilon=0.1)
        
        print(y_test[0])
        print(model(X_test[0].reshape(1,n_features)).eval())
        print(model(adv).eval())
        #print(X_test[0])
        #print(adv)
        print(mean_squared_error(X_test[0], adv).eval())
        