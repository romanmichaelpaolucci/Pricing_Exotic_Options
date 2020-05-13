import math
import random
import numpy as np
from OptionContract import *
from sklearn.model_selection import KFold
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense

def neural_black_scholes():
    # Generate Random Options
    a = OptionTools().generate_random_option(10000)
    train = a[:int(len(a)*.7)]
    test = a[int(len(a)*.7):]
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for option in train:
        X_train.append([option.asset_price, option.asset_volatility, option.strike_price, option.time_to_expiration, option.risk_free_rate])
        y_train.append([option.price, option.delta])
    for option in test:
        X_test.append([option.asset_price, option.asset_volatility, option.strike_price, option.time_to_expiration, option.risk_free_rate])
        y_test.append([option.price, option.delta])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential()
    model.add(Dense(32, input_shape=(5,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='relu'))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    t = OptionTools().generate_random_option(1)
    z1=[]
    z2=[]
    for option in t:
        z1.append([option.asset_price, option.asset_volatility, option.strike_price, option.time_to_expiration, option.risk_free_rate])
        z2.append([option.price, option.delta])
    z1 = np.array(z1)
    z2 = np.array(z2)
    print('actual: ', z2)
    print('predicted: ', model.predict(z1))

neural_black_scholes()
