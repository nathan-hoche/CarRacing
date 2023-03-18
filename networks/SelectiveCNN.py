from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import random


LEARNING_RATE = 0.1
ACTION_SPACE = [
    [-1,   1, 0.2], [0,   1, 0.2], [1,   1, 0.2],
    [-1,   0, 0.2], [0,   0, 0.2], [1,   0, 0.2],
    [-1,   1,   0], [0,   1,   0], [1,   1,   0],
    [-1, 0.2,   0], [0, 0.2,   0], [1, 0.2,   0]
]

def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 96, 96)

class brain:
    def __init__(self, EstimatorName) -> None:
        self.name = "SelectiveCNN"
        self.saveName = "saves/" + self.name + '_' + EstimatorName + '.h5'
        try:
            self.model = load_model(self.saveName)
            print("\n======> LOAD MODEL\n")
        except:
            print("\n======> CREATE NEW MODEL\n")
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=8, padding='same', input_shape=(1, 96, 96), activation='relu')) # Peut etre changer 1 pour le nombre de frame
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            self.model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

            self.model.add(Flatten())
            self.model.add(Dense(128, activation='sigmoid', name='dense1'))
            self.model.add(Dense(32, activation='sigmoid', name='dense2'))
            self.model.add(Dense(len(ACTION_SPACE), activation='softmax', name='dense3'))
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, epsilon=1e-7), metrics=['accuracy'])
        self.model.summary()
        self.epsilon = 0

    def __str__(self) -> str:
        return self.name

    def save(self, score):
        self.model.save(self.saveName)

    def predict(self, observation = None, check=False):
        if check:
            return
        if random.uniform(0, 1) > self.epsilon:
            action = self.model.predict(clearObservation(observation), verbose=False)[0]
            index = np.argmax(action)
        else:
            index = random.randint(0, len(ACTION_SPACE) - 1)
        step = ACTION_SPACE[index].copy()
        step[0] = (step[0] + 1) / 2
        return {"step": step, "index": index}

    def getWeights(self, name, bias=False):
        return self.model.get_layer(name).get_weights()[bias]

    def getAllWeights(self):
        res = {}

        res["dense1"] = {}
        res["dense1"]["weight"] = self.getWeights("dense1")
        res["dense1"]["bias"] = self.getWeights("dense1", bias=True)

        res["dense2"] = {}
        res["dense2"]["weight"] = self.getWeights("dense2")
        res["dense2"]["bias"] = self.getWeights("dense2", bias=True)

        res["dense3"] = {}
        res["dense3"]["weight"] = self.getWeights("dense3")
        res["dense3"]["bias"] = self.getWeights("dense3", bias=True)
        return res


    def train(self, weights:dict=None, check=False):
        if check:
            return
        for key in weights:
            formatedWeights = [weights[key]["weight"], weights[key]["bias"]]
            self.model.get_layer(key).set_weights(formatedWeights)