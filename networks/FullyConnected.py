from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np

def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 96, 96)

class brain:
    def __init__(self, EstimatorName) -> None:
        self.name = "FullyConnected"
        self.saveName = "saves/" + self.name + '_' + EstimatorName + '.h5'
        try:
            self.model = load_model(self.saveName)
            print("\n======> LOAD MODEL\n")
        except:
            print("\n======> CREATE NEW MODEL\n")
            self.model = Sequential()
            self.model.add(Flatten(input_shape=(1, 96, 96)))
            self.model.add(Dense(32, activation='relu', name='dense1'))
            self.model.add(Dense(64, activation='sigmoid', name='dense2'))
            self.model.add(Dense(3, activation='sigmoid', name='dense3'))

            self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        self.model.summary()

    def __str__(self) -> str:
        return self.name

    def save(self, score):
        self.model.save(self.saveName)

    def predict(self, observation = None, check=False):
        if check:
            return;
        return self.model.predict(clearObservation(observation), verbose=False)

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



