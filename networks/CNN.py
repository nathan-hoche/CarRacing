from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np

def clearObservation(observation):
    obs = np.zeros((94, 94))
    for x in range(0, 94):
        for y in range(0, 94):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 94, 94)

class brain:
    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(Conv2D(20, kernel_size=5, padding='same', input_shape=(1, 94, 94), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Conv2D(5, kernel_size=5, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(30, activation='sigmoid', name='dense1'))
        self.model.add(Dense(20, activation='sigmoid', name='dense2'))
        self.model.add(Dense(3, activation='sigmoid', name='dense3'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def __str__(self) -> str:
        return "CNN"

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


    def train(self, wd1=None, wd2=None, wd3=None, check=False):
        if check:
            return;
        if wd1 is not None:
            self.model.get_layer("dense1").set_weights(wd1)
        if wd2 is not None:
            self.model.get_layer("dense2").set_weights(wd2)
        if wd3 is not None:
            self.model.get_layer("dense3").set_weights(wd3)



