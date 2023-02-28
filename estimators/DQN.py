import numpy as np
import random
import json
import sys

############# UPDATE FUNC #############

def clearObservation(observation):
    obs = np.zeros((94, 94))
    for x in range(0, 94):
        for y in range(0, 94):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 94, 94)

def dqnUpdate(model, memory:list[dict]):
    batch_size = 32
    gamma = 0.95

    sys.stdout.write("[%s]" % (" " * batch_size))
    sys.stdout.flush()
    sys.stdout.write("\b" * (batch_size+1))

    minibatch = random.sample(memory, batch_size)
    for state, _, reward, next_state in minibatch:
        target = reward + gamma * np.array(model.predict(clearObservation(next_state), verbose=False))
        model.fit(clearObservation(state), target, epochs=1, verbose=False)
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\n")
    return None

############# UTILS #############

def convertToNumpy(arr):
    score = arr.pop("score")
    for x in arr.keys():
        for y in arr[x].keys():
            arr[x][y] = np.array(arr[x][y])
    return arr, score

############# CLASS #############

class estimator:
    def __init__(self, networkName) -> None:
        self.networkName = networkName
        self.name = "DQN"
        try:
            with open("saves/" + self.networkName + "_" + self.name + ".json" , 'r') as fd:
                self.bestWeights, self.bestScore = convertToNumpy(json.load(fd))
        except:
            self.bestWeights, self.bestScore = None, -1
        self.memory = []

    def __str__(self) -> str:
        return self.name

    def saveBestWeights(self):
        res = {"score": self.bestScore}

        for x in self.bestWeights.keys():
            res[x] = {}
            for y in self.bestWeights[x].keys():
                res[x][y] = self.bestWeights[x][y].tolist()
        save = open("saves/" + self.networkName + "_" + self.name + ".json", "w")
        json.dump(res, save, indent=4)

    def getBestCase(self, check=False):
        if check:
            return
        return self.bestWeights

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        self.memory.append((observation, step, reward, nextObservation))

    def update(self, brain:object=None, score=None, check=False):
        if check:
            return
        weights = brain.getAllWeights()
        if score > self.bestScore:
            self.bestScore = score
            self.bestWeights = weights
            self.saveBestWeights()
        return dqnUpdate(brain.model, self.memory)