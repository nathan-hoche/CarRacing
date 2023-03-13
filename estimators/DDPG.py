import numpy as np
import json
import sys

############# UPDATE FUNC #############

def randomUpdate(narray):
    return narray + np.random.uniform(-0.1, 0.1, narray.shape)

def randomChoiceWeights(narray):
    return np.random.choice(narray.reshape(narray.size), narray.size).reshape(narray.shape)

def fullRandomWeights(narray):
    return np.random.uniform(-1, 1, narray.size).reshape(narray.shape)


############# UTILS #############

def convertToNumpy(arr):
    score = arr.pop("score")
    for x in arr.keys():
        for y in arr[x].keys():
            arr[x][y] = np.array(arr[x][y])
    return arr, score

def loopAll(weights: dict, func) -> dict:
    res = {}

    for x in weights.keys():
        res[x] = {}
        for y in weights[x].keys():
            res[x][y] = func(weights[x][y])
    return res

def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

############# CLASS #############

class ReplayMemory:
    def __init__(self, capacity, inputShape, actionsNumber):
        self.capacity = capacity
        self.memoryCounter = 0
        self.memory = [np.zeros((self.capacity, *inputShape), dtype=np.float32)],
        self.newMemory = [np.zeros((self.capacity, *inputShape), dtype=np.float32)],
        self.actionMemory = [np.zeros((self.capacity, actionsNumber), dtype=np.float32)],
        self.rewardMemory = np.zeros(self.capacity, dtype=np.float32),
        self.terminalMemory = np.zeros(self.capacity, dtype=np.float32)

    def storeTransition(self, state, action, reward, newState, done):
        index = self.memoryCounter % self.capacity
        self.memory[index] = state
        self.newMemory[index] = newState
        self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        self.terminalMemory[index] = done

        self.memoryCounter += 1

    def sampleBuffer(self, batchSize):
        maxMemory = min(self.memoryCounter, self.capacity)
        batch = np.random.choice(maxMemory, batchSize)

        states = self.memory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        newStates = self.newMemory[batch]
        dones = self.terminalMemory[batch]

        return states, actions, rewards, newStates, dones

class estimator:
    def __init__(self) -> None:
        self.name = "DDPG"
        self.model = None
        raise NotImplementedError(f"{self.name} is not implemented yet")

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return

    def setup(self, weights: object):
        return

    def applyWeights(self, chromosome):
        weights = {}

        self.model = weights

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return

        return self.model