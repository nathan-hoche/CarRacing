import numpy as np
import json

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

############# CLASS #############

class estimator:
    def __init__(self) -> None:
        self.name = "Custom"
        self.bestWeights, self.bestScore = None, -1

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        pass

    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        weights = brain.getAllWeights()
        if score > self.bestScore:
            self.bestScore = score
            self.bestWeights = weights
            brain.save(score)
        return loopAll(self.bestWeights, randomUpdate)