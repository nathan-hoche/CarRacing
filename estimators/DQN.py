import numpy as np
import json

############# UPDATE FUNC #############

def dqnUpdate(narray):
    
    return narray

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
        self.memory.append({"observation": observation, "step": step, "reward": reward, "nextObservation": nextObservation})

    def update(self, weights:dict = None, score = None, check=False):
        if check:
            return
        if score > self.bestScore:
            self.bestScore = score
            self.bestWeights = weights
            self.saveBestWeights()
        return loopAll(self.bestWeights, dqnUpdate)