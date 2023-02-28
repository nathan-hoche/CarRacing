import numpy as np
import random
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

def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

def convertToNumpy(arr):
    score = arr.pop("score")
    for x in arr.keys():
        for y in arr[x].keys():
            arr[x][y] = np.array(arr[x][y])
    return arr, score

############# CLASS #############

class estimator:
    def __init__(self) -> None:
        self.name = "DQN"
        self.bestWeights, self.bestScore = None, -1
        self.memory = []

    def __str__(self) -> str:
        return self.name

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
            brain.save(score)
        else:
            brain.train(wd1=formatWeights(self.bestWeights["dense1"]), wd2=formatWeights(self.bestWeights["dense2"]), wd3=formatWeights(self.bestWeights["dense3"]))
        return dqnUpdate(brain.model, self.memory)