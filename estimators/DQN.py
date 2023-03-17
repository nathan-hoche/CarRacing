import numpy as np
import random
import sys
import importlib

EPSILON_MIN = 0.05
GAMMA = 0.95
BATCH_SIZE = 128
EPSILON_DECAY = 0.99
EPSILON_START = 0.25
TAU = 0.005

############# UPDATE FUNC #############

def clearObservation(observation):
    obs = np.zeros((94, 94))
    for x in range(0, 94):
        for y in range(0, 94):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1, 94, 94)

def dqnUpdate(brain, memory:list[dict], targetModel):
    model = brain.model
    lim = len(memory)
    if (lim > BATCH_SIZE):
        lim = BATCH_SIZE

    sys.stdout.write("Mutation: [%s]" % (" " * lim))
    sys.stdout.flush()
    sys.stdout.write("\b" * (lim+1))
        
    minibatch = random.sample(memory, lim)
    train_state = []
    train_target = []
    clearFunc = brain.clearObservation if hasattr(brain, "clearObservation") else clearObservation
    shapeFunc = brain.reshapeObservation if hasattr(brain, "reshapeObservation") else (lambda x: (x, 1, 94, 94))
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(clearFunc(state), verbose=False)[0]
        if done:
            target[action["index"]] = reward
        else:
            target[action["index"]] = reward + GAMMA * np.amax(targetModel.predict(clearFunc(next_state), verbose=False)[0])
        train_state.append(clearFunc(state))
        train_target.append(target)
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\n")
    model.fit(np.array(train_state).reshape(shapeFunc(lim)), np.array(train_target), epochs=1, verbose=1)
    if hasattr(brain, "epsilon") and brain.epsilon > EPSILON_MIN:
        #brain.epsilon = brain.epsilon * EPSILON_DECAY ** (len(memory) / 100)
        brain.epsilon = brain.epsilon * EPSILON_DECAY
        print("Epsilon: ", brain.epsilon)
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
    def __init__(self) -> None:
        self.name = "DQN"
        self.bestWeights, self.bestScore = None, -1
        self.memory = []
        self.memoryPos = 0
        self.epsilonSetup = False
        self.target = None

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return
        self.memory.append((observation, step, reward, nextObservation, self.memoryPos))
        self.memoryPos += 1

    def update(self, brain:object=None, score=None, check=False):
        if check:
            return
        brain.save(score)

        if not self.epsilonSetup and hasattr(brain, "epsilon"):
            self.epsilonSetup = True
            fd = importlib.import_module(brain.__str__())
            self.target = fd.brain(self.name)
            self.target.train(brain.getAllWeights())
            brain.epsilon = EPSILON_START
            print("CONFIG DONE")
            return

        #weights = brain.getAllWeights()
        # if score > self.bestScore:
        #     self.bestScore = score
        #     self.bestWeights = weights
        #     brain.save(score)
        # else:
        #     brain.train(self.bestWeights)
        returnValue = dqnUpdate(brain, self.memory, self.target.model)

        weights = brain.getAllWeights()
        targetWeights = self.target.getAllWeights()

        for layer in weights.keys():
            for key in weights[layer].keys():
                targetWeights[layer][key] = TAU * weights[layer][key] + (1 - TAU) * targetWeights[layer][key]
        self.target.train(targetWeights)

        self.memory = []
        self.memoryPos = 0
        return returnValue