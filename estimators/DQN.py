import numpy as np
import random
import sys

EPSILON_MIN = 0.1
GAMMA = 0.95
BATCH_SIZE = 64
EPSILON_DECAY = 0.999 - BATCH_SIZE / 1000

############# UPDATE FUNC #############

def clearObservation(observation):
    obs = np.zeros((94, 94))
    for x in range(0, 94):
        for y in range(0, 94):
            obs[x][y] = observation[x][y].mean()

    return obs.reshape(1, 1,94, 94)

def dqnUpdate(brain, memory:list[dict]):
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
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(clearObservation(state), verbose=False)[0]
        if done:
            target[action["index"]] = reward
        else:
            target[action["index"]] = reward + GAMMA * np.amax(model.predict(clearObservation(next_state), verbose=False)[0])
        train_state.append(clearObservation(state))
        train_target.append(target)
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\n")
    model.fit(np.array(train_state).reshape(lim, 1, 94, 94), np.array(train_target), epochs=1, verbose=1)
    if hasattr(brain, "epsilon") and brain.epsilon > EPSILON_MIN:
        brain.epsilon *= EPSILON_DECAY
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
        weights = brain.getAllWeights()
        if score > self.bestScore:
            self.bestScore = score
            self.bestWeights = weights
            brain.save(score)
        else:
            brain.train(self.bestWeights)
        returnValue = dqnUpdate(brain, self.memory)
        self.memory = []
        self.memoryPos = 0
        return returnValue