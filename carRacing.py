import gym
import random
from Brain import brain
import numpy as np

env = gym.make("CarRacing-v2", render_mode="human")

# Actions: [steering, gas, brake] [{-1 ... 1}, {0 ... 1}, {0 ... 1}]
# Observaton space: [[r, g, b] * 94] * 94] (RGB image)

BRAIN = brain()

for _ in range(1): # Nombre de simulations

    observation, info = env.reset()
    score = 0
    while score >= 0:
        step = BRAIN.predict(observation)
        print(step)
        step = step[0]
        step[0] = step[0] * 2 - 1
        step[1] = 1
        step[2] = 0

        observation, reward, terminated, truncated, info = env.step(step)

        if terminated or truncated:
            break
        score += reward
        print(score)
    
    step = BRAIN.train(wd3=[np.random.choice(BRAIN.getWeights("dense3").reshape(60), size=60).reshape(20, 3), np.zeros(3,)])

env.close()