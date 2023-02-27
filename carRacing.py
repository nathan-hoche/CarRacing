import gym
from Brain import brain
from Estimator import estimator

ENV = gym.make("CarRacing-v2", render_mode="human")

# Actions: [steering, gas, brake] [{-1 ... 1}, {0 ... 1}, {0 ... 1}]
# Observaton space: [[r, g, b] * 94] * 94] (RGB image)

ESTIMATOR = estimator()
BRAIN = brain()
PENALITY = 0.10


def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

save = ESTIMATOR.getBestCase()
BRAIN.train(wd1=formatWeights(save["dense1"]), wd2=formatWeights(save["dense2"]), wd3=formatWeights(save["dense3"]))

def main():
    for _ in range(100): # Nombre de simulations

        observation, info = ENV.reset()
        score = 0
        Max_score = 0
        while score >= 0:
            step = BRAIN.predict(observation)
            step = step[0]
            step[0] = step[0] * 2 - 1
            step[1] = 1
            step[2] = 0
            print("Move:", step, end=" \t")

            observation, reward, terminated, truncated, info = ENV.step(step)

            if terminated or truncated:
                break
            score += reward - PENALITY
            if score > Max_score:
                Max_score = score
            print("Score:", score)

        print("=====================================> END OF SIMULATION")
        print("Max score:", Max_score)
        newWeight = ESTIMATOR.update(BRAIN.getAllWeights(), Max_score)
        step = BRAIN.train(wd1=formatWeights(newWeight["dense1"]), wd2=formatWeights(newWeight["dense2"]), wd3=formatWeights(newWeight["dense3"]))

if __name__ == "__main__":
    main()
    ENV.close()