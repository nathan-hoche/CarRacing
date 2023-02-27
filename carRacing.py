import gym
import sys
import os
import importlib

ENV = gym.make("CarRacing-v2", render_mode="rgb_array")

# Actions: [steering, gas, brake] [{-1 ... 1}, {0 ... 1}, {0 ... 1}]
# Observaton space: [[r, g, b] * 94] * 94] (RGB image)

PENALITY = 0.10

def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

def loadBrain(newtorkFile: str, estimatorFile) -> object:
    """
    Load an network from a file
    """
    try:
        sys.path.append(os.getcwd() + "/networks/")
        networkFd = importlib.import_module(newtorkFile.replace(".py", ""))
        print("Load : ", newtorkFile, "found.")
    except Exception as e:
        print("ERROR: File", newtorkFile, "not found.")
        print(e)
        exit(0)
    try:
        networkClassFd = networkFd.brain()
        print("Load: Class found. -> ", type(networkClassFd))
        ####### Check if sample fonction is set
        networkClassFd.train(check=True)
        networkClassFd.predict(check=True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)

    """
    Load an estimator from a file
    """
    try:
        sys.path.append(os.getcwd() + "/estimators/")
        estimatorFd = importlib.import_module(estimatorFile.replace(".py", ""))
        print("Load : ", estimatorFile, "found.")
    except Exception as e:
        print("ERROR: File", estimatorFile, "not found.")
        print(e)
        exit(0)
    try:
        estimatorClassFd = estimatorFd.estimator(networkClassFd.__str__())
        print("Load: Class found. -> ", type(estimatorClassFd))
        ####### Check if sample fonction is set
        estimatorClassFd.update(check=True)
        estimatorClassFd.getBestCase(check=True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)
    return networkClassFd, estimatorClassFd

def main(brain, estimator):
    BRAIN, ESTIMATOR = loadBrain(brain, estimator)
    print("config: ", ESTIMATOR, BRAIN)
    for _ in range(100): # Number of simulations

        observation, info = ENV.reset()
        score = 0
        Max_score = 0
        while score >= 0:
            step = BRAIN.predict(observation)
            step = step[0]
            step[0] = step[0] * 2 - 1
            step[1] = 1
            step[2] = 0
            # print("Move ", "\t".join(["{}{:0.2f}".format(("+" if x >= 0 else ""),x) for x in step]), end=" \t")
            print("Move:", "{}{:0.2f}".format(("+" if step[0] >= 0 else ""),step[0]), step[1], step[2], sep="\t", end=" \t")

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
    sys.argv.pop(0)
    if len(sys.argv) == 2:
        main(sys.argv[0], sys.argv[1])
    else:
        Bdir =  os.listdir(os.getcwd() + "/networks/")
        Bdir = [x.replace(".py", "") for x in Bdir if x.endswith(".py")]
        Edir =  os.listdir(os.getcwd() + "/estimators/")
        Edir = [x.replace(".py", "") for x in Edir if x.endswith(".py")]
        print("ERROR: Invalid number of arguments")
        print("USAGE: python3 carRacing.py Brain Estimator")
        print("EXAMPLE: python3 carRacing.py CNN Custom")
        print("List of Brain:\n\t-", "\n\t- ".join(Bdir))
        print("List of Estimator:\n\t-", "\n\t- ".join(Edir))
    ENV.close()