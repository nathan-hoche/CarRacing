import gym
import sys
import os
import importlib

from saves.stats import stats
# For save the observation
#from PIL import Image

ENV = gym.make("CarRacing-v2", render_mode="rgb_array")
print("ENV obs must be 96 ==== ",ENV.observation_space.shape[0])

print("ENV obs must be 3 ==== ",ENV.action_space.shape[0])

# Actions: [steering, gas, brake] [{-1 ... 1}, {0 ... 1}, {0 ... 1}]
# Observaton space: [[r, g, b] * 96] * 96] (RGB image)

PENALITY = 0.10
LIMIT_NEGATIVE_STEP = 100

def skipUselessStep():
    for _ in range(100):
        ENV.step([0, 0, 0])

def loadBrain(newtorkFile: str, estimatorFile) -> object:
    """
    Load an network and an estimator from a file
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
        estimatorClassFd = estimatorFd.estimator()
        print("Load: Class found. -> ", type(estimatorClassFd))
        ####### Check if sample fonction is set
        estimatorClassFd.update(check=True)
        estimatorClassFd.memorize(check=True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)
    try:
        sys.path.append(os.getcwd() + "/networks/")
        networkFd = importlib.import_module(newtorkFile.replace(".py", ""))
        print("Load : ", newtorkFile, "found.")
    except Exception as e:
        print("ERROR: File", newtorkFile, "not found.")
        print(e)
        exit(0)
    try:
        networkClassFd = networkFd.brain(estimatorClassFd.__str__())
        print("Load: Class found. -> ", type(networkClassFd))
        ####### Check if sample fonction is set
        networkClassFd.train(check=True)
        networkClassFd.predict(check=True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)
    return networkClassFd, estimatorClassFd

def log(score, maxScore, nbNegatif):
    print("Meab:", score, "\tMax:", maxScore, "\tNeg:", nbNegatif, end="\r")

def main(brain, estimator):
    BRAIN, ESTIMATOR = loadBrain(brain, estimator)

    STATS = stats(BRAIN.__str__() + "_" + ESTIMATOR.__str__())

    ## If estimator has setup function call it
    ESTIMATOR.setup(BRAIN) if hasattr(ESTIMATOR, "setup") else None

    print("config: ", BRAIN, ESTIMATOR)
    for _ in range(10000): # Number of simulations

        observation, info = ENV.reset(seed=1)
        score = 10
        MaxScore = 0
        allScore = []
        skipUselessStep()
        nbNegatif = 0
        while score >= 0:
            action = BRAIN.predict(observation)
            step = action[0] if type(action) != dict else action["step"]
            step[0] = step[0] * 2 - 1
            # To go forward
            # step[1] = 1
            # To remove the brake
            # step[2] = 0

            print("Move:", "{}{:0.2f}\t{:0.2f}\t{:0.2f}".format(("+" if step[0] >= 0 else ""), step[0], step[1], step[2]), end=" \t")

            nextObservation, reward, terminated, truncated, info = ENV.step(step)
            ESTIMATOR.memorize(observation, action, reward, nextObservation)

            observation = nextObservation
            # To save the observation
            # im = Image.fromarray(observation)
            # im.save("test.png")

            if terminated or truncated or nbNegatif == LIMIT_NEGATIVE_STEP:
                break
            score += reward - PENALITY
            if score > MaxScore:
                MaxScore = score
            if reward < 0:
                nbNegatif += 1
            else:
                nbNegatif = 0
            allScore.append(score)
            print("Score:", score)

        print("=====================================> END OF SIMULATION")
        averageScore = sum(allScore) / len(allScore)
        print("Max score:", MaxScore, "\tAverage score:", averageScore)
        STATS.save(MaxScore, averageScore, score)
        newWeight = ESTIMATOR.update(BRAIN, MaxScore)
        if newWeight != None:
            BRAIN.train(newWeight)

def help():
    Bdir =  os.listdir(os.getcwd() + "/networks/")
    Bdir = [x.replace(".py", "") for x in Bdir if x.endswith(".py")]
    Edir =  os.listdir(os.getcwd() + "/estimators/")
    Edir = [x.replace(".py", "") for x in Edir if x.endswith(".py")]
    print("USAGE: python3 carRacing.py Brain Estimator")
    print("EXAMPLE: python3 carRacing.py CNN Custom")
    print("List of Brain:\n\t-", "\n\t- ".join(Bdir))
    print("List of Estimator:\n\t-", "\n\t- ".join(Edir))


if __name__ == "__main__":
    sys.argv.pop(0)
    if "-h" in sys.argv or "--help" in sys.argv:
        help()
    elif len(sys.argv) == 2:
        if (sys.argv[0] == "NeatBrain" and sys.argv[1] != "NEAT") or (sys.argv[0] != "NeatBrain" and sys.argv[1] == "NEAT"):
            print("ERROR: NEAT need to be used with NEAT")
            help()
        else:
            main(sys.argv[0], sys.argv[1])
    else:
        print("ERROR: Invalid number of arguments")
        help()

    ENV.close()