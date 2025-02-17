import gym
import sys
import os
import importlib

# For save the observation
#from PIL import Image

# FOR GENERATE VIDEO
SAVE = True

if SAVE:
    # FOR GENERATE VIDEO
    # from gym.utils.save_video import save_video
    ENV = gym.make("CarRacing-v2", render_mode="rgb_array")
else:
    ENV = gym.make("CarRacing-v2", render_mode="human")

# Actions: [steering, gas, brake] [{-1 ... 1}, {0 ... 1}, {0 ... 1}]
# Observaton space: [[r, g, b] * 94] * 94] (RGB image)

PENALITY = 0.10
LIMIT_NEGATIVE_STEP = 100

def skipUselessStep():
    for _ in range(100):
        ENV.step([0, 0, 0])

def loadBrain(newtorkFile: str, estimatorName: str) -> object:
    """
    Load an network and an estimator from a file
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
        networkClassFd = networkFd.brain(estimatorName)
        print("Load: Class found. -> ", type(networkClassFd))
        ####### Check if sample fonction is set
        networkClassFd.train(check=True)
        networkClassFd.predict(check=True)
        #######################################
    except Exception as e:
        print("ERROR: class/method crashed")
        print(e)
        exit(0)
    return networkClassFd

def main(brain, estimatorName, seed=None):
    BRAIN = loadBrain(brain, estimatorName)

    print("config: ", BRAIN, estimatorName + "\tseed:", seed)

    observation, info = ENV.reset(seed=seed)

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

        observation = nextObservation

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
    # FOR GENERATE VIDEO
    # if SAVE:
    #    save_video(ENV.render(), "img/video/" + BRAIN.__str__() + '_' + estimatorName, fps=ENV.metadata["render_fps"])

    print("=====================================> END OF SIMULATION")
    averageScore = sum(allScore) / len(allScore)
    print("Max score:", MaxScore, "\tAverage score:", averageScore)


if __name__ == "__main__":
    sys.argv.pop(0)
    if len(sys.argv) == 2:
        main(sys.argv[0], sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[0], sys.argv[1], int(sys.argv[2]))
    else:
        Bdir =  os.listdir(os.getcwd() + "/networks/")
        Bdir = [x.replace(".py", "") for x in Bdir if x.endswith(".py")]
        Edir =  os.listdir(os.getcwd() + "/estimators/")
        Edir = [x.replace(".py", "") for x in Edir if x.endswith(".py")]
        print("ERROR: Invalid number of arguments")
        print("USAGE: python3 vizualizer.py Brain Estimator [seed]")
        print("EXAMPLE: python3 carRacing.py CNN Custom")
        print("Seed (int) is optional, by default is random")
        print("List of Brain:\n\t-", "\n\t- ".join(Bdir))
        print("List of Estimator:\n\t-", "\n\t- ".join(Edir))
    ENV.close()