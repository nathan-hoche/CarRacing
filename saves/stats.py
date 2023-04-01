import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def GetGeneration(filename:str) -> dict:
    tmp = pd.read_csv(filename, sep=";")
    if "NEAT" in filename or "Genetic" in filename:
        tmp = tmp.groupby(tmp.index // 50).max()
    return tmp

def GetMaxGeneration(filename:str) -> dict:
    tab = {"Max score": []}
    tabMax = -1
    tmp = pd.read_csv(filename, sep=";")
    if "NEAT" in filename or "Genetic" in filename:
        tmp = tmp.groupby(tmp.index // 50).max()
    for i in range(0, len(tmp["Max score"])):
        tabMax = max(tabMax, tmp["Max score"][i])
        tab["Max score"].append(tabMax)
    return tab

def GetSimulations(filename:str) -> dict:
    tmp = pd.read_csv(filename, sep=";")
    return tmp[:2000]

def GetMaxSimulations(filename:str) -> dict:
    tab = {"Max score": []}
    tabMax = -1
    tmp = pd.read_csv(filename, sep=";")
    for i in range(0, len(tmp["Max score"])):
        tabMax = max(tabMax, tmp["Max score"][i])
        tab["Max score"].append(tabMax)
    tab["Max score"] = tab["Max score"][:2000]
    return tab


class stats():
    def __init__(self, name:str=None, isRead:bool=False) -> None:
        self.isRead = isRead
        if name != None:
            self.filename = "saves/" + name + ".csv"
        else:
            return
        self.df = None
        if self.isRead == True:
            try :
                self.df = pd.read_csv(self.filename, sep=";")
            except:
                print("Error: " + self.filename + " not found")
                sys.exit()
        elif (not os.path.isfile(self.filename)):
            self.fd = open(self.filename, "w+")
            self.fd.write("Max score;Average score;Last score" + "\n")
            self.fd.close()
            self.allreadyExist = False
        else:
            self.allreadyExist = True

    def __str__(self) -> str:
        return self.nom

    def save(self, max, average, lastScore):
        if self.isRead == True:
            print("Error: not in write mode")
            return None
        if self.allreadyExist == True:
            self.allreadyExist = False
            return None

        fd = open(self.filename, "a")
        fd.write(str(max) + ";" + str(average) + ";" + str(lastScore) + "\n")
        fd.close()

    def read(self):
        if self.df is None:
            print("Error: not in read mode")
            return None
        plt.ion()
        fig, axs = plt.subplots(3, 1)
        fig.tight_layout(pad=1.2)
        mGraph, = axs[0].plot(self.df["Max score"])
        axs[0].set_title("Max score")
        aGraph, = axs[1].plot(self.df["Average score"])
        axs[1].set_title("Average score")
        axs[2].set_title("Last score")
        lGraph, = axs[2].plot(self.df["Last score"])
        stamp = os.stat(self.filename).st_mtime
        fig.canvas.draw()
        fig.canvas.flush_events()
        while (plt.fignum_exists(fig.number)):
            if stamp != os.stat(self.filename).st_mtime:
                stamp = os.stat(self.filename).st_mtime
                self.df = pd.read_csv(self.filename, sep=";")
                mGraph.set_xdata(range(0, len(self.df["Max score"])))
                mGraph.set_ydata(self.df["Max score"])
                aGraph.set_xdata(range(0, len(self.df["Average score"])))
                aGraph.set_ydata(self.df["Average score"])
                lGraph.set_xdata(range(0, len(self.df["Last score"])))
                lGraph.set_ydata(self.df["Last score"])
                fig.canvas.flush_events()
                fig.canvas.draw()
            plt.pause(0.05)
        fig.savefig("img/stats/" + self.filename[:-4].replace("saves/", "") + ".png")
    
    
    def AllStats(self, type):
        Possibilities = {"MaxSimulation":GetMaxSimulations, "MaxGeneration":GetMaxGeneration, "Simulation":GetSimulations, "Generation":GetGeneration}
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv"):
                dfs[filename[:-4]] = Possibilities[type]("saves/" + filename)
        if len(dfs) == 0:
            print("No stats found for : " + type)
            return
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        if "Simulation" in type:
            plt.xlabel("Number of simulations")
        else:
            plt.xlabel("Number of generations")
        plt.ylabel("Score")
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.legend(loc='upper left')
    
    def SpecficStats(self, type, stats):
        Possibilities = {"MaxSimulation":GetMaxSimulations, "MaxGeneration":GetMaxGeneration, "Simulation":GetSimulations, "Generation":GetGeneration}
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv") and type in filename:
                dfs[filename[:-4]] = Possibilities[stats]("saves/" + filename)
        if len(dfs) == 0:
            print("No stats found for : " + type)
            return
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        if "Simulation" in type:
            plt.xlabel("Number of simulations")
        else:
            plt.xlabel("Number of generations")
        plt.ylabel("Score")
        plt.legend()


SAVE = True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stats.py <nom>")
        exit(1)
    elif sys.argv[1] == "ALL":
        stats = stats(isRead=True)
        Possibilities = ["MaxSimulation", "MaxGeneration", "Simulation", "Generation"]
        if len(sys.argv) < 3 or sys.argv[2] not in Possibilities:
            print("Usage: python stats.py ALL <MaxSimulation/MaxGeneration/Simulation/Generation>")
            exit(1)
        stats.AllStats(sys.argv[2])
        plt.show()

    elif sys.argv[1] == "SPECIFIC":
        if len(sys.argv) < 4:
            print("Usage: python stats.py SPECIFIC <type> <MaxSimulation/MaxGeneration/Simulation/Generation>")
            exit(1)
        stats = stats(isRead=True)
        stats.SpecficStats(sys.argv[2], sys.argv[3])
        if SAVE:
            plt.savefig("img/stats/" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".png")
        plt.show()
    else:
        stats = stats(sys.argv[1], isRead=True)
        stats.read()
        plt.show()