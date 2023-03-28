import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

class stats():
    def __init__(self, name:str=None, isRead:bool=False) -> None:
        self.isRead = isRead
        if name != None:
            self.filename = "saves/" + name + ".csv"
        else:
            return
        self.df = None
        if self.isRead == True:
            self.df = pd.read_csv(self.filename, sep=";")
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
        
    def AllMaxGenerations(self):
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv"):
                tab = {"Max score": []}
                tabMax = -1
                tmp = pd.read_csv("saves/" + filename, sep=";")
                if "NEAT" in filename or "Genetic" in filename:
                    tmp = tmp.groupby(tmp.index // 50).max()
                for i in range(0, len(tmp["Max score"])):
                    tabMax = max(tabMax, tmp["Max score"][i])
                    tab["Max score"].append(tabMax)
                    
                dfs[filename[:-4]] = tab
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        plt.xlabel("Number of simulations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()
       
    def AllGenerations(self):
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv"):
                tmp = pd.read_csv("saves/" + filename, sep=";")
                if "NEAT" in filename or "Genetic" in filename:
                    tmp = tmp.groupby(tmp.index // 50).max()
                dfs[filename[:-4]] = tmp
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        plt.xlabel("Number of simulations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def AllSimulations(self):
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv"):
                tmp = pd.read_csv("saves/" + filename, sep=";")
                dfs[filename[:-4]] = tmp
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        plt.xlabel("Number of simulations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def AllMaxSimulations(self):
        os.listdir("saves")
        dfs = {}
        for filename in os.listdir("saves"):
            if filename.endswith(".csv"):
                tab = {"Max score": []}
                tabMax = -1
                tmp = pd.read_csv("saves/" + filename, sep=";")
                for i in range(0, len(tmp["Max score"])):
                    tabMax = max(tabMax, tmp["Max score"][i])
                    tab["Max score"].append(tabMax)
                    
                dfs[filename[:-4]] = tab
        for key, value in dfs.items():
            plt.plot(value["Max score"], label=key)
        plt.title("Max score")
        plt.xlabel("Number of simulations")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stats.py <nom>")
        exit(1)
    elif sys.argv[1] == "ALL":
        stats = stats(isRead=True)
        Possibilities = {"MaxSimulation": stats.AllMaxSimulations, "MaxGeneration": stats.AllMaxGenerations, "Simulation": stats.AllSimulations, "Generation": stats.AllGenerations}
        if len(sys.argv) < 3 or sys.argv[2] not in Possibilities:
            print("Usage: python stats.py ALL <MaxSimulation/MaxGeneration/Simulation/Generation>")
            exit(1)
        Possibilities[sys.argv[2]]()
    else:
        stats = stats(sys.argv[1], isRead=True)
        stats.read()