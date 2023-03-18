import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

class stats():
    def __init__(self, nom:str, isRead:bool=False) -> None:
        self.nom = nom
        self.isRead = isRead
        self.filename = "saves/" + self.nom + ".csv"
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stats.py <nom>")
        exit(1)
    stats = stats(sys.argv[1], isRead=True)
    stats.read()