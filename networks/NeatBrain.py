import neat
import numpy as np
from sklearn.cluster import KMeans



def clearObservation(observation):
    obs = np.zeros((96, 96))
    for x in range(0, 96):
        for y in range(0, 96):
            obs[x][y] = observation[x][y].mean()
    return KMeans(n_clusters=3, random_state=0, n_init="auto").fit_predict(obs).reshape(1, 1, 96)


class brain:
    def __init__(self, estimatorName, genome=None, config=None):
        self.name: str = "NEAT"
        self.network : neat.nn.FeedForwardNetwork = None
        if (genome != None and config != None):
            self.network  = neat.nn.FeedForwardNetwork.create(genome[1], config)



    def __str__(self) -> str:
        return "NeatBrain"

    # def save(self, score):
    #     self.saver.save_checkpoint(config=self.config, population=self.population, species_set=self.population.species, generation=self.population.generation, filename=self.saveName)


    def predict(self, observation = None, check=False):
        if check:
            return;
        if (self.network == None):
            print("[ERROR BRAIN] Network not load")
            return [[0, 0, 0]]
        tmp = self.network.activate(clearObservation(observation).flatten())
        return [tmp]



    def train(self, weights:dict=None, check=False):
        pass