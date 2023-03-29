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
        self.estimatorName = estimatorName
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
        if self.estimatorName == "NEATKNN":
            tmp = self.network.activate(clearObservation(observation).flatten())
        elif self.estimatorName == "NEATCNN":
            tmp = self.network.activate((np.dot(observation, [0.2989, 0.5870, 0.1140])).flatten())
        return [tmp]



    def train(self, weights:dict=None, check=False):
        pass