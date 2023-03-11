import numpy as np
import json

############# UPDATE FUNC #############

def randomUpdate(narray):
    return narray + np.random.uniform(-0.1, 0.1, narray.shape)

def randomChoiceWeights(narray):
    return np.random.choice(narray.reshape(narray.size), narray.size).reshape(narray.shape)

def fullRandomWeights(narray):
    return np.random.uniform(-1, 1, narray.size).reshape(narray.shape)


############# UTILS #############

def convertToNumpy(arr):
    score = arr.pop("score")
    for x in arr.keys():
        for y in arr[x].keys():
            arr[x][y] = np.array(arr[x][y])
    return arr, score

def loopAll(weights: dict, func) -> dict:
    res = {}

    for x in weights.keys():
        res[x] = {}
        for y in weights[x].keys():
            res[x][y] = func(weights[x][y])
    return res

############# CLASS #############

class individual:
    def __init__(self, chromosome) -> None:
        self.genes = 32
        self.chromosome = chromosome
        self.fitness = 0
        self.model = None

    def __str__(self) -> str:
        return str(self.chromosome)

    def getChromosome(self):
        return self.chromosome

    def setChromosome(self, chromosome):
        self.chromosome = chromosome

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

class estimator:
    def __init__(self) -> None:
        self.generation = 0
        self.currentIndividual = 0
        self.populationSize = 10
        self.name = "Genetic"

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return

    def createPopulation(self):
        self.population = []
        ## Get model weights
        weights = self.model.get_weights()

        ## Based on the weights, create a population of 100 individuals
        for i in range(100):
            chromosome = []
            for j in range(len(weights)):
                chromosome.append(np.random.randf(0, 1))
            self.population.append(individual(chromosome))

    def applyWeights(self, chromosome):
        ## Replace the weights of the model with the chromosome
        weights = self.model.get_weights()

        for i in range(len(weights)):
            weights[i] = chromosome[i]

        self.model.set_weights(weights)

        return self.model

    def predict(self, observation):
        return

    def crossover(self, parent1, parent2):
        ## Get the chromosome of each parent
        chromosome1 = parent1.getChromosome()
        chromosome2 = parent2.getChromosome()

        ## Create a new chromosome
        chromosome = []

        ## Loop through each gene in the chromosome
        for i in range(len(chromosome1)):
            ## Randomly select a gene from either parent
            if np.random.random() < 0.5:
                chromosome.append(chromosome1[i])
            else:
                chromosome.append(chromosome2[i])

        return chromosome

    def mutate(self, chromosome):
        ## Try to mutate each gene in the chromosome
        ## The higher the fitness, the less likely it is to mutate
        for i in range(len(chromosome)):
            if np.random.random() < (1 - self.population[i].getFitness()):
                ## Add a random value that does not exceed 1 or -1
                mutation = np.random.uniform(-0.1, 0.1)
                if (chromosome[i] + mutation) > 1:
                    chromosome[i] -= mutation
                elif (chromosome[i] + mutation) < -1:
                    chromosome[i] += mutation

        return chromosome

    def newGeneration(self):
        ## Select the best 10% of the population
        tenPercent = self.populationSize / 10
        self.population = self.population[:tenPercent]

        ## Generate 90 new individuals from the best 10% of the population
        for i in range(0, tenPercent):
            ## Select 2 random individuals from the best 10%
            parent1 = self.population[np.random.randint(0, tenPercent)]
            parent2 = self.population[np.random.randint(0, tenPercent)]

            ## Generate a new individual by crossing over the 2 parents
            child = self.crossover(parent1, parent2)

            ## Add the new individual to the population
            self.population.append(individual(child))

        ## Mutate the population
        for i in range(0, len(self.population)):
            self.population[i].setChromosome(self.mutate(self.population[i].getChromosome()))


    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return
        ## Set the model the first time update is called to get the neural network configuration
        if (not self.model):
            if (not model):
                raise Exception(f"Model not found for {self.name} estimator")

        self.currentIndividual += 1

        ## If all individuals have been tested, create a new generation
        if (self.currentIndividual == self.populationSize):
            self.currentIndividual = 0
            self.generation += 1
            self.newGeneration()
        ## If not, test the next individual
        else:
            self.applyWeights(self.population[self.currentIndividual].getChromosome())
            return

        return self.model;