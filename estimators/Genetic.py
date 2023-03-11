import numpy as np
import json
import sys

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

def formatWeights(weights:dict) -> list:
    return [weights["weight"], weights["bias"]]

############# CLASS #############

class individual:
    def __init__(self, chromosome: list) -> None:
        self.genes = len(chromosome)
        self.chromosome = chromosome
        self.fitness = 0

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
        self.bestScore = -1
        self.name = "Genetic"
        self.population = []
        self.model = None

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return

    def setup(self, weights: object):
        self.createPopulation(weights)
        self.model = weights

    def createPopulation(self, weights: object):
        self.population = []

        print(weights)
        ## Based on the weights, create a population
        for _ in range(self.populationSize):
            chromosome = np.array([])
            for layer in weights.keys():
                for type in weights[layer].keys():
                    chromosome = np.append(np.random.uniform(-1, 1, weights[layer][type].size), chromosome)

            print(f"chromosome: {len(chromosome)}")
            self.population.append(individual(chromosome))

    def applyWeights(self, chromosome):
        weights = {}
        weightIndex = 0
        for layer in self.model.keys():
            weights[layer] = {}
            for type in self.model[layer].keys():
                print(type)
                typeSize = self.model[layer][type].size
                weights[layer][type] = chromosome[weightIndex:weightIndex + typeSize].reshape(self.model[layer][type].shape)
                weightIndex += typeSize

        self.model = weights

    def crossover(self, parent1, parent2):
        ## Get the chromosome of each parent
        chromosome1 = parent1.getChromosome()
        chromosome2 = parent2.getChromosome()

        ## Create a new chromosome
        chromosome = np.array([])

        ## Loop through each gene in the chromosome
        for i in range(chromosome1.size):
            ## Randomly select a gene from either parent
            if np.random.random() < 0.5:
                chromosome = np.append(chromosome, chromosome1[i])
            else:
                chromosome = np.append(chromosome, chromosome2[i])

        return chromosome

    def mutate(self, chromosome, probability=0.5):
        ## Try to mutate each gene in the chromosome
        ## The higher the fitness, the less likely it is to mutate
        for i in range(len(chromosome)):
            ## Random between 0 and 1
            if (np.random.random() < probability):
                if (np.random.random() < 0.5):
                    ## Add a random value that does not exceed 1 or -1
                    mutation = np.random.uniform(-0.1, 0.1)
                    if (chromosome[i] + mutation) > 1:
                        chromosome[i] -= mutation
                    elif (chromosome[i] + mutation) < -1:
                        chromosome[i] += mutation

        return chromosome

    def newGeneration(self):
        print(f"Generation: {self.generation}")
        ## Select the best 10% of the population
        tenPercent = int(self.populationSize / 10)
        if (tenPercent == 0 or tenPercent == 1):
            tenPercent = 2

        self.population = self.population[:tenPercent]

        ## Generate 90% new individuals from the best 10% of the population
        for i in range(0, self.populationSize - tenPercent):
            ## Select 2 random individuals from the best 10%
            parent1 = self.population[np.random.randint(0, tenPercent)]
            parent2 = self.population[np.random.randint(0, tenPercent)]

            ## Generate a new individual by crossing over the 2 parents
            child = self.crossover(parent1, parent2)

            ## Add the new individual to the population
            self.population.append(individual(child))

        ## Mutate the population
        sys.stdout.write("Mutation: [%s]" % (" " * len(self.population) - 1))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(self.population)))

        for i in range(1, len(self.population)):
            self.population[i].setChromosome(self.mutate(self.population[i].getChromosome(), i / len(self.population)))
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")



    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return

        ## On the first update, create the population based on the model
        if (not self.population):
            self.createPopulation(model)

        ## If all individuals have been tested, create a new generation
        if (self.currentIndividual == self.populationSize):
            self.currentIndividual = 0
            self.generation += 1
            self.newGeneration()
        ## If not, test the next individual
        else:
            print(f"Individual: {self.currentIndividual}")
            if (not score):
                raise Exception(f"Score not found for {self.name} estimator")
            self.population[self.currentIndividual].setFitness(score)
            self.applyWeights(self.population[self.currentIndividual].getChromosome())
            if (score > self.bestScore):
                self.bestScore = score
                brain.save(score)

        ## Update the current individual
        self.currentIndividual += 1
        return self.model