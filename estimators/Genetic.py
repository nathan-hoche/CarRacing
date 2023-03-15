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
        self.populationSize = 50
        self.bestScore = -1
        self.name = "Genetic"
        self.population = []
        self.model = None

    def __str__(self) -> str:
        return self.name

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        if check:
            return

    def setup(self, brain: object):
        weights = brain.getAllWeights()
        self.createPopulation(weights)
        self.newGeneration()
        self.model = weights

    def createPopulation(self, weights: object):
        self.population = []

        print(weights)
        ## Based on the weights, create a population
        for i in range(self.populationSize):
            chromosome = np.array([])
            for layer in weights.keys():
                for type in weights[layer].keys():
                    chromosome = np.append(chromosome, weights[layer][type].reshape(weights[layer][type].size))

            self.population.append(individual(chromosome))

    def applyWeights(self, chromosome):
        weights = {}
        weightIndex = 0
        for layer in self.model.keys():
            weights[layer] = {}
            for type in self.model[layer].keys():
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

        ratio = np.random.randint(0, chromosome1.size)

        ## Merge the 2 chromosomes randomly
        if (np.random.random() < 0.5):
            chromosome = np.append(chromosome, chromosome1[:ratio])
            chromosome = np.append(chromosome, chromosome2[ratio:])
        else:
            chromosome = np.append(chromosome, chromosome2[:ratio])
            chromosome = np.append(chromosome, chromosome1[ratio:])

        return chromosome

    def mutate(self, chromosome, probability):
        ## Try to mutate each gene in the chromosome
        ## The higher the fitness, the less likely it is to mutate
        for i in range(chromosome.size):
            if (np.random.random() < probability):
                ## Add and prevent overflow
                chromosome[i] += np.random.uniform(-0.1, 0.1)
                if (chromosome[i] > 1):
                    chromosome[i] = 1
                elif (chromosome[i] < -1):
                    chromosome[i] = -1

        return chromosome

    def newGeneration(self):
        print(f"Generation: {self.generation + 1}")
        ## Select the best 10% of the population
        self.population = sorted(self.population, key=lambda x: x.getFitness(), reverse=True)
        ## Print each individual's fitness
        for i in range(len(self.population)):
            print(f"{i}: {self.population[i].getFitness()}")
        tenPercent = int(self.populationSize / 10)
        if (tenPercent == 0 or tenPercent == 1):
            tenPercent = 2

        self.population = self.population[:tenPercent]

        sys.stdout.write("Cross-over:\t [%s]" % (" " * (len(self.population) - 1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(self.population)))
        ## Generate 90% new individuals from the best 10% of the population
        for i in range(0, self.populationSize - tenPercent):
            ## Select 2 random individuals from the best 10%
            parent1 = self.population[np.random.randint(0, tenPercent)]
            parent2 = self.population[np.random.randint(0, tenPercent)]

            ## Generate a new individual by crossing over the 2 parents
            child = self.crossover(parent1, parent2)

            ## Add the new individual to the population
            self.population.append(individual(child))
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")

        ## Mutate the population
        sys.stdout.write("Mutation:\t [%s]" % (" " * (len(self.population) - 1)))
        sys.stdout.flush()
        sys.stdout.write("\b" * (len(self.population)))

        ## Mutate each individual in the population based on its fitness (the higher the fitness, the less likely it is to mutate)
        ## Except for the best individual
        for i in range(1, self.populationSize):
            self.population[i].setChromosome(self.mutate(self.population[i].getChromosome(), (i + 1) / self.populationSize))
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")



    def update(self, brain:object=None, score=None, model=None, check=False):
        if check:
            return

        ## On the first update, create the population based on the model
        if (not self.population):
            self.createPopulation(model)

        ## Update the current individual
        print(f"Individual: {self.currentIndividual}")
        if (not score):
            raise Exception(f"Score not found for {self.name} estimator")
        self.population[self.currentIndividual].setFitness(score)
        if (score > self.bestScore):
            self.bestScore = score
            brain.save(score)

        ## If all individuals have been tested, create a new generation
        if (self.currentIndividual == self.populationSize - 1):
            self.currentIndividual = 1
            self.generation += 1
            self.newGeneration()
            self.applyWeights(self.population[self.currentIndividual].getChromosome())
            return self.model

        ## Test the next individual
        self.currentIndividual += 1
        self.applyWeights(self.population[self.currentIndividual].getChromosome())
        return self.model