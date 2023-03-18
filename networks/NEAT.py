import neat

def eval_genome(genome, config):
    # for genome_id, genome in genomes:
    net = neat.nn.FeedForwardNetwork.create(genome, config)

def eval_fitness(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)



class brain:
    def __init__(self, estimatorName):
        self.name = "NEAT"
        self.config: neat.Config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'estimators/Neat/configNEAT')
        self.population: neat.Population = neat.Population(self.config)
        self.saver: neat.Checkpointer = neat.Checkpointer()

        self.bestGenome = None

        self.saveName = f"save/{self.name}_{estimatorName}.pkl"

    def __str__(self) -> str:
        pass

    def save(self, score):
        self.saver.save_checkpoint(config=self.config, population=self.population, species_set=self.population.species, generation=self.population.generation, filename=self.saveName)


    def predict(self, observation = None, check=False):
        self.population.run(eval_fitness, 1)
        pass


    def train(self, weights:dict=None, check=False):
        pass