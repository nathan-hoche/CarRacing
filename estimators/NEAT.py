import neat
from neat.six_util import itervalues, iteritems
import os
import gzip
import pickle
import random


class CompleteExtinctionException(Exception):
    pass

def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return (population, species_set, generation)

class estimator:

    def __init__(self) -> None:

        self.configName = 'estimators/ConfigNeat/NEATKNN'
        # self.configName = 'estimators/ConfigNeat/NEATCNN'
        self.name = self.configName.split('/')[-1]
        self.saveName = f"saves/NEAT_{self.name}"

        self.config: neat.Config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, self.configName)
        self.population: neat.Population = None
        self.saver: neat.Checkpointer = neat.Checkpointer(1, filename_prefix=self.saveName)

        try:
            init_stat = restore_checkpoint(self.saveName)
            self.population = neat.Population(self.config, init_stat)
        except Exception as e:
            self.population = neat.Population(self.config)

        self.population.add_reporter(self.saver)

        self.currentGenomeIndex = 0
        self.generation = 0
        self.populationSize = self.config.pop_size



    def __str__(self) -> str:
        return "NEAT"

    def memorize(self, observation=None, step=None, reward=None, nextObservation=None, check=False):
        """Actualy no need to memorize
        """
        pass

    def finishGeneration(self, pop):
        if pop.config.no_fitness_termination:
            pop.reporters.found_solution(pop.config, pop.generation, pop.best_genome)
        return pop.best_genome

    def newGeneration(self, pop: neat.Population):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        pop.reporters.start_generation(pop.generation)

            # Evaluate all genomes using the user-provided function.

            # Gather and report statistics.
        best = None
        for g in itervalues(pop.population):
            if best is None or g.fitness > best.fitness:
                best = g
        pop.reporters.post_evaluate(pop.config, pop.population, pop.species, best)

        # Track the best genome ever seen.
        if pop.best_genome is None or best.fitness > pop.best_genome.fitness:
            pop.best_genome = best

        if not pop.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = pop.fitness_criterion(g.fitness for g in itervalues(pop.population))
            if fv >= pop.config.fitness_threshold:
                pop.reporters.found_solution(pop.config, pop.generation, best)
                return pop.finishGeneration(pop)

        # Create the next generation from the current generation.
        pop.population = pop.reproduction.reproduce(pop.config, pop.species,
                                                          pop.config.pop_size, pop.generation)

        # Check for complete extinction.
        if not pop.species.species:
            pop.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if pop.config.reset_on_extinction:
                pop.population = pop.reproduction.create_new(pop.config.genome_type,
                                                               pop.config.genome_config,
                                                               pop.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        pop.species.speciate(pop.config, pop.population, pop.generation)

        pop.reporters.end_generation(pop.config, pop.population, pop.species)

        pop.generation += 1

        return self.finishGeneration(pop)

    def update(self, brain:object=None, score=None, check=False):
        """TODO:
        return new individual (model from self.population.population)
        In brain create fromweight that will load the new model
        Using the return score for setup the fitness, when all the population is tested, run the rest of the run function from NEAT lib
        If you stuggeling whith the architecture, look at the Genetic.py

        Args:
            brain (object, optional): _description_. Defaults to None.
            score (_type_, optional): _description_. Defaults to None.
            check (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if check:
            return
        list(iteritems(self.population.population))[self.currentGenomeIndex][1].fitness = score
        self.currentGenomeIndex += 1
        if self.currentGenomeIndex == self.populationSize:
            self.currentGenomeIndex = 0
            self.generation += 1
            os.system(f"rm {self.saveName}*")
            self.newGeneration(self.population)
            os.system(f"mv {self.saveName}* {self.saveName}")
            print(f"Generation {self.generation} done")
        brain.__init__(self.name, list(iteritems(self.population.population))[self.currentGenomeIndex], self.config)
        return None

    def setup(self, brain: object):
        """Load the first population (model from self.population.population)

        Args:
            brain (object): _description_
        """
        brain.__init__(self.name, list(iteritems(self.population.population))[self.currentGenomeIndex], self.config)