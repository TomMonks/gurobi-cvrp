import numpy as np
from abc import ABC, abstractmethod

class AbstractPopulationGenerator(ABC):
    @abstractmethod
    def generate(self, population_size):
        pass

class EvolutionaryAlgorithm(object):
    '''
    Encapsulates a simple Evolutionary algorithm
    with mutation at each generation.
    '''
    def __init__(self, initialiser, objective, _lambda, 
                 strategy, maximisation=True, generations=1000):
        '''
        Parameters:
        ---------
        tour        - np.array, cities to visit
        matrix      - np.array, cost matrix travelling from city i to city j
        _lambda     - int, initial population size
        strategy    - AbstractEvolutionStrategy, evolution stratgy
        maximisation- bool, True if the objective is a maximisation and 
                      False if objective is minimisation (default=True)
        generations - int, maximum number of generations  (default=1000)
        '''
        self._initialiser = initialiser
        self._max_generations = generations
        self._objective = objective
        self._strategy = strategy
        self._lambda = _lambda
        self._best = None
        self._best_fitness = np.inf
        
        if maximisation:
            self._negate = 1
        else:
            self._negate = -1

    def _get_best(self):
        return self._best
    
    def _get_best_fitness(self):
        return self._best_fitness * self._negate

    def solve(self):

        #population = initiation_population(self._lambda, self._tour)
        population = self._initialiser.generate(self._lambda)
        fitness = None
    
        for generation in range(self._max_generations):
            fitness = self._fitness(population)
            
            max_index = np.argmax(fitness)

            if self._best is None or (fitness[max_index] > self._best_fitness):
                self._best = population[max_index]
                self._best_fitness = fitness[max_index]
            
            population = self._strategy.evolve(population, fitness)
            

    
    def _fitness(self, population):
        fitness = np.full(len(population), -1.0, dtype=float)
        for i in range(len(population)):
            
            #specific to the TSP - needs to be encapsulated...
            #fitness[i] = tour_cost(population[i], self._matrix)
            fitness[i] = self._objective.evaluate(population[i])

        return fitness * self._negate
            
    best_solution = property(_get_best)
    best_fitness = property(_get_best_fitness)






            












    


