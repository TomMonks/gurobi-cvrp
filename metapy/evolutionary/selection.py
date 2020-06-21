'''
Selection functions for evolutionary algorithms
'''

import numpy as np
from abc import ABC, abstractmethod

class AbstractSelector(ABC):
    @abstractmethod
    def select(self, population, fitness):
        pass


class TruncationSelector(AbstractSelector):
    '''
    Simple truncation selection of the mew fittest 
    individuals in the population
    '''
    def __init__(self, mew):
        self._mew = mew

    def select(self, population, fitness):
        fittest_indexes = np.argpartition(fitness, fitness.size - self._mew)[-self._mew:]
        return population[fittest_indexes]

class TournamentSelector(AbstractSelector):
    '''
    Encapsulates a popular GA selection algorithm called
    Tournament Selection.  An individual is selected at random
    (with replacement) as the best from the population and competes against
    a randomly selected (with replacement) challenger.  If the individual is
    victorious they compete in the next round.  If the challenger is successful
    they become the best and are carried forward to the next round. This is repeated
    for t rounds.  Higher values of t are more selective.  
    '''
    def __init__(self, tournament_size=2):
        '''
        Constructor

        Parameters:
        ---------
        tournament_size, int, must be >=1, (default=2)
        '''
        if tournament_size < 1:
            raise ValueError('tournamant size must int be >= 1')
        
        self._tournament_size = tournament_size
        
    def select(self, population, fitness):
        '''
        Select individual from population for breeding using
        a tournament approach.  t tournaments are conducted.

        Parameters:
        ---------
        population -    numpy.array.  Matrix of chromosomes 
        fitness -       numpy.array, vector of floats representing the
                        fitness of individual chromosomes

        Returns:
        --------
        numpy.array, vector (1D array) representing the chromosome
        that won the tournament.

        '''

        tournament_participants = np.random.randint(0, population.shape[0], 
                                                    size=self._tournament_size)
        best = population[np.argmax(fitness[tournament_participants])]

        return best