'''
Mutation operators for Evolutionary Algorithms
'''
import numpy as np
from abc import ABC, abstractmethod

class AbstractMutator(ABC):
    @abstractmethod
    def mutate(self, individual):
        pass

class TwoCityMutator(AbstractMutator):
    '''
    Mutates an individual tour by
    randomly swapping two cities.

    Designed to work with the TSP.
    '''
    def mutate(self, tour):
        '''
        Randomly swap two cities
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(0, len(tour) - 1, 2)

        tour[to_swap[0]], tour[to_swap[1]] = \
            tour[to_swap[1]], tour[to_swap[0]]

        return tour


class TwoOptMutator(AbstractMutator):
    '''
    Mutates an individual tour by
    randomly swapping two cities.

    Designed to work with the TSP
    '''
    def mutate(self, tour):
        '''
        Randomly reverse a section of the route
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(0, len(tour) - 1, 2)

        if to_swap[1] < to_swap[0]:
            to_swap[0], to_swap[1] = to_swap[1], to_swap[0]

        return self._reverse_sublist(tour, to_swap[0], to_swap[1])


    def _reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end] = lst[start:end][::-1]
        return lst