'''
Cross over operators for evolutionary algorithms
'''
import numpy as np
from abc import ABC, abstractmethod

class AbstractCrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent_a, parent_b):
        pass

class PartiallyMappedCrossover(AbstractCrossoverOperator):
    '''
    Partially Mapped Crossover operator
    '''
    def __init__(self):
        pass

    def crossover(self, parent_a, parent_b):
    
        child_a = self._pmx(parent_a.copy(), parent_b)
        child_b = self._pmx(parent_b.copy(), parent_a)

        return child_a, child_b

    def _pmx(self, child, parent_to_cross):
        x_indexes = np.sort(np.random.randint(0, len(child), size=2))
        
        for index in range(x_indexes[0], x_indexes[1]):
            city = parent_to_cross[index]
            swap_index = np.where(child == city)[0][0]
            child[index], child[swap_index] = child[swap_index], child[index]

        return child

