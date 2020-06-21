'''
Evolution strategies for evolutionary algorithms
'''

import numpy as np
from abc import ABC, abstractmethod

from metapy.evolutionary.selection import TruncationSelector

class AbstractEvolutionStrategy(ABC):
    @abstractmethod
    def evolve(self, population, fitness):
        pass


class GeneticAlgorithmStrategy(AbstractEvolutionStrategy):
    '''
    The Genetic evolution
    Individual chromosomes in the population
    compete to cross over and breed children.  
    Children are randomly mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, _lambda, selector, xoperator, mutator):
        '''
        Constructor

        Parameters:
        --------

        _lambda -   int, controls the size of each generation. (make it even)

        selector -  AbstractSelector, selects an individual chromosome for crossover

        xoperator - AbstractCrossoverOperator, encapsulates the logic
                    crossover two selected parents
        
        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
 
        self._lambda = _lambda
        self._selector = selector
        self._xoperator = xoperator
        self._mutator = mutator

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        next_gen = np.full((self._lambda, len(population[0])),
                             fill_value=-1, dtype=np.byte)

        index = 0
        for crossover in range(int(self._lambda / 2)):
            
            parent_a = self._selector.select(population, fitness)
            parent_b = self._selector.select(population, fitness)
            
            c_a, c_b = self._xoperator.crossover(parent_a, parent_b)
           
            self._mutator.mutate(c_a)
            self._mutator.mutate(c_b)

            next_gen[index], next_gen[index+1] = c_a, c_b
            
            index += 2
        return next_gen


class ElitistGeneticAlgorithmStrategy(AbstractEvolutionStrategy):
    '''
    The Genetic evolution
    Individual chromosomes in the population
    compete to cross over and breed children.  
    Children are randomly mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, mew, _lambda, selector, xoperator, mutator):
        '''
        Constructor

        Parameters:
        --------

        _lambda -   int, controls the size of each generation. (make it even)

        selector -  AbstractSelector, selects an individual chromosome for crossover

        xoperator - AbstractCrossoverOperator, encapsulates the logic
                    crossover two selected parents
        
        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
 
        self._mew = mew
        self._lambda = _lambda
        self._selector = selector
        self._xoperator = xoperator
        self._mutator = mutator
        self._trunc_selector = TruncationSelector(mew)

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        next_gen = np.full((self._mew + self._lambda, len(population[0])),
                             fill_value=-1, dtype=np.byte)


        #the n fittest chromosomes in the population (breaking ties at random)
        #this is the difference from the standard GA strategy
        fittest = self._trunc_selector.select(population, fitness)
        next_gen[:len(fittest),] = fittest                     

        index = self._mew
        for crossover in range(int(self._lambda / 2)):
            
            parent_a = self._selector.select(population, fitness)
            parent_b = self._selector.select(population, fitness)
            
            c_a, c_b = self._xoperator.crossover(parent_a, parent_b)
           
            self._mutator.mutate(c_a)
            self._mutator.mutate(c_b)

            next_gen[index], next_gen[index+1] = c_a, c_b
            
            index += 2
        return next_gen


class MewLambdaEvolutionStrategy(AbstractEvolutionStrategy):
    '''
    The (mew, lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, mew, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mew -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.


        mutator -   AbstractMutator, encapsulates the logic of mutation for a 
                    selected individual
        '''
        self._mew = mew
        self._lambda = _lambda
        self._selector = TruncationSelector(mew)
        self._mutator = mutator

    
    def evolve(self, population, fitness):
        '''
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(individual))

        fitness --    numpy.array, vector, size lambda, representing the cost of the 
                      tours in population

        Returns:
        --------
        numpy.array - matrix of new generation, size (lambda, len(individual))
        '''

        fittest = self._selector.select(population, fitness)
        population = np.full((self._lambda, fittest[0].shape[0]),
                             fill_value=-1, dtype=np.byte)

        index = 0
        for parent in fittest:
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(parent.copy())
                population[index] = child
                index += 1

        return population
        

class MewPlusLambdaEvolutionStrategy(AbstractEvolutionStrategy):
    '''
    The (mew+lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.  The mew fittest parents compete with 
    their offspring int he new generation.

    The first generation is of size lambda.
    The second generation is of size mew+lambda
    '''
    def __init__(self, mew, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mew -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.

        mutator -   AbstractMutator, encapsulates the logic of mutation for an indiviudal
        '''
        self._mew = mew
        self._lambda = _lambda
        self._mutator = mutator
        self._selector = TruncationSelector(mew)

    
    def evolve(self, population, fitness):
        '''
        Only mew fittest individual survice.
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda+mew, len(tour))

        fitness     -- numpy.array, vector, size lambda, representing the fitness of the 
                       individuals in the population

        Returns:
        --------
        numpy.arrays - matric a new generation of individuals, 
                       size (lambda+mew, len(individual))
        '''

        fittest = self._selector.select(population, fitness)
        
        #this is the difference from (mew, lambda)
        #could also use np.empty - quicker for larger populations...
        population = np.full((self._lambda+self._mew, fittest[0].shape[0]),
                             0, dtype=np.byte)

        population[:len(fittest),] = fittest
    
        index = self._mew
        for parent in range(len(fittest)):
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(fittest[parent].copy())
                population[index] = child
                index += 1

        return population