# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:57:46 2017

@author: tm3y13
"""

from random import shuffle
import numpy as np
from metapy.evolutionary.evolutionary import AbstractPopulationGenerator

def random_tour(tour):
    """
    Initial solution to tour is psuedo random
    """
    rnd_tour = tour[1:len(tour)-1]
    base_city = tour[0]
    shuffle(rnd_tour)
    rnd_tour.append(base_city)
    rnd_tour.insert(0, base_city)
    return rnd_tour


class TSPPopulationGenerator(AbstractPopulationGenerator):
    def __init__(self, cities_in_tour):
        self._cities_in_tour = cities_in_tour

    def generate(self, population_size):
        '''
        Generate a list of @population_size tours.  Tours
        are randomly generated and unique to maximise
        diversity of the population.

        Parameters:
        ---------
        population_size -- the size of the population

        Returns:
        ---------
        np.array. matrix size = (population_size, len(tour)). Contains
                the initial generation of tours
        '''

        population = {}

        #return data as
        population_arr = np.full((population_size, len(self._cities_in_tour)), 
            -1, dtype=np.byte)

        i = 0
        while i < population_size:
            #sample a permutation
            new_tour = np.random.permutation(self._cities_in_tour)
            
            #check its unique to maximise diversity
            if str(new_tour) not in population:
                population[str(new_tour)] = new_tour
                i = i + 1

        #save unique permutation
        population_arr[:,] = list(population.values())

        return population_arr
    
    
class VRPPopulationGenerator(AbstractPopulationGenerator):
    def __init__(self, cities_in_tour):
        self._cities_in_tour = cities_in_tour

    def generate(self, population_size):
        '''
        Generate a list of @population_size tours.  Tours
        are randomly generated and unique to maximise
        diversity of the population.

        Parameters:
        ---------
        population_size -- the size of the population

        Returns:
        ---------
        np.array. matrix size = (population_size, len(tour)). Contains
                the initial generation of tours
        '''

        population = {}

        #return data as
        population_arr = np.full((population_size, len(self._cities_in_tour)), 
            -1, dtype=np.byte)

        i = 0
        while i < population_size:
            #sample a permutation
            new_tour = np.random.permutation(self._cities_in_tour[1:])
            new_tour = np.concatenate([np.array([0]), new_tour])
            
            #check its unique to maximise diversity
            if str(new_tour) not in population:
                population[str(new_tour)] = new_tour
                i = i + 1

        #save unique permutation
        population_arr[:,] = list(population.values())

        return population_arr


def initiation_population(population_size, tour):
    '''
    Generate a list of @population_size tours.  Tours
    are randomly generated and unique to maximise
    diversity of the population.

    Parameters:
    ---------
    population_size -- the size of the population

    Returns:
    ---------
    np.array. matrix size = (population_size, len(tour)). Contains
              the initial generation of tours
    '''

    population = {}
    #for i in range(population_size):
    i = 0
    while i < population_size:
        #some code is legacy and uses python
        #lists instead of numpy arrays... to fix!
        #the random tour bit varies between problem...
        #new_tour = random_tour(tour)
        new_tour = tour.copy()
        np.random.shuffle(new_tour)
        
        #return data as
        population_arr = np.full((population_size, len(tour)), -1, dtype=np.byte)

        if str(new_tour) not in population:
            population[str(new_tour)] = np.array(new_tour)
            i = i + 1

    population_arr[:,] = list(population.values())

    return population_arr
    