# -*- coding: utf-8 -*-
"""
Provides euclidean functions for working with TSP data.

One key function converts a numpy array of 2D coordinates into a matrix of euclidean distances
between them.
"""

import numpy as np

def gen_matrix(cities):
    """
    Creates a numpy array of euclidian distances between 2 sets of
    cities
    
    @points = numpy arrange of coordinate pairs
    
    """
    size = len(cities)
    matrix = np.zeros(shape=(size, size))
    
    row = 0
    col = 0
    
    for city1 in cities:
        col = 0
        for city2 in cities:
            matrix[row, col] = euclidean_distance(city1, city2)
            col+=1
        row +=1
        
    return matrix


def euclidean_distance(city1, city2):
    """
    Calculate euc distance between 2 cities
    5.5 ms to execute
    """
    return np.linalg.norm((city1-city2))



def euclidean_distance2(city1, city2):
    """
    An alterantive way to calculate euc distance 
    between two cities
    
    6-7 ms to execute
    """
    v = np.square(city1 - city2)
    return np.sqrt(np.sum(v))