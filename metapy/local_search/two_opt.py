# -*- coding: utf-8 -*-
"""
local search implemented with 2-opt swap

2-opt = Switch 2 edges

@author: Tom Monks
"""
import numpy as np

class OrdinaryDecentTwoOpt(object):
    """
    Local (neighbourhood) search implemented as first improvement 
    with 2-opt swaps
    """   
    def __init__(self, objective, init_solution):
        """
        Constructor Method
        
        Parameters:
        ------------
        objective: object
            objective function 

        init_solution - array-like
            initial tour e.g. [1, 2, 3, 4, 5, 6, 7]
        """
        self._objective = objective
        self.set_init_solution(init_solution)
        
    def set_init_solution(self, solution):  
        self.solution = np.asarray(solution)
        self.best_solutions = [solution]
        self.best_cost = self._objective.evaluate(self.solution)        
    
    def solve(self):
        """
        Run solution algoritm.
        Note: algorithm is the same as ordinary decent
        where 2 customers are swapped apare from call to swap 
        code.  Can I encapsulate the swap code so that it can be reused?
        """
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                #print("city1: {0}".format(city1))
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    #print("city2: {0}".format(city2))
                    
                    self.reverse_sublist(self.solution, city1, city2)
                    
                    new_cost = self._objective.evaluate(self.solution)
                    #if (new_cost == self.best_cost):
                        #self.best_solutions.append(self.solution)
                        #improvement = True
                    if (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.reverse_sublist(self.solution, city1, city2)
                        
                      
    def reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end

        Parameters:
        --------
        lst: np.array, 
            vector representing a solution
        
        start: int, 
            start index of sublist (inclusive)

        end:int, 
            end index of sublist (inclusive)

        """
        lst[start:end] = lst[start:end][::-1]