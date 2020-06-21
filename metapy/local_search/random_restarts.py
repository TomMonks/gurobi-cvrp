# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:59:46 2017

@author: tm3y13
"""

from abc import ABC, abstractmethod
import numpy as np

#from metapy.tsp.init_solutions import random_tour

           
class ILSPertubation(ABC):
    @abstractmethod
    def perturb(self, tour):
        pass


class ILSHomeBaseAcceptanceLogic(ABC):
    @abstractmethod
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        pass


class EpsilonGreedyHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base of 
    if sampled u > epsilon otherwise explore.
    '''
    def __init__(self, epsilon=0.2, exploit=None, explore=None):
        self.epsilon = epsilon
        if exploit is None:
            self.exploit = HigherQualityHomeBase()
        else:
            self.exploit = exploit

        if explore is None:
            self.explore = RandomHomeBase()
        else:
            self.explore = explore

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        
        u = np.random.rand()

        if u > self.epsilon:
            return self.exploit.new_home_base(home_base, 
                                              home_cost, 
                                              candidate, 
                                              candidate_cost)
        else:
            return self.explore.new_home_base(home_base, 
                                              home_cost, 
                                              candidate, 
                                              candidate_cost)

        

class HigherQualityHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base
    '''

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        if candidate_cost > home_cost:
            return candidate, candidate_cost
        else:
            return home_base, home_cost

class RandomHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Random walk homebase
    '''
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        return candidate, candidate_cost

class DoubleBridgePertubation(ILSPertubation):
    """
    Perform a random 4-opt ("double bridge") move on a tour.
        
         E.g.
        
            A--B             A  B
           /    \           /|  | \
          H      C         H------C
          |      |   -->     |  |
          G      D         G------D
           \    /           \|  |/
            F--E             F  E
        
        Where edges AB, CD, EF and GH are chosen randomly.

    """
    def perturb(self, tour):
        '''
        Perform a random 4-opt ("double bridge") move on a tour.
        
        Returns:
        --------
        numpy.array, vector. representing the tour

        Parameters:
        --------
        tour - numpy.array, vector representing tour between cities e.g.
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        '''

        n = len(tour)
        #end = int(n/4)+1
        end = int(n/3)+1
        pos1 = np.random.randint(1, end) 
        pos2 = pos1 + np.random.randint(1, end) 
        pos3 = pos2 + np.random.randint(1, end) 

        #print(tour[:pos1], tour[pos1:pos2], tour[pos2:pos3], tour[pos3:])

        p1 = np.concatenate((tour[:pos1] , tour[pos3:]), axis=None)
        p2 = np.concatenate((tour[pos2:pos3] , tour[pos1:pos2]), axis=None)
        #this bit will need updating if switching to alternative operation.
        #should i be adding the first city again?
        #return np.concatenate((p1, p2, p1[0]), axis=None)
        return np.concatenate((p1, p2), axis=None)




class IteratedLocalSearch(object):
    '''
    Iterated Local Search (ILS) Meta Heuristic
    '''
    def __init__(self, objective, local_search, accept=None, 
                 perturb=None, maximisation=True, random_state=None):
        """
        Constructor Method

        Note minimisation implemented via
        negating the cost.

        Parameters:
        --------
        local_search: object
            hill climbing solver or similar

        accept: object
            logic for accepting or rejecting 
            a new homebase

        perturb: ILSPertubation, 
            logic for pertubation 
            from the local optimimum in each iteration

        maximisation: bool optional (default=True)
            problem is a hill climbing (True) or decent (False)

        """
        self._objective = objective
        self._local_search = local_search
        if accept == None:
            self._accepter = RandomHomeBase()
        else:
            self._accepter = accept

        if perturb == None:
            self._perturber = DoubleBridgePertubation()
        else:
            self._perturber = perturb

        if maximisation:
            self._negate = 1
        elif not maximisation:
            self._negate = -1
        else:
            raise ValueError('maximisation must be of type bool (True|False)')
            
        self._solutions = []
        self._best_cost = np.inf * self._negate
        
        np.random.seed(random_state)
        
    def run(self, n):
        """
        Re-run solver n times using a different initial solution
        each time.  Init solution is generated randomly each time.

        The potential power of iteratedl ocal search lies in 
        its biased sampling of the set of local optima.

        """
        current = self._local_search.solution
        #np.random.shuffle(current) # random tour
        
        home_base = current
        home_base_cost = self._objective.evaluate(current) * self._negate
        self._best_cost = home_base_cost 
        self._solutions.append(current)
                
        for _ in range(n):

            #Hill climb from new starting point
            self._local_search.set_init_solution(current)
            self._local_search.solve()
            current = self._local_search.best_solutions[0]

            #will need to refactor 2Opt search from decent to ascent....maybe...
            iteration_best_cost = self._local_search.best_cost * self._negate

            if iteration_best_cost > self._best_cost:
                self._best_cost = iteration_best_cost
                self._solutions = self._local_search.best_solutions

            elif iteration_best_cost == self._best_cost:
                self._solutions.append(self._local_search.best_solutions[0])
                #[self._solutions.append(i) for i in self._local_search.best_solutions]

            home_base, home_base_cost = self._accepter.new_home_base(home_base, 
                                                                     home_base_cost, 
                                                                     current, 
                                                                     iteration_best_cost)
            current = self._perturber.perturb(home_base)
            
    def get_best_solutions(self):
        return self._best_cost * self._negate, self._solutions

  
if __name__ == '__main__':
    tour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    d = DoubleBridgePertubation()
    new_tour = d.perturb(tour) 
    print(new_tour)           