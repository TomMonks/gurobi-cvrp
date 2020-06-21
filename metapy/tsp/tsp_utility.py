# -*- coding: utf-8 -*-
"""
Useful but misc utility functions for solving TSP problems.
"""

def trim_base(full_tour):
    """
    Trims start/end city from tour and returns the trimmed
    tour and the base city.
    
    E.g. Passed a full symmetric tour [0, 1, 2, 0] 
    trim_base returns two objects:
        trimmed_tour = [1, 2]
        base_city 0
    """
    base_city = full_tour[0]
    trimmed_tour = full_tour[1:len(full_tour)-1]
    return trimmed_tour, base_city


def append_base(trimmed_tour, base_city):
    """
    Appends a common start/end to a tour
    E.g. when passed
    trimmed_tour = [1,2]
    base_city = 0
    The func returns [0, 1, 2, 0]
    """
    trimmed_tour.append(base_city)
    trimmed_tour.insert(0, base_city)
    return trimmed_tour

