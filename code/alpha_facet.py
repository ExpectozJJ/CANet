import numpy as np
import gudhi
import itertools
import math
import bisect
import matplotlib.pyplot as plt

class alpha_facet:
    def __init__(self, points, max_dim = 0, min_edge_length = 0.01, max_edge_length = 1.0, num_samples=100):
        self.points = points
        self.max_dim = max_dim
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length
        self.num_samples = num_samples


    def alpha_simplices_birth_death(self):
        """
        Compute birth and death times of all simplices (up to max_dim) in the Vietoris-Rips filtration
        for the given points, using Gudhi for fast complex construction.
        
        Birth time for a simplex is its filtration value (max edge length among its vertices).
        Death time is defined as the minimum filtration value among its cofaces (simplices one dimension higher).
        For maximal simplices (of dimension max_dim), death time is set to infinity.
        
        Intervals where birth equals death are ignored.
        
        Parameters:
            points (ndarray): An array of shape (n, d) with n points in d dimensions.
            max_dim (int): The maximal dimension of simplices to consider.
            max_edge_length (float): The maximum edge length used to build the Rips complex.
            
        Returns:
            barcodes (dict): A dictionary where barcodes[d] is a list of (birth, death) intervals
                            for d-dimensional simplices.
        """
        n = len(self.points)
        
        # Build the Alpha complex using Gudhi.
        alpha_complex = gudhi.AlphaComplex(points=self.points)
        # We build up to dimension max_dim+1 because a simplex with (max_dim+1) vertices is of dimension max_dim.
        simplex_tree = alpha_complex.create_simplex_tree()
        
        # Extract all simplices with their filtration values.
        # Each entry is a tuple (simplex, filtration_value), where simplex is a tuple of vertices.
        simplex_entries = list(simplex_tree.get_simplices())
        
        # Build a dictionary mapping each simplex (as a frozenset) to its filtration (birth) time.
        subset_birth = {}
        for simplex, filt in simplex_entries:
            if (len(simplex) - 1) <= self.max_dim:
                subset_birth[frozenset(simplex)] = filt

        # Compute death times.
        subset_death = {}
        for subset, birth in subset_birth.items():
            k = len(subset)
            # For maximal dimension simplices, death is infinity.
            if k == self.max_dim + 1:
                subset_death[subset] = math.inf
            else:
                candidate_deaths = []
                # Iterate over vertices to form all possible cofaces (simplices with one more vertex).
                for v in range(n):
                    if v not in subset:
                        superset = subset.union({v})
                        # Only consider immediate cofaces (dimension k+1)
                        if len(superset) == k + 1 and superset in subset_birth:
                            candidate_deaths.append(subset_birth[superset])
                subset_death[subset] = min(candidate_deaths) if candidate_deaths else math.inf

        # Build the barcode dictionary for each dimension.
        barcodes = {dim: [] for dim in range(self.max_dim + 1)}
        for subset, birth in subset_birth.items():
            dim = len(subset) - 1
            if dim <= self.max_dim:
                death = subset_death[subset]
                # Ignore intervals where birth equals death.
                if birth != death:
                    barcodes[dim].append((birth, death))
        
        return barcodes
    
    def prepare_intervals_from_barcodes(self, dimension=None):
        """
        Given a barcodes dictionary, extract and sort birth and death times.
        
        Parameters:
            barcodes (dict): Keys are dimensions, values are lists of (birth, death) intervals.
            dimension (int or None): If specified, only use intervals of that dimension.
                                    If None, combine intervals from all dimensions.
        
        Returns:
            births (list): Sorted list of birth times.
            deaths (list): Sorted list of death times.
        """
        barcodes = self.alpha_simplices_birth_death()
        intervals = []
        if dimension is not None:
            intervals.extend(barcodes.get(dimension, []))
        else:
            for d in barcodes:
                intervals.extend(barcodes[d])
        
        births = sorted(b for b, d in intervals)
        deaths = sorted(d for b, d in intervals)
        return births, deaths
    

    def count_active_intervals_sorted(self, births, deaths, t):
        """
        Count active intervals at time t using pre-sorted birth and death lists.
        
        Parameters:
            births (list): Sorted list of birth times.
            deaths (list): Sorted list of death times.
            t (float): The moment in time.
            
        Returns:
            int: The number of active intervals.
        """
        started = bisect.bisect_right(births, t)
        ended = bisect.bisect_right(deaths, t)
        return started - ended

    def compute_active_curve(self, dimension, t_values):
        """
        Compute the active intervals curve for a specific dimension.
        
        Parameters:
            barcodes (dict): Barcode dictionary with dimension keys.
            dimension (int): The dimension for which to compute the curve.
            t_values (array-like): Values of t at which to compute the active intervals.
            
        Returns:
            counts (list): List of active interval counts at each t in t_values.
        """
        births, deaths = self.prepare_intervals_from_barcodes(dimension=dimension)
        counts = [self.count_active_intervals_sorted(births, deaths, t) for t in t_values]
        return counts

    def compute_active_rates(self, dimension, t_values):
        """
        Compute the active intervals curve for a specific dimension.
        
        Parameters:
            barcodes (dict): Barcode dictionary with dimension keys.
            dimension (int): The dimension for which to compute the curve.
            t_values (array-like): Values of t at which to compute the active intervals.
            
        Returns:
            counts (list): List of active interval counts at each t in t_values.
        """
        births, deaths = self.prepare_intervals_from_barcodes(dimension=dimension)
        rates = [self.count_active_intervals_sorted(births, deaths, t)/t for t in t_values if t!= 0]
        return rates

    def facet_curves(self):
        barcodes = self.alpha_simplices_birth_death()
        t_values = np.linspace(self.min_edge_length, self.max_edge_length, self.num_samples)
        curves = [self.compute_active_curve(dimension, t_values) for dimension in range(self.max_dim+1)]
        return curves

    def facet_rates(self):
        barcodes = self.alpha_simplices_birth_death()
        t_values = np.linspace(self.min_edge_length, self.max_edge_length, self.num_samples)
        curves = [self.compute_active_rates(dimension, t_values) for dimension in range(self.max_dim+1)]
        return curves

#num_points = 100
#dim = 3
#points = np.random.rand(num_points, dim)

#model = alpha_facet(points, max_dim = 3, min_edge_length=2, max_edge_length=15, num_samples=100)
#print(np.hstack(model.facet_curves())) # facet persistent betti curves
#print(np.hstack(model.facet_rates())) # facet persistent betti curves average over radius 