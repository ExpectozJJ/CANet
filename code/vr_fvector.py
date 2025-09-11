
import numpy as np
import gudhi
import matplotlib.pyplot as plt

class vr_fvector:
    def __init__(self, distance_matrix, min_edge_length = 0.0, max_edge_length = 1.0, max_dim = 0, num_instances = 100):
        self.distance_matrix = distance_matrix
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.num_instances = num_instances

    def compute_f_vector_at_t(self, simplices_with_filtration, t):
        """
        Given a list of (simplex, filtration) pairs, compute the f-vector for the
        subcomplex at time t (i.e. include all simplices with filtration value <= t).
        
        Parameters:
            simplices_with_filtration (list): List of tuples (simplex, filtration_value)
            t (float): The filtration threshold.
            max_dim (int): Maximum simplex dimension to consider.
            
        Returns:
            f_vector (list): List of counts [f0, f1, ..., f_max_dim] at filtration t.
        """
        f_vector = [0] * (self.max_dim + 1)
        for simplex, filtration in simplices_with_filtration:
            if filtration <= t:
                dim = len(simplex) - 1
                if dim <= self.max_dim:
                    f_vector[dim] += 1
        return f_vector
    

    def compute_f_vector_curves(self):
        """
        Compute the f-vector curves for a range of filtration values.
        
        Parameters:
            simplices_with_filtration (list): List of (simplex, filtration) pairs.
            t_values (array-like): The values of t at which to evaluate the subcomplex.
            max_dim (int): Maximum simplex dimension to consider.
            
        Returns:
            f_curves (dict): Dictionary where f_curves[d] is a list of the number of 
                            d-dimensional simplices for each t in t_values.
        """

        # Construct the Vietoris-Rips complex using Gudhi.
        rips_complex = gudhi.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=self.max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dim+1)
        
        # Extract all simplices along with their filtration values.
        simplices_with_filtration = list(simplex_tree.get_simplices())
        
        # # Determine a range for t.
        # finite_filtrations = [filtration for simplex, filtration in simplices_with_filtration 
        #                     if filtration < float('inf')]
        t_min = self.min_edge_length
        # t_max = max(finite_filtrations) if finite_filtrations else self.max_edge_length
        t_max = self.max_edge_length
        t_values = np.linspace(t_min, t_max, self.num_instances)

        f_curves = {d: [] for d in range(self.max_dim + 1)}
        for t in t_values:
            f_vector = self.compute_f_vector_at_t(simplices_with_filtration, t)
            for d in range(self.max_dim + 1):
                f_curves[d].append(f_vector[d])
        return f_curves

    def compute_rate_curves(self):
        """
        Compute rate curves for each dimension using two methods:
        1. Derivative of f(t) using np.gradient.
        2. Cumulative rate: f(t) / t.
        
        Parameters:
            f_curves (dict): Dictionary of f-vector curves per dimension.
            t_values (array-like): t-values corresponding to f_curves.
            
        Returns:
            derivative_rates (dict): Approximate derivatives (instantaneous rate).
            cumulative_rates (dict): Cumulative rates, f(t)/t (with t=0 handled as 0).
        """
        derivative_rates = {}
        cumulative_rates = {}
        f_curves = self.compute_f_vector_curves()

        t_min = self.min_edge_length
        t_max = self.max_edge_length
        t_values = np.linspace(t_min, t_max, self.num_instances)

        for d, f_vals in f_curves.items():
            f_vals = np.array(f_vals)
            # Compute derivative using finite differences.
            deriv = np.gradient(f_vals, t_values)
            derivative_rates[d] = deriv
            
            # Compute cumulative rate f(t)/t (handling division by zero)
            cum_rate = np.zeros_like(f_vals)
            nonzero = t_values > 0
            cum_rate[nonzero] = f_vals[nonzero] / t_values[nonzero]
            cumulative_rates[d] = cum_rate
        # return derivative_rates, cumulative_rates
        return cumulative_rates

#num_points = 100
#dim = 2
#points = np.random.rand(num_points, dim)

#fmodel = vr_fvector(points, min_edge_length = 0.01, max_edge_length = 10.0, max_dim = 1, num_instances = 100)
#print(np.hstack(list(fmodel.compute_f_vector_curves().values()))) # the f-vector curves components stacked
#print(np.hstack(list(fmodel.compute_rate_curves().values()))) # the average of the radius f/t