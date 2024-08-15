import time
import numpy as np
import random
from itertools import product
import decimal
import utils

import pulp as plp
import time


class STPN:
    intervals = []

    def __init__(self, intervals):
        self.intervals = intervals
    
    @classmethod
    def custom(cls, intervals):
        return cls(utils.valid_intervals(intervals))
    
    @classmethod
    def random(cls, dimension, left, right):
        intervals = cls.random_intervals(dimension,left, right)
        return cls(intervals)
    
    @classmethod
    def random_example(cls, dimension, left, right, cardinality, random_intervals = True):
        intervals = cls.random_intervals(dimension,left, right, random_intervals)
        stpn = cls(intervals)
        log = stpn.random_log(cardinality)
        return stpn, log

    @staticmethod
    def random_intervals(dimension, left=0, right=100, random_intervals= True):
        intervals = []
        for _ in range(dimension):
            if random_intervals:
              start = round(random.uniform(left, right),2)
              end = round(random.uniform(start+left, start + right),2)
            else:
               start = round(left,2)
               end = round(start + right,2)
 
            intervals.append((start, end))
        return utils.valid_intervals(intervals)

    
    def __repr__(self):
        return f"STPN contraints: {self.intervals}"
    
    def dim(self):
        return len(self.intervals)
    
    def min_point(self):
        min_point = 0
        result = []
        
        for interval in self.intervals:
            min_point += interval[0]
            result.append(min_point)

        return result
    
    def max_point(self):
        max_point = 0
        result = []
        
        for interval in self.intervals:
            max_point += interval[1]
            result.append(max_point)
        
        return result

    def random_log(self, cardinality):
        log = []
        
        for _ in range(cardinality):
            trace = []
            prev_value = 0
            for interval in self.intervals:
                random_value = random.uniform(interval[0], interval[1])
                prev_value += random_value
                trace.append(round(prev_value, 2))
            
            log.append(trace)
        
        return log

    def is_accepted_log(self, log):
        log = utils.valid_log(log)
        for trace in log:
            if not self.is_accepted_trace(trace):
                raise ValueError(f"Trace {trace} is not accepted by the STPN.")
        return log
    
    def is_accepted_trace(self, trace):
        if len(trace) != self.dim():
            return False
        
        prev_val= 0
        for i, interval in enumerate(self.intervals):
            prev_val = 0 if i == 0 else trace[i - 1]
            if not round(interval[0],2) <= round(trace[i] - prev_val,2) <= round(interval[1],2):
                return False            
        return True
    
    def search_space(self, distance_type = 0, n_div = 5):
        if distance_type == 0:
            return self.search_space_stamponly(1,0, [],[], n_div)
        elif distance_type == 1:
            return self.search_space_delayonly(n_div)

    def search_space_stamponly(self, counter, brought, gamma, search_space, step):
        if counter-1 == self.dim():
            # Base case: all loops have been executed
            search_space.append(gamma.copy())
            #print(gamma, search_space)
            return search_space
        else:
            lower = self.intervals[counter-1][0] + brought
            upper = self.intervals[counter-1][1] + brought
            for i in  np.linspace(lower, upper, step):
                gamma.append(i)  # Modify the values for the current loop
                self.search_space_stamponly(counter + 1, i , gamma, search_space, step)  #Recursively call the function for the next loop
                gamma.pop()  # Remove the current loop's value

        if counter == 1:
            print('Stamp Only BF search space built')
            return search_space
    
    def search_space_delayonly(self, n_div):
        #builds the search space when using the delay only distance
        #(discretized, still with the parameter n_div determining the number of splits of the space of each dimension into equidistant points)

        # Create a list of linspaces for each interval
        linspaces = [np.linspace(interval[0], interval[1], n_div) for interval in self.intervals]

        # Generate all combinations of points using itertools.product
        combinations = set(product(*linspaces))

        return combinations
        
    def transform_log(self, log):
        # Create a new list to store the transformed log
        transformed_log = []

        # Iterate over each trace in the original log
        for trace in log:
            if self.dim() == 1:
                trace_list = [trace]
            else:
                trace_list = list(trace)

            old_trace = trace_list.copy()

            # Calculate deltas and ensure they are rounded to two decimal places
            for j in range(1, len(trace_list)):
                # Calculate the difference and round to two decimal places
                trace_list[j] = round(trace_list[j] - old_trace[j - 1], 2)

            # Add the transformed trace to the new list
            transformed_log.append(trace_list)

        # Return the new transformed log
        return transformed_log

    def solver_bf(self, log, distance_type = 0, n_div = 5):
        #finds all possible anti-alignments considering the discretized search space, and comparing every point brute force

        search_space = self.search_space(distance_type, n_div)
        
        log = self.is_accepted_log(log)

        if distance_type == 1:
            log = self.transform_log(log)
        
        # Initialize variables
        max_distance = float('-inf')
        max_points = []

        # Convert the list L to a numpy array for efficient vector operations
        L_array = np.array(log)

        # Iterate over each point in the search space
        for gamma in search_space:
            # Convert gamma to a numpy array for efficient vector operations
            gamma_array = np.array(gamma)

            # Calculate the minimum distance for the current gamma
            distances = np.linalg.norm(L_array - gamma_array, ord=1, axis=1)  # Efficient L1 norm computation
            min_distance = np.min(distances)  # Minimum distance to any point in L

            # Update max_distance and max_points based on the minimum distance
            if min_distance > max_distance:
                max_distance = min_distance
                max_points = [gamma]
            elif min_distance == max_distance:
                max_points.append(gamma)

        return max_distance, max_points

    def LPSolver(self, log, distance_type = 0):
        #Linear Programming solver with constraints and variables as described in the paper

        # Check if the log is accepted by the STPN
        log = self.is_accepted_log(log)
        
        # Define the dimensionality of the problem  
        d = self.dim() #dimension of the space
        n = len(log) #number of points in L

        model = self.intervals

        if distance_type == 1: #if delay-only distance is used, the log is transformed in equivalent flow functions traces
            start =  time.time()
            log = self.transform_log(log)
            elapsed_time = time.time() - start
            print(elapsed_time)

        # Create the LP problem
        prob = plp.LpProblem("Maximize minimal manhattan distance", plp.LpMaximize)

        # Define the decision variables
        x = plp.LpVariable.dicts("x", range(d), lowBound=0, cat = 'Continuous')

        #Define the variable to maximize
        z = plp.LpVariable("z", lowBound=0, cat = 'Continuous')

        # Define the constraints for x, i.e. the search space constraints
        if distance_type == 0:
            for i in range(d):
                if i == 0:
                    prob += x[i] >= model[i][0]
                    prob += x[i] <= model[i][1]
                else:
                    prob += x[i] >= model[i][0] + x[i-1]
                    prob += x[i] <= model[i][1] + x[i-1]
        else:
            for i in range(d):
                prob += x[i] >= model[i][0]
                prob += x[i] <= model[i][1]


        # Define the constraints for the absolute value of the difference between x and each point in L
        diff_plus = plp.LpVariable.dicts("diff_plus", (range(n), range(d)), lowBound=0, cat = 'Continuous')
        diff_minus = plp.LpVariable.dicts("diff_minus", (range(n), range(d)), lowBound=0, cat = 'Continuous')

        M = np.linalg.norm(np.array(self.max_point()) - np.array(self.min_point()), ord=1)
        print("M:",M)

        #binary variables
        b = plp.LpVariable.dicts("b", (range(n), range(d)), cat = 'Binary')

        for j in range(n): #for every sigma in L
            for i in range(d): #for every dimension
                #print(i)
                if len(model) == 1: #if the model is 1-dimensional
                    prob += diff_plus[j][i] - diff_minus[j][i] == log[j] - x[i]
                else:
                    prob += diff_plus[j][i] - diff_minus[j][i] == log[j][i] - x[i]

                prob += diff_plus[j][i] <= M*b[j][i]
                prob += diff_minus[j][i] <= M*(1-b[j][i])

            prob += z <= plp.lpSum([diff_plus[j][i] + diff_minus[j][i] for i in range(d)])

        #print(prob)

        # Define the objective function
        prob += z

        # Solve the problem
        prob.solve(plp.PULP_CBC_CMD(timeLimit =21600))

        # Print the results
        print("Status:", plp.LpStatus[prob.status])
        #print("Max Distance:", plp.value(prob.objective))

        optimal_point = [plp.value(x[i]) for i in range(d)]
        #print('Optimal Point:', optimal_point)

        return plp.value(prob.objective), optimal_point, prob.numVariables(), prob.numConstraints()

