import time
import numpy as np
import random
from itertools import product
import decimal
import re
import utils
import math


import pulp as plp
from pulp import PULP_CBC_CMD
import time

class ATMG:

    transitions = []
    places = []
    arcs = []

    ordered_trns = []
    
    def __init__(self, arcs, intervals):

        self.transitions = []
        self.places = set()
        self.arcs = arcs

        self.mapping = dict()
        self.ordered_trns = []
        self.ordered_arcs = []

        transitions = set()
        for arc in self.arcs:
            left = arc[0]
            right = arc[1]

            if left.startswith('p'):  # If it's a place, add it to the places set
                self.places.add(self.check_place(left))
            elif left.startswith('t'):  # If it's a transition, add it to the transitions set
                transitions.add(self.check_transition(left))

            if right.startswith('p'):  # If it's a place, add it to the places set
                self.places.add(self.check_place(right))
            elif right.startswith('t'):  # If it's a transition, add it to the transitions set
                transitions.add(self.check_transition(right))

        # Create a list of transitions ordered by transition number (extracted from the transition name)
        ordered_transitions = sorted(transitions, key=lambda t: int(t[1:]))
    
        if len(ordered_transitions) != len(intervals):
            raise ValueError(f"The number of transitions ({len(ordered_transitions)}) must be equal to the number of intervals ({len(intervals)}).")

        # Create the transitions + intervals vector
        self.transitions = []
        prev = -1
        for t in ordered_transitions:
            # Since transitions are ordered, we can get the corresponding interval from the intervals vector
            index = int(t[1:])
            if index == prev + 1:
                self.transitions.append((t, intervals[index]))
                prev = index
            else:
                raise ValueError(f"Transition {t} cannot be used without first using t{prev+1}.")

        self.ordered_trns = self.order_transitions()
    
    def order_transitions(self):
        #orders and renames the transitions of the generated petri net

        ordered_transitions = []
        visited = set()

        def dfs(transition):
            visited.add(transition)
            for child in self.get_children(self.idx(transition)):
                if child not in visited:
                    dfs(child)
            ordered_transitions.append(transition)

        for transition in self.transitions:
            if transition[0] not in visited:
                dfs(transition[0])

        ordered_transitions.reverse()
        self.mapping = self.retrieve_mapping(ordered_transitions)

        #print(self.mapping)
        temp = [("t"+str(i), self.get_interval(self.idx(transition))) for i,transition in enumerate(ordered_transitions)]

        self.ordered_trns = temp
        self.modify_arcs(self.mapping)

        return self.ordered_trns

    def idx(self, transition_str, ordered=False):
        if ordered:
            transition_str = self.mapping[transition_str]
        return int(transition_str[1:])
    
    def retrieve_mapping(self, temp):
        mapping = dict()
        
        partial_transitions = [t[0] for t in self.transitions]
        #print(partial_transitions)

        for i,transition in enumerate(self.transitions):
            after = transition[0]
            before = temp[partial_transitions.index(after)]
            #print(before, after)
            mapping[before] = after
        return mapping

    def modify_arcs(self, mapping):

        self.ordered_arcs = [None] * len(self.arcs)
        for i, arc in enumerate(self.arcs):
            initial = arc[0]
            final = arc[1]

            #if initial is a string that starts with "t"
            if initial.startswith("t"):
                initial = mapping[initial]

            #if final is a string that starts with "t"
            if final.startswith("t"):
                final = mapping[final]

            arc = (initial, final)
            self.ordered_arcs[i] = arc

    def check_transition(self, transition):
        transition_pattern = re.compile(r'^t\d+$')

        if not transition_pattern.match(transition):
            raise ValueError(f"Transition {transition} is not in the correct format.")
        
        return transition

    def check_place(self, place):
        place_pattern = re.compile(r'^p\d+$')

        if not place_pattern.match(place):
            raise ValueError(f"Place {place} is not in the correct format.")
        
        return place
    
    @classmethod
    def custom(cls, arcs, intervals):
        return cls(arcs, utils.valid_intervals(intervals))
    
    @classmethod
    def random(cls, dimension, left, right, type_atmg, random_intervals = True ):
        intervals = cls.generate_random_intervals(dimension, left, right, random_intervals)
        if type_atmg == 0:
            arcs = cls.gen_arcs_type0(dimension)
        elif type_atmg == 1:
            arcs = cls.gen_arcs_type1(dimension)
        elif type_atmg == 2:
            arcs = cls.gen_arcs_type2(dimension)
        else: 
            raise ValueError(f"Type {type_atmg} is not a valid type.")
        return cls(arcs,intervals)

    @classmethod
    def random_example(cls, dimension, left, right, cardinality, type_atmg, random_intervals = True ):
        atmg = cls.random(dimension, left, right, type_atmg, random_intervals)
        log = atmg.random_log(cardinality)
        return atmg, log
    
    @classmethod
    def stpn_as_atmg(cls,dimension, left, right,random_intervals=True):
        intervals = cls.generate_random_intervals(dimension, left, right,random_intervals)
        arcs = cls.gen_arcs_stpn(dimension)
        return cls(arcs,intervals)

    @staticmethod
    def gen_arcs_stpn(dimension):
        arcs = []
        for i in range(dimension):
            arcs.append(("p"+str(i), "t"+str(i)))
            arcs.append(("t"+str(i), "p"+str(i+1)))
        return arcs

    def dim(self):
    #returns the dimension (#transitions) of the TPN
        return len(self.transitions)
    
    def __str__(self):
        return f"Places: {self.places} \nTransitions: {self.transitions} \nArcs: {self.arcs}"
    
    def get_interval(self, idx_transition, ordered = False):
        if ordered:
            return self.ordered_trns[idx_transition][1]

        #returns the timestamp interval of a transition
        return self.transitions[idx_transition][1]
    
    def get_parents(self, idx_transition, ordered = False):
        #returns the direct parents of a transition in the petri net
        if ordered:
            trns = self.ordered_trns
            arcs = self.ordered_arcs
        else:
            trns = self.transitions
            arcs = self.arcs

        parents = []
        places = []
        for arc in arcs:
            if arc[1] == trns[idx_transition][0]:
                if arc[0] not in places:
                    places.append(arc[0])

        for arc in arcs:
            if arc[1] in places:
                if arc[0] not in parents:
                    parents.append(arc[0])
        return parents
    
    def get_children(self, idx_transition, ordered = False):
        #returns the direct successors of a transition in the petri net
        if ordered:
            trns = self.ordered_trns
            arcs = self.ordered_arcs
        else:
            trns = self.transitions
            arcs = self.arcs

        children = []
        places = []
        for arc in arcs:
            if arc[0] == trns[idx_transition][0]:
                if arc[1] not in places:
                    places.append(arc[1])

        for arc in arcs:
            if arc[0] in places:
                if arc[1] not in children:
                    children.append(arc[1])
        return children
    
    def reverse_mapping(self, point):
        #returns the point of the model given the point in the ordered model
        inverted_mapping = {v: k for k, v in self.mapping.items()}

        reverse_point = [0] * len(point)
        for i,val in enumerate(point):
            mapped_dim = int(inverted_mapping.get("t"+str(i))[1:])
            reverse_point[mapped_dim] = val
        return reverse_point

    def max_point(self):
        #returns the maximal point \sigma_M of the model 
        max_point = []

        for i in range(len(self.ordered_trns)):
            if len(self.get_parents(i,ordered=True)) == 0:
                max_point.append(self.get_interval(i,ordered=True)[1])
            else:
                list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(i,ordered=True)]
                parent_value = max([max_point[j] for j in list_of_parents_idxs])
                max_point.append(self.get_interval(i,ordered=True)[1] + parent_value)
        return self.reverse_mapping(max_point)

    def min_point(self):
        #returns the minimal point \sigma_m of the model
        min_point = []

        for i in range(len(self.ordered_trns)):
            if len(self.get_parents(i,ordered=True)) == 0:
                min_point.append(self.get_interval(i,ordered=True)[0])
            else:
                list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(i,ordered=True)]
                parent_value = max([min_point[j] for j in list_of_parents_idxs])
                min_point.append(self.get_interval(i,ordered=True)[0] + parent_value)
        return self.reverse_mapping(min_point)


    def random_log(self, cardinality):
        log = []

        for _ in range(cardinality):
            ex = []
            for i in range(len(self.transitions)):
                if len(self.get_parents(i, ordered=True)) == 0:
                    ex.append(round(random.uniform(self.get_interval(i,ordered=True)[0], self.get_interval(i,ordered=True)[1]), 2))
                    #print(self.transitions[i], ex)
                else:
                    #print(self.transitions[i])
                    list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(i,ordered=True)]
                    #print(list_of_parents_idxs)

                    parent_value = max([ex[i] for i in list_of_parents_idxs])
                    #print(parent_value)

                    ex.append(round(random.uniform(self.get_interval(i,ordered=True)[0], self.get_interval(i,ordered=True)[1]) + parent_value, 2))
                    #print(ex)
            log.append(self.reverse_mapping(ex))

        return log
    
    @staticmethod
    def gen_arcs_type2(dimension):
        
        arcs = []

        last_p = 0
        carry = 4

        dim_copy =  dimension
        if dimension%2 !=0:
            dimension-=1    
        for i in range(0,dimension,2):
            arcs.append(("p"+str(last_p), "t"+str(i)))
            arcs.append(("p"+str(last_p+2), "t"+str(i)))
            arcs.append(("t"+str(i), "p"+str(last_p+carry)))
            arcs.append(("t"+str(i), "p"+str(last_p+carry+1)))
            
            arcs.append(("p"+str(last_p+1), "t"+str(i+1)))
            arcs.append(("p"+str(last_p+3), "t"+str(i+1)))
            arcs.append(("t"+str(i+1), "p"+str(last_p+carry+2)))
            arcs.append(("t"+str(i+1), "p"+str(last_p+carry+3)))
            last_p += carry

        if dim_copy%2 !=0:
            arcs.append(("p"+str(last_p), "t"+str(dim_copy-1)))
            arcs.append(("p"+str(last_p+1), "t"+str(dim_copy-1)))
            arcs.append(("p"+str(last_p+2), "t"+str(dim_copy-1)))
            arcs.append(("p"+str(last_p+3), "t"+str(dim_copy-1)))
            arcs.append(("t"+str(dim_copy-1), "p"+str(last_p+5)))

        return arcs
        
    
    @staticmethod
    def gen_arcs_type0(dimension):
        #returns a petri net with all transitions in parallel and not linked to each other
        arcs = []
        for i in range(dimension):
            arcs.append(("p"+str(i), "t"+str(i)))
            arcs.append(("t"+str(i), "p"+str(dimension+i)))
        return arcs
    

    @staticmethod
    def gen_arcs_type1(dimension):
        arcs = [("p0","t0"), ("t"+str(dimension-1), "p"+str(2*(dimension-2)+1))]

        for i in range(1,dimension-1):
            arcs.append(("t0", "p"+str(i)))
            arcs.append(("p"+str(i), "t"+str(i)))
            arcs.append(("t"+str(i), "p"+str(dimension+i-2)))
            arcs.append(("p"+str(dimension+i-2), "t"+str(dimension-1)))

        return arcs
    
    @staticmethod
    def generate_random_intervals(dimension, left=0, right=100, random_intervals = True):
        intervals = []
        for _ in range(dimension):
            if random_intervals:
                start = round(random.uniform(left, right),2)
                end = round(random.uniform(start+left, start + right),2)
            else:
                start = round(left,2)
                end = round(start+ right,2)
            intervals.append((start, end))
        return utils.valid_intervals(intervals)
    
    def search_space(self, distance_type = 0, n_div = 5):
        if distance_type == 0:
                return self.search_space_stamponly(0,0, [],[], n_div)
        else:
            return self.search_space_delayonly(n_div)

    def get_interval(self, idx_transition, ordered = False):
        if ordered:
            return self.ordered_trns[idx_transition][1]
        else: 
            return self.transitions[idx_transition][1]

    def search_space_stamponly(self, counter, brought, gamma, search_space,step):
        #build the search space (set of "all" points accepted by the model) brute force, for a ATMG as described in the experiments section of the paper
        #as the space is continuous, the parameter "step" determines the number of splits of the space of each dimension into equidistant points

        if counter == self.dim():
            # Base case: all loops have been executed, do something with the values
            search_space.append(self.reverse_mapping(gamma.copy()))
            #print(gamma, search_space)
            return search_space
        else:
            if len(self.get_parents(counter, ordered=True)) == 0:
                brought = 0
            else:
                list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(counter, ordered=True)]
                brought = max([gamma[i] for i in list_of_parents_idxs])

            lower = self.get_interval(counter, ordered=True)[0] + brought
            upper = self.get_interval(counter, ordered=True)[1] + brought
            for i in  np.linspace(lower, upper, step):
                gamma.append(i)  # Modify the values for the current loop

                self.search_space_stamponly(counter + 1, 0, gamma, search_space, step)  #Recursively call the function for the next loop
                gamma.pop()  # Remove the current loop's value

        if counter == 0:
            print('done')
            return search_space
        
    def search_space_delayonly(self, n_div):
        #builds the search space when using the delay only distance
        #(discretized, still with the parameter n_div determining the number of splits of the space of each dimension into equidistant points)

        # Create a list of linspaces for each interval
        linspaces = [np.linspace(interval[0], interval[1], n_div) for interval in [couple[1] for couple in self.transitions]]

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

            for j in range(len(trace_list)):
                #print(trace_list[j])
                if len(self.get_parents(j)) != 0:
                    list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(j)]
                    parent_value = max([old_trace[i] for i in list_of_parents_idxs])
                    trace_list[j] = round(trace_list[j] - parent_value, 2)

            # Add the transformed trace to the new list
            transformed_log.append(trace_list)

        # Return the new transformed log
        return transformed_log
    
    def is_accepted_log(self, log):
        log = utils.valid_log(log)
        for trace in log:
            if not self.is_accepted_trace(trace):
                raise ValueError(f"Trace {trace} is not accepted by the ATMG.")
        return log
    
    def is_accepted_trace(self, trace):
        if len(trace) != self.dim():
            return False
        
        prev_val= 0
        for i, trns in enumerate(self.transitions):
            interval = trns[1]
            if len(self.get_parents(i)) == 0:
                prev_val = 0
            else:
                list_of_parents_idxs = [self.idx(parent) for parent in self.get_parents(i)]
                prev_val = max([trace[i] for i in list_of_parents_idxs])
            if not round(interval[0],2) <= round(trace[i] - prev_val,2) <= round(interval[1],2):
                return False            
        return True
    
    def solver_bf(self, log, distance_type = 0, n_div = 5):
        #finds all possible anti-alignments considering the discretized search space, and comparing every point brute force

        search_space = self.search_space( distance_type, n_div)
        
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

        
    def LPSolver(self, log, distance_type=0):
        #Linear Programming solver with constraints and variables as described in the paper



        # Define the dimensionality of the problem
        d = self.dim() #dimension of the space
        n = len(log) #number of points in L

        if distance_type ==1 : #if delay-only distance is used, the log is transformed in equivalent flow functions traces
            log = self.transform_log(log)

        #print(model.transitions, log[0])

        # Create the LP problem

        prob = plp.LpProblem("Maximize minimal manhattan distance", plp.LpMaximize)

        # Define the decision variables

        x = plp.LpVariable.dicts("x", range(d), lowBound=0, cat = 'Continuous')

        #Define the variable to maximize
        z = plp.LpVariable("z", lowBound=0, cat = 'Continuous')

        #print(x, z)

        
        M = np.linalg.norm(np.array(self.max_point()) - np.array(self.min_point()), ord=1)
        print('M', M)

        # Define the constraints for x, i.e. the search space constraints
        if distance_type == 0:
            max_var = plp.LpVariable.dicts("max_var", range(d), cat = 'Continuous')
            for i in range(d):
                #print(i)
                if len(self.get_parents(i)) == 0:
                    prob += x[i] >= self.get_interval(i)[0]
                    prob += x[i] <= self.get_interval(i)[1]
                else:
                    binary = plp.LpVariable.dicts("binary_"+str(i), range(len(self.get_parents(i))), cat="Binary")
                    for j, parent in enumerate(self.get_parents(i)):
                        prob+= max_var[i] >= x[self.idx(parent)]
                        prob+= max_var[i] <= x[self.idx(parent)] + (1-binary[j])*M
                        prob += x[i] >= self.get_interval(i)[0] + x[self.idx(parent)]
                    prob += plp.lpSum([binary[j] for j in range(len(self.get_parents(i)))]) == 1
                    prob += x[i] <= self.get_interval(i)[1] + max_var[i]
        else:
            for i in range(d):
                prob += x[i] >= self.get_interval(i)[0]
                prob += x[i] <= self.get_interval(i)[1]


        # Define the constraints for the absolute value of the difference between x and each point in L
        diff_plus = plp.LpVariable.dicts("diff_plus", (range(n), range(d)), lowBound=0, cat = 'Continuous')
        diff_minus = plp.LpVariable.dicts("diff_minus", (range(n), range(d)), lowBound=0, cat = 'Continuous')

        b = plp.LpVariable.dicts("b", (range(n), range(d)), cat = 'Binary')


        for j in range(n): #for every sigma in L
            for i in range(d): #for every dimension
                if self.dim() == 1: #if the model is 1-dimensional
                    prob += diff_plus[j][i] - diff_minus[j][i] == log[j] - x[i]
                else:
                    prob += diff_plus[j][i] - diff_minus[j][i] == log[j][i] - x[i]

                prob += diff_plus[j][i] <= M*b[j][i]
                prob += diff_minus[j][i] <= M*(1-b[j][i])

            prob += z <= plp.lpSum([diff_plus[j][i] + diff_minus[j][i] for i in range(d)])


        # Define the objective function
        prob += z

        #print(prob)

        # Solve the problem
        prob.solve(plp.PULP_CBC_CMD(msg=0, timeLimit =21600))

        # Print the results
        print("Status:", plp.LpStatus[prob.status])
        #print("Max Distance:", plp.value(prob.objective))

        optimal_point = [plp.value(x[i]) for i in range(d)]
        #print('Optimal Point:', optimal_point)

        return plp.value(prob.objective), optimal_point, prob.numVariables(), prob.numConstraints()
        
        
    def test_min(self, distance_type):
        
        log = [self.min_point()]
        max_point = self.max_point()


        distance, optimal_point, _, _= self.LPSolver(log, distance_type)

        if(distance_type == 1):
            log = self.transform_log(log)
            max_point = self.transform_log([max_point])

        max_dist = np.linalg.norm(np.array(log[0]) - np.array(max_point), 1)


        if np.linalg.norm(np.array(optimal_point) - np.array(max_point), 1) > 0.01 and abs(distance - max_dist) > 0.01:

            print("Min Test failed")
            print("Distance:", distance, "Max Distance:", max_dist, "Difference:", distance - max_dist)

    def test_max(self, distance_type):
            
            log = [self.max_point()]
            min_point = self.min_point()

            distance, optimal_point, _, _= self.LPSolver(log, distance_type)

            if(distance_type == 1):
                log = self.transform_log(log)
                min_point = self.transform_log([min_point])
        
            max_dist = np.linalg.norm(np.array(log[0]) - np.array(min_point), 1)

            if np.linalg.norm(np.array(optimal_point) - np.array(min_point), 1) > 0.01 and abs(distance - max_dist) > 0.01:
                print("Max Test failed") 
                print("Distance:", distance, "Max Distance:", max_dist, "Difference:", distance - max_dist)
        
    def test_mid(self, distance_type):

        log = [self.max_point(), self.min_point()]
        print(log)
        
        distance, optimal_point, _, _ = self.LPSolver(log, distance_type)

        
        if(distance_type == 1):
            log = self.transform_log(log)
        
        max_dist = np.linalg.norm(np.array(log[0]) - np.array(log[1]), 1)
        
        if not math.isclose(distance,(max_dist/2), abs_tol = 0.001):
        
            print("Mid Test failed")
            print("Distance:", distance, "Max Distance:", max_dist, "Difference:", distance - round((max_dist/2),2))
        
    def test_mid_loop(self, depth, distance_type):
        
            log = [self.max_point(), self.min_point()]
            
            if(distance_type == 1):
                log_transformed = self.transform_log(log)
                old_distance = np.linalg.norm(np.array(log_transformed[0]) - np.array(log_transformed[1]), 1)/2
            else:
                old_distance = np.linalg.norm(np.array(log[0]) - np.array(log[1]), 1)/2
        
            for i in range(depth):
                distance, optimal_point, _, _= self.LPSolver(log, distance_type)

                if distance > old_distance:
                    if not math.isclose(distance, old_distance, abs_tol=0.001):
                        print("Loop Test failed")
                        print("Distance:", distance, "Old Distance:", old_distance, "Difference:", distance - old_distance)
                        break
                
                log.append(optimal_point)
                old_distance = distance

    def test_random_loop(self, depth, cardinality, distance_type):
        
        log = self.random_log(cardinality)
        """
        if(distance_type == 1):
            log_transformed = self.transform_log(log)
            min_pt = self.transform_log([self.min_point()])
            max_pt = self.transform_log([self.max_point()])
            internal_distances = max([np.linalg.norm(np.array(p2) - np.array(p1), 1) for p1,p2 in product(log_transformed,log_transformed)])/2
            distance_to_min = min([np.linalg.norm(np.array(p) - min_pt, 1) for p in log_transformed])
            distance_to_max = min([np.linalg.norm(np.array(p) - max_pt, 1) for p in log_transformed])
        else:              
            internal_distances = max([np.linalg.norm(np.array(p2) - np.array(p1), 1) for p1,p2 in product(log,log)])/2
            #distance_to_min = min([np.linalg.norm(np.array(p) - self.min_point(), 1) for p in log])
            #distance_to_max = min([np.linalg.norm(np.array(p) - self.max_point(), 1) for p in log])
         """   

        #old_distance = max(internal_distances, distance_to_min, distance_to_max)

        old_distance = float('inf')
        for _ in range(depth):
            distance, optimal_point, _, _ = self.LPSolver(log, distance_type)

            if distance > old_distance:
                if not math.isclose(distance, old_distance, abs_tol=0.001):
                    
                    print("Distance:", distance, "Old Distance:", old_distance, "Difference:", distance - old_distance)
                    print("Random Loop Test failed")            
                    break
            log.append(optimal_point)
            old_distance = distance

"""
ex = 3

if ex == 1:
    arcs = [("p0", "t0"), ("t0", "p1"), ("t0", "p2"), ("p1", "t1"), ("p2", "t2"), ("t1","p3"), ("t2","p4") ,("p3", "t3") , ("p4", "t3"), ("t3", "p5")]
    #arcs = [("p0", "t1"), ("t1", "p1"), ("t1", "p2"), ("p1", "t2"), ("p2", "t3"), ("t2","p3"), ("t3","p4") ,("p3", "t4") , ("p4", "t4"), ("t4", "p5")]
    intervals =  [[0, 7], [0, 5], [0, 3], [0,1]]
elif ex == 2:
    arcs = [("p0", "t0"), ("t0", "p1"), ("t0", "p2"), ("p1", "t1"), ("p2", "t2"), ("t1","p3"), ("t1","p6"), ("t2","p4"), ("t2", "p7"), ("p3", "t4"), ("p7","t4"),("p4", "t3"),("p6","t3"),("t3", "p5"), ("t4","p8"), ("p5","t5"), ("p8","t5"), ("t5","p9")]
    intervals = [ [0, 7], [0, 5], [0, 3], [1,2], [1,4], [0,1]]

elif ex == 3:
    arcs = [("p0", "t0"), ("t0", "p1"), ("t0", "p2"), ("p1", "t1"), ("p2", "t2"), ("t1","p3"), ("t2","p4") ,("p3", "t3") , ("p4", "t3"), ("t3", "p5"), ("p6", "t4"), ("t4","p0"), ("t4", "p7"), ("p7","t5"), ("t5", "p8"), ("p8","t6"), ("p5", "t6"), ("t6", "p9"), ("t4", "p10"), ("p10", "t7"), ("t7", "p11"), ("p11", "t6"), ("p12", "t8"), ("t8", "p13"), ("p13", "t6")]
    intervals = [ [0, 7], [0, 5], [0, 3], [0,1], [0,3], [1,3], [1,5] , [0,55], [0,66]]
elif ex == 4:
    #3-dimensional for paper DTPN
    arcs = [("p0", "t1"), ("t1", "p1"),("p2", "t2"), ("t2", "p3"),("p1", "t3"),("p3", "t3"),("t3", "p4")]
    intervals = [(0,2),(0,3),(0,1)]
"""

model = ATMG.random(7, 0, 10, 0, False)
print(0,model)

model = ATMG.random(7, 0, 10, 1, False)
print(1,model)

model = ATMG.random(7, 0, 10, 2, False)
print(2,model)