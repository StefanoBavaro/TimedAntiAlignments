import random
from decimal import Decimal, ROUND_HALF_UP

def round2(value):
    #Rounds a value to two decimal places with ROUND_HALF_UP.
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def valid_intervals(list_of_intervals):
    if not isinstance(list_of_intervals, list):
        raise ValueError(f"Invalid time constraints {list_of_intervals}: must be a list.")
    
    if len(list_of_intervals) == 0:
        raise ValueError("The given list of time constraints is empty.")

    # Check if the given list is a list of valid intervals.
    for interval in list_of_intervals:
        if len(interval) != 2:
            raise ValueError(f"Invalid time constraint {interval}: must have exactly two endpoints.")
        start, end = interval
        if start < 0 or end < 0:
            raise ValueError(f"Invalid time constraint {interval}: endpoints must be non-negative.")
        if start > end:
            raise ValueError(f"Invalid time constraint {interval}: start endpoint {start} cannot be greater than end endpoint {end}.")
    return list_of_intervals

def valid_log(log):
    #Check if the given log is accepted by the given STPN.
    if not isinstance(log, list):
        raise ValueError(f"Invalid log {log}: must be a list.")
    
    if len(log) == 0:
        raise ValueError("The given log is empty.")
    
    return log