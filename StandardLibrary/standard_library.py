# standard_library.py
"""Python Essentials: The Standard Library.
<Name> Benjamin McElroy
<Class> MTH 420 Modern Methods Applied Math
<Date> April 17th
"""

from math import sqrt


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    return min(L), max(L), sum(L) / len(L)
if __name__ == "__main__":
    sample_list = [3, 7, 2, 9, 5]
    result = prob1(sample_list)
    print(f"Min: {result[0]}, Max: {result[1]}, Avg: {result[2]}")

   


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """

    print("---- int ----")
    int_1 = 5
    int_2 = int_1
    int_2 += 1
    print(int_1 == int_2)  

    print("---- str ----")
    str_1 = "hello"
    str_2 = str_1
    str_2 += "!"
    print(str_1 == str_2)  

    print("---- list ----")
    list_1 = [1, 2, 3]
    list_2 = list_1
    list_2.append(4)
    print(list_1 == list_2)  

    print("---- tuple ----")
    tuple_1 = (1, 2, 3)
    tuple_2 = tuple_1
    tuple_2 += (4,)  
    print(tuple_1 == tuple_2) 

    print("---- set ----")
    set_1 = {1, 2, 3}
    set_2 = set_1
    set_2.add(4)
    print(set_1 == set_2)  

if __name__ == "__main__":
    prob2()

    print("\n--- Conclusion ---")
    print("Mutable types: list, set")
    print("Immutable types: int, str, tuple")

    

# Problem 3

import calculator 

def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
   

    square_a = calculator.product(a, a)
    square_b = calculator.product(b, b)
    sum_squares = calculator.sum(square_a, square_b)
    return calculator.sqrt(sum_squares)


if __name__ == "__main__":
    h = hypot(3, 4)
    print(f"The hypotenuse is: {h}")

   


# Problem 4
import itertools 
from itertools import combinations  
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    result = []
    for r in range(len(A) + 1):
        for combo in itertools.combinations(A, r):
            result.append(set(combo))
    return result
if __name__ == "__main__":
    A = ['a', 'b', 'c']
    pset = power_set(A)
    for subset in pset:
        print(subset)

    


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    raise NotImplementedError("Problem 5 Incomplete")
