# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name> Benjamin Mcelroy
<Class> MATH 420 Modern Methods Applied MATH
<Date> April 24th 
"""

import numpy as np


def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    
    A = np.array([
        [3, -1, 4],
        [1,  5, -9]])


    B = np.array([
        [ 2,  6, -5,  3],
        [ 5, -8,  9,  7],
        [ 9, -3, -2, -3]])

   
    return A @ B
   


if __name__ == "__main__":
    result = prob1()
    print("Matrix product AB:")
    print(result)
   


def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
   
    A = np.array([
        [3, 1, 4],
        [1, 5, 9],
        [-5, 3, 1]])

    
    A2 = A @ A
    A3 = A @ A2

   
    result = -A3 + 9*A2 - 15*A

    return result

# Example usage
if __name__ == "__main__":
    result = prob2()
    print("Result of -A^3 + 9A^2 - 15A:")
    print(result)
    
    
    


def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
   
    A = np.triu(np.ones((7, 7)))

    
    upper_five = np.triu(np.full((7, 7), 5), 1)  
    B = upper_five + np.tril(np.full((7, 7), -1))  

    
    result = A @ B @ A

    
    result = result.astype(np.int64)

    return result

# Example usage
if __name__ == "__main__":
    print("Result of ABA:")
    print(prob3())



def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    A_copy = A.copy()
    A_copy[A_copy < 0] = 0
    return A_copy

if __name__ == "__main__":
    A = np.array([-3, -1, 3])
    print(prob4(A))  



def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
   
    A = np.array([[0, 2, 4],
                  [1, 3, 5]])       
    
    B = np.array([[3, 0, 0],
                  [3, 3, 0],
                  [3, 3, 3]])        
    
    C = -2 * np.eye(3, dtype=int)    
    I = np.eye(3, dtype=int)         

    
    Z3x3 = np.zeros((3, 3), dtype=int)
    Z3x2 = np.zeros((3, 2), dtype=int)
    Z2x2 = np.zeros((2, 2), dtype=int)
    Z2x3 = np.zeros((2, 3), dtype=int)

    
    top_row = np.hstack((Z3x3, A.T, I))            
    middle_row = np.hstack((A, Z2x2, Z2x3))        
    bottom_row = np.hstack((B, Z3x2, C))          

    
    block_matrix = np.vstack((top_row, middle_row, bottom_row))  

    return block_matrix

if __name__ == "__main__":
    print(prob5())



def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    
    A = np.atleast_2d(A)  
    row_sums = A.sum(axis=1, keepdims=True)
    return A / row_sums

if __name__ == "__main__":
    print(prob6(A))
    


def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    raise NotImplementedError("Problem 7 Incomplete")
