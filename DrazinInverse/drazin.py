# drazin.py
"""Volume 1: The Drazin Inverse.
<Name> Benjamin McElroy
<Class> MATH 420 Modern Methods Applied MATH 
<Date> Thursday May 15th 
"""

import numpy as np
from scipy import linalg as la 
from scipy.linalg import schur, inv


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """
    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, AD, k, tol=1e-5):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    A = np.array(A)
    AD = np.array(AD)

    # Check three properties of Drazin inverse
    cond1 = np.allclose(A @ AD, AD @ A, atol=tol)
    cond2 = np.allclose(np.linalg.matrix_power(A, k+1) @ AD, np.linalg.matrix_power(A, k), atol=tol)
    cond3 = np.allclose(AD @ A @ AD, AD, atol=tol)

    return cond1 and cond2 and cond3

if __name__ == "__main__":
    
    A = np.array([[1, 3, 0, 0],
              [0, 1, 3, 0],
              [0, 0, 1, 3],
              [0, 0, 0, 0]])

    AD = np.array([[1, -3, 9, 81],
               [0, 1, -3, -18],
               [0, 0, 1, 3],
               [0, 0, 0, 0]])


    B = np.array([[1, 1, 3],
              [5, 2, 6],
              [-2, -1, -3]])

    BD = np.zeros((3, 3))


print("AD is Drazin inverse of A:", is_drazin(A, AD, 1)) 
print("BD is Drazin inverse of B:", is_drazin(B, BD, 3))




# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    n = A.shape[0]

    T1, Q1, k1 = schur(A, sort=lambda x: abs(x) > tol)
    
    T2, Q2, k2 = schur(A, sort=lambda x: abs(x) <= tol)

    U = np.hstack((Q1[:, :k1], Q2[:, :(n - k1)]))

    U_inv = inv(U)
    V = U_inv @ A @ U

    Z = np.zeros((n, n), dtype=complex)

    if k1 != 0:
        M_inv = inv(V[:k1, :k1])
        Z[:k1, :k1] = M_inv

    A_D = U @ Z @ U_inv

    if np.all(np.isreal(A)):
        A_D = A_D.real

    return A_D

if __name__ == "__main__":
    
    A = np.array([
        [1, 3, 0, 0],
        [0, 1, 3, 0],
        [0, 0, 1, 3],
        [0, 0, 0, 0]
    ], dtype=float)

    AD = drazin_inverse(A)

    print("Drazin inverse of A:")
    print(AD)




# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = A.shape[0]
    
    deg = np.diag(np.sum(A, axis=1))
    
    L = deg - A

    L_drazin = drazin_inverse(L)

    R = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            diff = e_i - e_j
            R_ij = diff @ L_drazin @ diff
            R[i, j] = R[j, i] = R_ij

    return R

def drazin_inverse(A, tol=1e-5):
    n = A.shape[0]
    T1, Q1, k1 = schur(A, sort=lambda x: abs(x) > tol)
    T2, Q2, _ = schur(A, sort=lambda x: abs(x) <= tol)
    U = np.hstack((Q1[:, :k1], Q2[:, :(n - k1)]))
    U_inv = inv(U)
    V = U_inv @ A @ U
    Z = np.zeros((n, n), dtype=complex)
    if k1 > 0:
        Z[:k1, :k1] = inv(V[:k1, :k1])
    A_D = U @ Z @ U_inv
    if np.all(np.isreal(A)):
        A_D = A_D.real
    return A_D

if __name__ == "__main__":

    A = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=float)

    R = effective_resistance(A)
    print("Effective resistance matrix:")
    print(R)



# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        raise NotImplementedError("Problem 4 Incomplete")


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        raise NotImplementedError("Problem 5 Incomplete")
