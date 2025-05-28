# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name> Benjamin McElroy
<Class> MATH 420 Modern Methods Applied MATh 
<Date>
"""

import numpy as np
import cvxpy as cp

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    x = cp.Variable(3, nonneg=True)  # x1, x2, x3 with x >= 0

    objective = cp.Minimize(2 * x[0] + x[1] + 3 * x[2])

    constraints = [
    x[0] + 2 * x[1] <= 3,
    x[1] - 4 * x[2] <= 1,
    2 * x[0] + 10 * x[1] + 3 * x[2] >= 12
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    print("Optimal x:", x.value)
    print("Optimal value (primal objective):", prob.value)

if __name__ == "__main__":
    prob1()

 


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    n = A.shape[1]
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm1(x))
    constraints = [A @ x == b]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value

if __name__ == "__main__":
    A = np.array([[1, 2, 1, 1],
                  [0, 3, -2, -1]])
    b = np.array([7, 4])

    x_star, obj = l1Min(A, b)

    print("Optimal x:", np.round(x_star, 3))
    print("Objective value (||x||_1):", round(obj, 3))
    
    



# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    x = cp.Variable(6, nonneg=True)

    cost = np.array([4, 7, 6, 8, 8, 9])

    constraints = [
        x[0] + x[1] == 7, 
        x[2] + x[3] == 2,   
        x[4] + x[5] == 4, 
        x[0] + x[2] + x[4] == 5, 
        x[1] + x[3] + x[5] == 8 
    ]

    objective = cp.Minimize(cost @ x)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value

if __name__ == "__main__":
    shipment, total_cost = prob3()
    print("Optimal shipment (p1 to p6):", np.round(shipment, 2))
    print("Total minimum cost:", round(total_cost, 2))
    
    



# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    x = cp.Variable(3)

    Q = np.array([
        [3, 2, 1],
        [2, 4, 2],
        [1, 2, 3]
    ])
    c = np.array([3, 0, 1])

    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)

    prob = cp.Problem(objective)
    prob.solve()

    return x.value, prob.value

if __name__ == "__main__":
    minimizer, minimum = prob4()
    print("Minimizer x:", np.round(minimizer, 4))
    print("Minimum value g(x):", round(minimum, 4))



# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    m, n = A.shape
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm2(A @ x - b))
    constraints = [
        cp.sum(x) == 1,
        x >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, prob.value

if __name__ == "__main__":
    A = np.array([
        [1, 2, 1, 1],
        [0, 3, -2, -1]
    ])
    b = np.array([7, 4])

    x_opt, obj_val = prob5(A, b)
    print("Minimizer x:", np.round(x_opt, 4))
    print("Objective value:", round(obj_val, 4))



# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")