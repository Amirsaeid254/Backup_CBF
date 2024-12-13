from cvxopt import matrix,solvers
import torch
import numpy as np


def solve_lp_batch_cvxopt(c_batch, G_batch, h_batch):
    """
    Solve a batch of LP problems using CVXOPT and return the solution as a PyTorch tensor.

    Each problem is:
        minimize    c^T x
        subject to  Gx <= h

    Args:
        c_batch (torch.Tensor): Batch of objective function coefficients, shape (batch_size, n).
        G_batch (torch.Tensor): Batch of inequality constraint matrices, shape (batch_size, m, n).
        h_batch (torch.Tensor): Batch of inequality constraint vectors, shape (batch_size, m).

    Returns:
        torch.Tensor: Solutions to the LPs, shape (batch_size, n).
    """
    batch_size, n = c_batch.shape
    m = G_batch.shape[1]

    # Prepare empty list to store the solutions
    solutions = []

    # Solve each LP problem individually
    for i in range(batch_size):
        # Extract the c, G, h for the i-th problem
        c = c_batch[i].numpy()  # Convert to NumPy
        G = G_batch[i].numpy()  # Convert to NumPy
        h = h_batch[i].numpy()  # Convert to NumPy

        # Convert c, G, and h to the format CVXOPT expects
        c = matrix(c.astype(np.float64))
        G = matrix(G.astype(np.float64))
        h = matrix(h.astype(np.float64))


        # Solve the LP using CVXOPT's linprog function
        solvers.options['show_progress'] = False
        sol = solvers.lp(c, G, h, msg=False)

        # Extract the solution and append it to the list
        x = np.array(sol['x']).flatten()  # Convert solution to a 1D array
        solutions.append(x)

    # Convert solutions to a PyTorch tensor and return
    return torch.tensor(np.array(solutions), dtype=c_batch.dtype, device=c_batch.device)


# Example usage
torch.manual_seed(0)
batch_size = 5  # Number of LP problems
n = 3  # Number of variables
m = 4  # Number of constraints

# Randomly generate problem data
c_batch = torch.rand(batch_size, n)  # Objective coefficients for each problem
G_batch = torch.rand(batch_size, m, n)  # Constraint matrices for each problem
h_batch = torch.rand(batch_size, m)  # Constraint vectors for each problem

# Solve batched LPs using CVXOPT
solutions = solve_lp_batch_cvxopt(c_batch, G_batch, h_batch)
print("Solutions for batched LPs (Torch):")
print(solutions)
