
from cvxopt import solvers,matrix
from hocbf_composition.utils.utils import *
# from torchdiffeq import odeint_adjoint as odeint


def get_trajs_from_batched_action_func(x0, dynamics, action_funcs, timestep, sim_time, method='euler'):
    action_num = len(action_funcs)
    return odeint(
            func=lambda t, y: torch.cat([dynamics.rhs(yy.squeeze(0), action(yy.squeeze(0)))
                                         for yy, action in zip(y.chunk(action_num, dim=1), action_funcs)],
                                        dim=0),
            y0=x0.unsqueeze(0).repeat(1, action_num, 1),
            t=torch.linspace(0.0, sim_time, int(sim_time / timestep)),
            method=method
        ).squeeze(1)


# def get_trajs_from_batched_action_func(x0, dynamics, action_funcs, timestep, sim_time, method='euler'):
#     action_num = len(action_funcs)
#     return odeint(
#             func=lambda t, y: torch.cat([dynamics.rhs(yy.squeeze(0), action(yy.squeeze(0)))
#                                          for yy, action in zip(y.chunk(action_num, dim=1), action_funcs)],
#                                         dim=0),
#             y0=x0.unsqueeze(0).repeat(1, action_num, 1),
#             t=torch.linspace(0.0, sim_time, int(sim_time / timestep)),
#             method=method
#         ).squeeze(1)
#
# def get_trajs_from_batched_action_func(x0, dynamics, action_funcs, timestep, sim_time, method='euler'):
#     action_num = len(action_funcs)
#     return odeint(
#             func=lambda t, y: torch.cat([dynamics.rhs(yy, action(yy))
#                                          for yy, action in zip(y.chunk(action_num, dim=0), action_funcs)],
#                                         dim=0),
#             y0=x0.repeat(action_num, 1),
#             t=torch.linspace(0.0, sim_time, int(sim_time / timestep)),
#             method=method
#         )


def lp_solver(c, G, h):

    batch_size, n = c.shape

    # Prepare empty list to store the solutions
    solutions = []

    # Solve each LP problem individually
    for i in range(batch_size):
        # Extract the c, G, h for the i-th problem
        c_np = c[i].numpy()  # Convert to NumPy
        G_np = G[i].numpy()  # Convert to NumPy
        h_np = h[i].numpy()  # Convert to NumPy

        # # Validate inputs
        # if np.any(np.isnan(c_np)) or np.any(np.isnan(G_np)) or np.any(np.isnan(h_np)):
        #     raise ValueError("NaN values detected in input")
        #
        # if np.any(np.isinf(c_np)) or np.any(np.isinf(G_np)) or np.any(np.isinf(h_np)):
        #     raise ValueError("Infinite values detected in input")

        # Scale the problem to improve numerical stability
        scale_c = np.max(np.abs(c_np)) if np.max(np.abs(c_np)) > 0 else 1.0
        scale_G = np.max(np.abs(G_np)) if np.max(np.abs(G_np)) > 0 else 1.0
        scale_h = np.max(np.abs(h_np)) if np.max(np.abs(h_np)) > 0 else 1.0

        c_scaled = matrix((c_np / scale_c).astype(np.float64))
        G_scaled = matrix((G_np / scale_G).astype(np.float64))
        h_scaled = matrix((h_np / scale_h).astype(np.float64))

        # Convert c, G, and h to the format CVXOPT expects
        # c_cvx = matrix(c_np.astype(np.float64))
        # G_cvx = matrix(G_np.astype(np.float64))
        # h_cvx = matrix(h_np.astype(np.float64))



        # Solve the LP using CVXOPT's linprog function
        solvers.options['show_progress'] = False
        # sol = solvers.lp(C_cvx, G_cvx, h_cvx, msg=False)
        sol = solvers.lp(c_scaled, G_scaled, h_scaled, msg=False)


        # Extract the solution and append it to the list
        x = np.array(sol['x']).flatten() * (scale_h / scale_G)
        solutions.append(x)

    # Convert solutions to a PyTorch tensor and return
    return torch.tensor(np.array(solutions), dtype=c.dtype, device=c.device)


def vectorized_lp_solver(c, G, h):
    """
    Solve multiple LP problems simultaneously using torch.block_diag for tensor operations.

    Args:
        c: Tensor of shape (batch_size, n) containing objective coefficients
        G: Tensor of shape (batch_size, m, n) containing inequality constraints
        h: Tensor of shape (batch_size, m) containing inequality bounds

    Returns:
        Tensor of shape (batch_size, n) containing solutions
    """
    batch_size, n = c.shape

    # Create block diagonal matrix using torch.block_diag
    G_blocks = torch.block_diag(*[G[i] for i in range(batch_size)])

    # Stack c and h using PyTorch operations
    c_stacked = c.reshape(-1)
    h_stacked = h.reshape(-1)

    # Convert to CVXOPT format
    c_cvx = matrix(c_stacked.cpu().numpy().astype(np.float64))
    G_cvx = matrix(G_blocks.cpu().numpy().astype(np.float64))
    h_cvx = matrix(h_stacked.cpu().numpy().astype(np.float64))

    # Solve the combined LP
    solvers.options['show_progress'] = False
    sol = solvers.lp(c_cvx, G_cvx, h_cvx, msg=False)

    # Convert solution back to PyTorch tensor
    x = torch.tensor(np.array(sol['x']), dtype=c.dtype, device=c.device)
    x_reshaped = x.reshape(batch_size, n)

    return x_reshaped


def make_ellipse_barrier_functional(center, A):

    center = vectorize_tensors(tensify(center)).to(torch.float64)
    A = tensify(A).to(torch.float64)
    return lambda x: 1 - torch.einsum('bi,ij,bj->b', vectorize_tensors(x) - center, A, vectorize_tensors(x) - center).unsqueeze(-1)