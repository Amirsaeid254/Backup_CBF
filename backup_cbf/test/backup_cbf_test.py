import torch
from torch.autograd.functional import jacobian
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torch.autograd import grad
from scipy.linalg import eig
torch.set_default_dtype(torch.float64)
from math import pi
from cvxopt import matrix, solvers
import cvxpy as cp
from time import time

def get_grad(func, x, **kwargs):
    """
    Compute the gradient of a given function with respect to its input.

    This function calculates the gradient of the function 'func' with respect to the input 'x'. It temporarily
    sets 'x' to require gradients, computes the gradient, and then restores the original gradient requirement.

    Args:
        func (callable): The function for which the gradient is computed.
        x (torch.Tensor): The input data with respect to which differentiation is performed.
        **kwargs: Additional keyword arguments passed to torch.autograd.grad.

    Returns:
        torch.Tensor: The gradient of 'func' with respect to 'x'.
    """
    if isinstance(x, tuple):
        requires_grad = [xx.requires_grad for xx in x]
        for xx in x:
            xx.requires_grad_()
        output = grad(func(*x), x, **kwargs)

        for xx, rg in zip(x, requires_grad):
            xx.requires_grad_(requires_grad=rg)
        return output

    requires_grad = x.requires_grad
    x.requires_grad_()
    output = grad(func(x), x, **kwargs)
    x.requires_grad_(requires_grad=requires_grad)
    return output[0]

def softmin(x, rho, conservative=False):
    return softmax(x=x, rho=-rho, conservative=conservative)

def softmax(x, rho, conservative=True):
    res = 1 / rho * torch.logsumexp(rho * x, dim=0)
    return res - np.log(x.size(0))/rho if conservative else res


def solve_qp(Q, c, A, b):
    """
        Solve a Quadratic Programming (QP) problem.

        Parameters:
            Q (numpy.ndarray): Quadratic cost matrix of shape (n, n).
            c (numpy.ndarray): Linear cost vector of shape (n,).
            A (numpy.ndarray): Constraint matrix of shape (p, n).
            b (numpy.ndarray): Constraint vector of shape (p,).

        Returns:
            numpy.ndarray: Optimal solution of the QP problem, a vector of shape (n,).
        """

    # Define the optimization variables
    n = Q.shape[0]
    x = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)

    # Define the constraints
    constraints = [A @ x <= b]

    # Formulate the QP problem
    problem = cp.Problem(objective, constraints)

    # Solve the QP problem
    problem.solve()

    # Extract the optimal solution
    optimal_x = x.value


    # Return the optimal solution and the optimal objective value
    return optimal_x, problem.value

def cbf_qp(h, Lfh, Lgh, alpha_func, Q, c, A_u=None, b_u=None):
    """
    Solve a Quadratic Program (QP) for Control Barrier Functions (CBF) with optional box constraints.

    This function formulates and solves a QP to enforce Control Barrier Functions (CBF) while optionally
    enforcing box constraints on the control input.

    Parameters:
    - h (numpy.ndarray): The CBF value at the current state.
    - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
    - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
    - alpha_func (function): The extended class-K alpha function.
    - Q (numpy.ndarray): The positive definite Q matrix for the QP cost.
    - c (numpy.ndarray): The linear term for the QP cost.
    - A_u (numpy.ndarray, optional): The inequality constraint matrix for control input.
    - b_u (numpy.ndarray, optional): The inequality constraint vector for control input.

    Returns:
    - numpy.ndarray: The optimal control input that satisfies the CBF and optional constraints.
    """
    A = np.vstack((-Lgh, A_u)) if A_u is not None else -Lgh
    b = np.hstack((Lfh + alpha_func(h), b_u)) if b_u is not None else Lfh + alpha_func(h)
    # A, b = preprocess_constraint(A, b)
    return solve_qp(Q, c, A, b)


def cbf_qp_box_constrained(h, Lfh, Lgh, alpha_func, Q, c, u_bound):
    """
        Solve a Quadratic Program (QP) for Control Barrier Functions (CBF) with box constraints on the control input.

        This function formulates and solves a QP to enforce Control Barrier Functions (CBF) while also enforcing box
        constraints on the control input.

        Parameters:
        - h (numpy.ndarray): The CBF value at the current state.
        - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
        - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
        - alpha_func (function): The extended class-K alpha function.
        - Q (numpy.ndarray): The positive definite Q matrix for the QP cost.
        - c (numpy.ndarray): The linear term for the QP cost.
        - u_bound (tuple or list): A tuple or list representing the upper and lower bounds for control input.

        Returns:
        - numpy.ndarray: The optimal control input that satisfies the CBF and box constraints.
        """

    A_u, b_u = make_box_constraints_from_bounds(u_bound)
    return cbf_qp(h=h, Lfh=Lfh, Lgh=Lgh, alpha_func=alpha_func, Q=Q, c=c, A_u=A_u, b_u=b_u)


def min_intervention_qp_box_constrained(h, Lfh, Lgh, alpha_func, u_des, u_bound):
    """
        Solve a Quadratic Program (QP) for Minimum Intervention Control Barrier Functions (CBF) with box constraints.

        This function formulates and solves a QP to achieve minimum intervention while enforcing Control Barrier Functions (CBF)
        and box constraints on the control input.

        Parameters:
        - h (numpy.ndarray): The CBF value at the current state.
        - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
        - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
        - alpha_func (function): The extended class-K alpha function.
        - u_des (numpy.ndarray): The desired control input.
        - u_bound (tuple or list): A tuple or list representing the upper and lower bounds for control input.

        Returns:
        - numpy.ndarray: The optimal control input that minimizes intervention while satisfying the CBF and box constraints.

        """
    Q = 2 * np.eye(u_des.shape[0])
    c = np.array([-2 * u_des])
    return cbf_qp_box_constrained(h, Lfh, Lgh, alpha_func, Q, c, u_bound)


def solve_lp(c, A, b):
    """
    Solve a Linear Programming (LP) problem.

    Parameters:
        c (numpy.ndarray): Linear cost vector of shape (n,).
        A (numpy.ndarray): Constraint matrix of shape (m, n).
        b (numpy.ndarray): Constraint vector of shape (m,).

    Returns:
        numpy.ndarray: Optimal solution of the LP problem, a vector of shape (n,).
    """
    # Define the optimization variables
    n = c.shape[0]
    x = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(c @ x)

    # Define the constraints
    constraints = [A @ x <= b]

    # Formulate the LP problem
    problem = cp.Problem(objective, constraints)


    # Set solver options
    solver_options = {'eps': 1e-7}
    problem.solve(solver=cp.SCS, **solver_options)
    # Solve the LP problem
    # problem.solve()

    # Extract the optimal solution
    optimal_x = x.value

    # Return the optimal solution
    return optimal_x, problem.value


def make_box_constraints_from_bounds(bounds):
    """
    Create box constraints (inequality constraints) for variables based on given lower and upper bounds.

    Given a set of lower and upper bounds for variables, this function generates box constraints in the form of
    inequality constraints that ensure each variable stays within its specified bounds.

    Parameters:
        bounds (list or numpy.ndarray): A list or numpy array of shape (n, 2) where n is the number of variables.
            Each row represents the lower and upper bounds for a variable.

    Returns:
        numpy.ndarray: A matrix A of shape (2n, n) representing the inequality coefficients for the box constraints.
        numpy.ndarray: A vector b of length 2n representing the right-hand side of the box constraints.

    Examples:
    bounds = [[-1, 1], [0, 2], [2, 4]]
    A, b = make_box_constraints_from_bounds(bounds)
    print(A)
    [[-1.  0.  0.]
     [ 0. -1.  0.]
     [ 0.  0. -1.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    print(b)
    [ 1.  0.  2. -1. -0. -4.]
    """
    bounds = np.array(bounds)
    num_variables = bounds.shape[0]

    # Create A
    A = np.vstack((-np.eye(num_variables), np.eye(num_variables)))

    # Create b for x_i >= x_min and x_i <= x_max
    b = np.concatenate((-bounds[:, 0], bounds[:, 1]))

    return A, b





# Define your dynamics as a custom nn.Module
# class DynamicsModule(nn.Module):
#     def forward(self, t, y):
#         return dynamics(y, u_b(y))
#
# Define dynamics

# # m = 0.5
# m = 3/225
# l = 15.0
# gr = 10.0
u_max = 1.5
t_seq = torch.linspace(0, 5, 50)

def f(x):
    # return torch.stack([x[1], 3 * gr * torch.sin(x[0])/(2 * l)])
    return torch.stack([x[1], torch.sin(x[0])])


def g(x):
    # return torch.tensor([0.0, 3 / (m * l**2)])
    return torch.tensor([0.0, 1.0])


def dynamics(x, u):
    return f(x) + g(x) * u


def dynamics_sensitivity(t, state_sens, u):
    x_dim = 2
    state = state_sens[:x_dim]
    sens = state_sens[x_dim:]
    sens = sens.view(x_dim, x_dim)
    dyn = dynamics(state, u)
    J = jacobian(lambda y: dynamics(y, u_b(y)), state)
    return torch.cat((dyn, torch.flatten(torch.matmul(J, sens))), dim=0)


def dyn_sens_forward_prop(state):
    y0 = torch.cat((state, torch.flatten(torch.eye(2))), dim=0)
    res = odeint(lambda t, y: dynamics_sensitivity(t, y, u_b(y[:2])), y0, t_seq, method='dopri5')
    states = res[:, :2]
    sens = res[:, 2:].view(-1, 2, 2)
    return states, sens


def fwd_prop(state, u, ts):
    t_seq = torch.linspace(0, ts, 5)
    y0 = state
    # return odeint_adjoint(dynamics_model, y0, t_seq)[-1, :]
    return odeint(lambda t, y: dynamics(y, u), y0, t_seq, method='dopri5')[-1, :]


# Define backup control
def u_b(x):
    k = torch.tensor([-3.0, -3.0], requires_grad=False)
    return u_max * torch.tanh(torch.dot(k, x) / u_max)


# Define desried conrol
def u_d(x):
    return np.array([0.0])

# Define Safe set and Backup Set
def h_b(x):
    c = 0.07
    p = torch.tensor([[1.25, 0.25], [0.25, 0.25]], requires_grad=False)
    result = torch.matmul(torch.matmul(x, p), x.T)
    return 1 - result/c if result.dim() == 0 else 1 - result.diag() / c


# def h_b(x):
#     c = 0.47
#     result = torch.matmul(torch.matmul(x, p), x.T)
#     return 1 - result/c if result.dim() == 0 else 1 - result.diag() / c

def h_s(x):
    # return 1 - torch.norm(x, p=10, dim=1) / np.pi
    return 1 - torch.norm((torch.atleast_1d(x)) / torch.tensor([pi, pi]), p=100, dim=1)

def softmin_backup_h_v1(traj, sens):
    h_func = lambda y: softmin(torch.cat((h_s(y), h_b(y[-1, :]).unsqueeze(0))), 100)
    h_states = get_grad(h_func, traj)     # partial of h with respect to states
    h = h_func(traj)
    h_grad = torch.sum(torch.bmm(h_states.unsqueeze(1), sens).squeeze(1), dim=0)
    return h, h_grad


def softmin_backup_h(state):
    state.requires_grad_()
    # traj = odeint_adjoint(dynamics_model, state, t_seq)
    traj = odeint(lambda t, y: dynamics(y, u_b(y)), state, t_seq, method='dopri5')
    h = softmin(torch.cat((h_s(traj), h_b(traj[-1, :]).unsqueeze(0))), 100)
    h_grad = grad(h, state)[0]     # partial of h with respect to states
    state.requires_grad_(requires_grad=False)
    return h, h_grad


def get_feasibility_factor(Lfh, Lgh, h, eps, alpha):
    A_u, b_u = make_box_constraints_from_bounds(u_bound)
    u, optval = solve_lp(-Lgh, A_u, b_u)
    Lghu_max = -optval
    return Lfh[0] + Lghu_max + alpha * (h - eps)

def get_softmin_u(state, j):
    if j == 1:
        traj, sens = dyn_sens_forward_prop(state)
        h, h_grad = softmin_backup_h_v1(traj, sens)
    else:
        h, h_grad = softmin_backup_h(state)

    h = h.detach().numpy()
    Lfh = np.atleast_1d(torch.dot(h_grad, f(state)).detach().numpy())
    Lgh = np.atleast_2d(torch.dot(h_grad, g(state)).detach().numpy())

    feas_fact = get_feasibility_factor(Lfh, Lgh, h, eps, alpha)

    gamma = min((h - eps) / 0.05, feas_fact / 0.05)
    u_b_temp = u_b(state).detach().numpy()
    if gamma <= 0:
        u = u_b_temp
    else:
        # if Lfh[0] + alpha * (h - eps) >= 0:
        #     u = u_d(state)
        # else:
            u, _ = min_intervention_qp_box_constrained(h=h - eps,
                                                       Lfh=Lfh, Lgh=Lgh,
                                                       alpha_func=lambda eta: alpha * eta,
                                                       u_des=u_d(state),
                                                       u_bound=u_bound)
    beta = (1 if gamma >= 1 else gamma) if gamma > 0 else 0

    return (1 - beta) * u_b_temp + beta * u, h

# def find_P_matrix():
#     x_eq = torch.tensor([0.0, 0.0]).requires_grad_()
#     A = jacobian(lambda x: dynamics(x, u_b(x)), x_eq).detach().numpy()
#     print(eig(A))
#     P = ct.lyap(A.T, 0.5 * np.eye(2))
#     return P


ts = 0.1
tf = 20.0
t_list = np.linspace(0, tf, int(tf / ts))
# dynamics_model = DynamicsModule()

x_eq = torch.tensor([0.0, 0.0]).requires_grad_()
# p = find_P_matrix()
# p = torch.as_tensor(p)
x0 = torch.tensor([0.5, 0])
state = x0
alpha = 1.0
eps = 0.0
u_bound = np.array([[-u_max, u_max]])

states_queue = []
controls_queue = []
backup_controls_queue = []
h = []

for j in range(1):
    states = [x0]
    controls = []
    backup_controls = []
    h=[]
    for i, t in enumerate(t_list):
        control, h_val = get_softmin_u(states[-1], j)
        backup_controls.append(u_b(states[-1]))
        h.append(h_val)
        with torch.no_grad():
            next_state = fwd_prop(states[-1], control, ts)
        states.append(next_state)
        controls.append(control)

    states = torch.vstack(states).detach().numpy()
    controls = np.hstack(controls)
    backup_controls = np.hstack(backup_controls)
    h = np.hstack(h)
    states_queue.append(states)
    controls_queue.append(controls)
    backup_controls_queue.append(backup_controls)


# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot state trajectories from both runs on the same subplot
for states in states_queue:
    ax.plot(states[:, 0], states[:, 1])

ax.set_xlabel("State (Dimension 1)")
ax.set_ylabel("State (Dimension 2)")
ax.set_title("State Trajectory")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# Show the combined plot
plt.show()

# Create a figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# Plot state trajectories from both runs
for states in states_queue:
    # First subplot: states[:, 0] vs t_list
    axes[0].plot(t_list, states[:-1, 0])

    # Second subplot: states[:, 1] vs t_list
    axes[1].plot(t_list, states[:-1, 1])

# Plot controls from both runs
for controls in controls_queue:
    # Third subplot: controls vs t_list
    axes[2].plot(t_list, controls)

for backup_controls in backup_controls_queue:
    # Third subplot: controls vs t_list
    axes[2].plot(t_list, backup_controls)


# Set common labels and titles
axes[0].set_xlabel("Time (t_list)")
axes[0].set_ylabel("State (Dimension 1)")
axes[0].set_title("State Dimension 1 vs Time")

axes[1].set_xlabel("Time (t_list)")
axes[1].set_ylabel("State (Dimension 2)")
axes[1].set_title("State Dimension 2 vs Time")

axes[2].set_xlabel("Time (t_list)")
axes[2].set_ylabel("Controls")
axes[2].set_title("Controls vs Time")

# Adjust spacing between subplots
plt.tight_layout()

# Show the combined plot
plt.show()


np.savez('test_data.npz', states=states, controls=controls)

