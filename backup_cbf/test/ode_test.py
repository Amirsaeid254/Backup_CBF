import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt


# Assuming you have these classes/functions defined
class Dynamics:
    def rhs(self, state, action):
        # Example dynamics (modify according to your system)
        return -0.1 * state + action


class Controller:
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, state):
        return -self.gain * state


def run_comparison_tests():
    # Setup parameters
    sim_time = 5.0
    timestep = 0.01
    x0 = torch.tensor([[1.0, 0.0]])  # Initial state
    action_num = 2
    dynamics = Dynamics()

    # Create two identical controllers
    action_funcs = [Controller(0.5) for _ in range(action_num)]

    # Time points
    t = torch.linspace(0.0, sim_time, int(sim_time / timestep) + 1)

    print("Running different ODE solving methods...")

    # Method 1: Batched approach
    print("\nMethod 1: Batched computation")
    Test_batched = odeint(
        func=lambda t, y: torch.cat([dynamics.rhs(y, action(y))
                                     for y, action in zip(y.chunk(action_num), action_funcs)],
                                    dim=0),
        y0=x0.repeat_interleave(action_num, dim=0),
        t=t,
        method='dopri5'
    ).detach()

    # Method 2: Batched with higher precision
    print("\nMethod 2: Batched computation with higher precision")
    Test_batched_precise = odeint(
        func=lambda t, y: torch.cat([dynamics.rhs(y, action(y))
                                     for y, action in zip(y.chunk(action_num), action_funcs)],
                                    dim=0),
        y0=x0.repeat_interleave(action_num, dim=0),
        t=t,
        method='dopri5',
        rtol=1e-7,
        atol=1e-9
    ).detach()

    # Method 3: Separate ODEs
    print("\nMethod 3: Separate computations")
    Test_separate = []
    for action in action_funcs:
        result = odeint(
            func=lambda t, y: dynamics.rhs(y, action(y)),
            y0=x0,
            t=t,
            method='dopri5'
        ).detach()
        Test_separate.append(result)
    Test_separate = torch.stack(Test_separate, dim=2)  # Stack results

    # Analyze results
    print("\nAnalyzing results...")

    # Split batched results
    Test_batched_split = Test_batched.chunk(2, dim=1)
    Test_batched_precise_split = Test_batched_precise.chunk(2, dim=1)

    # Compute differences
    diff_batched = (Test_batched_split[0] - Test_batched_split[1]).abs()
    diff_precise = (Test_batched_precise_split[0] - Test_batched_precise_split[1]).abs()
    diff_separate = (Test_separate[..., 0] - Test_separate[..., 1]).abs()

    print(f"\nMaximum differences between identical controllers:")
    print(f"Standard batched: {diff_batched.max().item():.2e}")
    print(f"High-precision batched: {diff_precise.max().item():.2e}")
    print(f"Separate integration: {diff_separate.max().item():.2e}")

    # Plotting
    plt.figure(figsize=(15, 5))

    # Plot first state component
    plt.subplot(121)
    plt.plot(t, Test_batched_split[0][:, 0, 0], 'b-', label='Batched - Controller 1')
    plt.plot(t, Test_batched_split[1][:, 0, 0], 'r--', label='Batched - Controller 2')
    plt.plot(t, Test_separate[:, 0, 0, 0], 'g:', label='Separate - Controller 1')
    plt.title('First State Component')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)

    # Plot differences
    plt.subplot(122)
    plt.semilogy(t, diff_batched[:, 0, 0], 'b-', label='Standard Batched')
    plt.semilogy(t, diff_precise[:, 0, 0], 'r--', label='High-precision Batched')
    plt.semilogy(t, diff_separate[:, 0, 0], 'g:', label='Separate')
    plt.title('Absolute Differences Between Controllers')
    plt.xlabel('Time')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run tests
    run_comparison_tests()