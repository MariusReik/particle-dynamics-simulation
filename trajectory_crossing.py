import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This script simulates a transformed system in two dimensions (X, Y).
It analyzes how initial conditions (Y0) influence crossings at specific X values (π/2 and 0).

Parameters:
- k: Wave number, calculated as `2 * π / 10`.
- omega: Angular frequency, derived as `sqrt(g * k)` where `g` is gravity (9.81 m/s^2).
- epsilon: Scaling factor (set to 1 here).
- M: Composite parameter `omega * epsilon` used in the differential equations.

Outputs:
- Scatter plots showing the relationship between initial Y0 and crossing values of Y at X=π/2 and X=0.
"""

# Parameter values
k = 2 * np.pi / 10  # Wave number
omega = np.sqrt(9.81 * k)  # Angular frequency
epsilon = 1  # Scaling factor
M = omega * epsilon  # Composite parameter

# Time span for simulation
t_span = (0, 10)  # Start and end time
t = np.linspace(0, 10, 5000)  # Time points for evaluation


def transformed_system(t: float, z: np.ndarray, M: float, k: float, omega: float) -> np.ndarray:
    """
    Defines the differential equations for the transformed system (X, Y).
    
    Parameters:
    - t: Time (scalar or vector).
    - z: State vector [X, Y].
    - M: Composite parameter.
    - k: Wave number.
    - omega: Angular frequency.
    
    Returns:
    - [dX, dY]: Derivatives of X and Y with respect to time.
    """
    X, Y = z
    dX = k * M * np.exp(Y) * np.cos(X) - omega
    dY = k * M * np.exp(Y) * np.sin(X)
    return np.array([dX, dY])


def find_crossings(x: np.ndarray, y: np.ndarray, t: np.ndarray, target: float) -> list[float]:
    """
    Detects the points where the trajectory (x, y) crosses a specified target value of X.

    Parameters:
    - x (np.ndarray): Array of X values along the trajectory.
    - y (np.ndarray): Array of Y values corresponding to the X values.
    - t (np.ndarray): Array of time values corresponding to the trajectory.
    - target (float): The target X value to detect crossings (e.g., π/2 or 0).

    Returns:
    - crossings (list[float]): A list of Y values at the crossing points where X equals the target.
      Each crossing corresponds to a point where the X values change from one side of the target to the other.
    """
    crossings = []
    for i in range(len(x) - 1):
        if (x[i] - target) * (x[i + 1] - target) < 0:  # Check for opposite signs
            t_cross = t[i] + (t[i + 1] - t[i]) * (target - x[i]) / (x[i + 1] - x[i])
            y_cross = y[i] + (y[i + 1] - y[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
            crossings.append(y_cross)
    return crossings


# Empty lists to store Y0 and corresponding Y values for X=π/2 and X=0
Y0_pi_half = []
Y_pi_half_list = []
Y0_zero = []
Y_0_crossings_list = []

# Initial conditions for X and a range of Y0 values
x0 = np.pi  # Fixed initial X value
y0_values = np.linspace(-0.1, -2, 10)  # Range of initial Y values

# Iterate through initial Y0 values
for y0 in y0_values:
    z0 = [x0, y0]  # Initial condition [X0, Y0]

    # Solve the system
    solution = solve_ivp(
        transformed_system, t_span, z0, args=(M, k, omega), t_eval=t, rtol=1e-10, atol=1e-12
    )
    x = solution.y[0]  # X values over time
    y = solution.y[1]  # Y values over time

    # Find crossings for X=π/2 and X=0
    pi_half_crossings = find_crossings(x, y, t, np.pi / 2)
    zero_crossings = find_crossings(x, y, t, 0)

    # Store results for current Y0
    if pi_half_crossings:
        Y0_pi_half.extend([y0] * len(pi_half_crossings))
        Y_pi_half_list.extend(pi_half_crossings)
    if zero_crossings:
        Y0_zero.extend([y0] * len(zero_crossings))
        Y_0_crossings_list.extend(zero_crossings)

# Plot results
plt.figure()
plt.plot(Y0_pi_half, Y_pi_half_list, 'o', label=r'$Y_{X=\pi/2}$')  # Crossings at X=π/2
plt.plot(Y0_zero, Y_0_crossings_list, 'x', label=r'$Y_{X=0}$')  # Crossings at X=0
plt.xlabel(r'$Y_0$ (Initial condition)')
plt.ylabel(r'$Y$ crossings')
plt.title(r'Relationship between $Y_0$ and crossings at $X=\pi/2$ and $X=0$')
plt.grid()
plt.legend()
plt.show()