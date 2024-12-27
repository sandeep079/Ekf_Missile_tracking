import numpy as np
import matplotlib.pyplot as plt

def simulate_ballistic_system(initial_state, time_steps, delta_t, K_interval, Q, R):
    # Constants
    g = 9.81  # Gravitational acceleration
    n_states = 7  # Number of states (x, y, z, vx, vy, vz, density gradient)

    # Extract K interval bounds
    K_min, K_max = K_interval

    # State vectors
    states = np.zeros((time_steps, n_states))
    states[0] = initial_state

    # Interval bounds
    lower_bounds = np.zeros((time_steps, n_states))
    upper_bounds = np.zeros((time_steps, n_states))

    # Set initial interval estimates
    lower_bounds[0] = initial_state - 0.1 * initial_state
    upper_bounds[0] = initial_state + 0.1 * initial_state

    for k in range(1, time_steps):
        
        # Update states using interval dynamics
        x = states[k - 1]
        x_dot = np.zeros_like(x)

        # State equations
        velocity = np.sqrt(x[3]**2 + x[4]**2 + x[5]**2)
        x_dot[0:3] = x[3:6]  # Update position
        x_dot[3] = -(1/2) * g * x[6] * x[3] * velocity  # Update velocity x
        x_dot[4] = -(1/2) * g * x[6] * x[4] * velocity  # Update velocity y
        x_dot[5] = -(1/2) * g * x[6] * x[5] * velocity - g  # Update velocity z
        x_dot[6] = -K_min * x[6]  # Update density gradient (lower bound example)

        # Numerical integration (Euler method)
        states[k] = states[k - 1] + delta_t * x_dot

        # Update interval bounds
        lower_bounds[k] = states[k] - 0.05 * states[k]
        upper_bounds[k] = states[k] + 0.05 * states[k]

    return states, lower_bounds, upper_bounds

# Parameters
time_steps = 100
delta_t = 1.0  # 1 second interval
initial_state = np.array([3.2e5, 3.2e5, 2.1e5, -1.5e4, -1.5e4, -8.1e3, 5e-10])
K_interval = [2.3e-5, 3.5e-5]
Q = np.diag([0, 0, 0, 100, 100, 100, 2.0e-18])  # Process noise covariance
R = np.diag([150, 150, 150])  # Measurement noise covariance

# Simulate
states, lower_bounds, upper_bounds = simulate_ballistic_system(
    initial_state, time_steps, delta_t, K_interval, Q, R
)

# Plot results
plt.figure(figsize=(12, 8))

for i, label in enumerate(["x", "y", "z", "vx", "vy", "vz", "density gradient"]):
    plt.subplot(4, 2, i + 1)
    plt.plot(states[:, i], label=f"State: {label}")
    plt.fill_between(
        np.arange(time_steps), lower_bounds[:, i], upper_bounds[:, i], alpha=0.3, label="Interval Bounds"
    )
    plt.xlabel("Time Steps")
    plt.ylabel(label)
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
