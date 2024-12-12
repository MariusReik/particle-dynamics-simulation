"""
def runge_kutta_4(func, z0, t_span, h, args=()):
    '''
    Funksjon som løser ODE med Runge-Kutta 4
    Input
        func: funksjon som definerer dz/dt=f(t, z)
        z0: to-dimensjonal vektor med startposisjon z=[x, y]
        t_span: tidsintervall
        h: skrittlengde
    Output
        t_values: en-dimensjonal vektor med tidspunkt
        z_values: to-dimensjonal matrise med løsningene (x,y) til ethvert tidspunkt
    '''
    t_start, t_end = t_span
    t_values = np.arange(t_start, t_end, h)
    z_values = np.zeros((len(t_values), len(z0)))

    z_values[0] = z0

    # Runge-Kutta 4 
    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        z = z_values[i - 1]

        k1 = h * func(t, z, *args)
        k2 = h * func(t + h / 2, z + k1 / 2, *args)
        k3 = h * func(t + h / 2, z + k2 / 2, *args)
        k4 = h * func(t + h, z + k3, *args)

        z_values[i] = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, z_values

# Solution using Runge-Kutta 4
t_values, z_values = runge_kutta_4(eq_31, z0, t_span, h, args=(M, k, omega))

t_values_transformed, z_values_transformed = runge_kutta_4(
    transformed_system, z0, t_span, h, args=(M, k, omega, y0))

# Plot (X,Y) using the Runge-Kutta 4
plt.plot(z_values[:, 0], z_values[:, 1], label='Particle Path')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Particle Path in 2D System, Runge-Kutta 4')
plt.grid()
plt.legend()
plt.show()
"""



# Finn løsninger for (x, y) og (X, Y)
x_values = z_values[:, 0]
y_values = z_values[:, 1]

X_values = k * x_values - omega * t_values
Y_values = k * y_values

# Plot particle paths in (x, y) and (X, Y)
plt.figure(figsize=(12, 6))

# Plot in (x, y)
plt.subplot(1, 2, 1)
plt.plot(x_values, y_values, label='Particle Path (x, y)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Particle Path in (x, y)')
plt.grid()
plt.legend()

# Plot in (X, Y)
plt.subplot(1, 2, 2)
plt.plot(X_values, Y_values, label='Particle Path (X, Y)', color='orange')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')
plt.title('Particle Path in (X, Y)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()



def poincare_section(func, z0, t_span, h, X_target, epsilon=0.01, record_all_crossings=False,args=()):
    '''
    Funksjon som skal følge en partikkelbane
    Input
    Outputs
    '''
    t_start, t_end = t_span
    t_values = [t_start]
    z_values = [z0]
    y_at_section = []

    z = np.array(z0)

    while t_values[-1] < t_end:
        t = t_values[-1]
        k1 = h * func(t, z, *args)
        k2 = h * func(t + h / 2, z + k1 / 2, *args)
        k3 = h * func(t + h / 2, z + k2 / 2, *args)
        k4 = h * func(t + h, z + k3, *args)

        z_next = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_next = t + h

        # Check for crossing X_target
        if abs(z_next[0] - X_target) < epsilon:
            # Record Y-value at crossing
            y_cross = z_next[1]
            y_at_section.append(y_cross)
            
            if not record_all_crossings:
                break  # Stop at first crossing if flag is False

        # Append results
        t_values.append(t_next)
        z_values.append(z_next)
        z = z_next

    # Convert to numpy arrays for easier indexing
    t_values = np.array(t_values)
    z_values = np.array(z_values)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(z_values[:, 0], z_values[:, 1], label="Trajectory")
    plt.axvline(X_target, color='red', linestyle='--', label=f"Poincaré Section (X = {X_target})")
    if y_at_section:
        plt.scatter([X_target] * len(y_at_section), y_at_section, color='orange', label="Crossings", zorder=5)
    plt.xlabel("X(t)")
    plt.ylabel("Y(t)")
    plt.title("Particle Trajectory and Poincaré Section")
    plt.legend()
    plt.grid()
    plt.show()

    return t_values, z_values, y_at_section


t_values, z_values, y_at_section = poincare_section(
    eq_31, z0, t_span, h, X_target=0.3, epsilon=0.01, record_all_crossings=False, args=(M, k, omega)
)
print("First crossing Y:", y_at_section)

t_values, z_values, y_at_section = poincare_section(
    eq_31, z0, t_span, h, X_target=0.3, epsilon=0.01, record_all_crossings=True, args=(M, k, omega)
)
print("All crossings Y:", y_at_section)




# Set event to stop the solver
event_X_pi.terminal = True  # Stop when the event is triggered
event_X_pi.direction = 0  # Detect both directions

# Solve the system
solution = solve_ivp(transformed_system, t_span, Z0, args=(M, k, omega), t_eval=t, events=event_X_pi)

# Extract the Y-value at the stopping point (X = pi)
if solution.t_events[0].size > 0:
    t_cross = solution.t_events[0][0]
    Y_pi = solution.y_events[0][0][1]  # Y-value at X = pi
    print(f"X = pi is crossed at t = {t_cross:.5f} with Y_pi = {Y_pi:.5f}")
else:
    print("No crossing detected for X = pi.")

plt.plot(t, solution.y[0], label='X(t)')
plt.axhline(np.pi, color='red', linestyle='--', label='X = pi')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title('Evolution of X(t)')
plt.legend()
plt.grid()
plt.show()

# Plot X(t) to analyze its behavior
plt.figure(figsize=(8, 4))
plt.plot(t, solution.y[0], label="X(t)")
plt.axhline(np.pi, color="r", linestyle="--", label="X = pi")
plt.xlabel("Time (t)")
plt.ylabel("X(t)")
plt.title("Evolution of X(t)")
plt.legend()
plt.grid()
plt.show()



# Define the event function for detecting X = pi
def event_X_pi(t, z, M, k, omega):
    '''
    Funksjon som skal finne når X=pi
    Input
        t: en-dimensjonal vektor
        z: to dimensjonal vektor z=[x, y]
    Output
        X = pi
    '''

    X, _ = z
    return X - np.pi  # Detect when X = pi

event_X_pi.terminal = True  # Stop when the event is triggered

solution_findX = solve_ivp(transformed_system, t_span, Z0, t_eval=t, args=(M, k, omega))

if solution_findX.t_events[0].size > 0:
    t_cross = solution_findX.t_events[0][0]
    Y_pi = solution_findX.y_events[0][0][1]
    print(f"X = pi is crossed at t = {t_cross:.5f} with Y_pi = {Y_pi:.5f}")
else:
    print("No crossing detected for X = pi.")

plt.plot(t, solution_findX.y[0], label="X(t)")
plt.axhline(np.pi, color="r", linestyle="--", label="X = pi")
plt.xlabel("Time (t)")
plt.ylabel("X(t)")
plt.title("Evolution of X(t)")
plt.legend()
plt.grid()
plt.show()



t = solution_XY.t
X = solution_XY.y[0]
Y = solution_XY.y[1]

# Wrap X(t) to the range [-2pi, 2pi]
X_wrapped = (X + 2 * np.pi) % (4 * np.pi) - 2 * np.pi

# Detect crossings of X = pi
crossings = np.where((X[:-1] < np.pi) & (X[1:] >= np.pi))[0]

# Crossing
if len(crossings) > 0:
    
    t1, t2 = t[crossings[0]], t[crossings[0] + 1]
    X1, X2 = X_wrapped[crossings[0]], X_wrapped[crossings[0] + 1]
    t_cross = t1 + (np.pi - X1) * (t2 - t1) / (X2 - X1)

    Y1, Y2 = Y[crossings[0]], Y[crossings[0] + 1]
    Y_cross = Y1 + (np.pi - X1) * (Y2 - Y1) / (X2 - X1)

    print(f'X = pi is crossed at t = {t_cross:.5f} with Y = {Y_cross:.5f}')
else:
    print('No crossing X = pi.')


# Plot X(t) with crossing marker
plt.figure(figsize=(6, 3))
plt.plot(t, X, label='X(t)')
plt.axvline(np.pi, color="r", linestyle="--", label="X = pi")
if len(crossings) > 0:
    plt.scatter([t_cross], [np.pi], color='red', label='Crossing Point')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title('X(t)')
plt.legend()
plt.grid()
plt.show()

# Plot the trajectory in (X, Y) space
plt.figure(figsize=(6, 3))
plt.plot(X, Y, label='Particle path (X, Y)')
plt.axvline(np.pi, color="r", linestyle="--", label="X = pi")
if len(crossings) > 0:
    plt.scatter([np.pi], [Y_cross], color="red", label="Crossing at X = pi")
plt.xlabel('X(t)')
plt.ylabel('Y(t)')
plt.title('Particle path (X, Y)')
plt.legend()
plt.grid()
plt.show()