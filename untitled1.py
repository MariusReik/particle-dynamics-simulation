import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameter values
k = 2 * np.pi / 10
omega = np.sqrt(9.81 * k)
epsilon = 1
M = omega * epsilon

# Time span
t_span = (0, 10)                     # start and sluttid
t = np.linspace(0, 10, 5000) 

def transformed_system(t, z, M, k, omega):
    '''
    Funksjon som transformerer systemet fra 'lille' x til 'store' X
    Input
      t: en-dimensjonal vektor
      z: to dimensjonal vektor z=[x, y]
    Outout
      to-dimensjonal vektor med løsningene [dX, dY]
    '''
    X, Y = z
    dX = k * M * np.exp(Y) * np.cos(X) - omega
    dY = k * M * np.exp(Y) * np.sin(X)
    
    return np.array([dX, dY])

# Empty lists to store Y0 and corresponding Y values for X=pi/2 and X=0
Y0_pi_half = []
Y_pi_half_list = []
Y0_zero = []
Y_0_crossings_list = []

# Initial data
x0 = np.pi
y0_values = np.linspace(-0.1, -2, 10)

for y0 in y0_values:
    z0 = [x0, y0]                                   # initial value
    
    # Solve the system
    solution = solve_ivp(transformed_system, t_span, z0, args=(M, k, omega), t_eval=t, rtol=1e-10, atol=1e-12)
    x = solution.y[0]
    y = solution.y[1]
    
    # Empty lists to store the crossing values
    pi_half_crossings = []
    zero_crossings = []
    
    for i in range(1, len(x) - 1):                  # start i i=1 for å trygt kunne sjekke x[i-1]
        # Check if crossing occurs at X = pi/2
        if (x[i] - np.pi/2) * (x[i + 1] - np.pi/2) < 0:
            t_cross = t[i] + (t[i + 1] - t[i]) * (np.pi/2 - x[i]) / (x[i + 1] - x[i])
            y_cross = y[i] + (y[i + 1] - y[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
            
            # Ensure this point is the closest to X=pi/2
            if abs(x[i] - np.pi/2) < abs(x[i - 1] - np.pi/2):   # Compare to the previous point
                pi_half_crossings.append(y_cross)               
                
        # Check if crossing occurs at X = 0
        if (x[i] - 0) * (x[i + 1] - 0) < 0:
            t_cross = t[i] + (t[i + 1] - t[i]) * (0 - x[i]) / (x[i + 1] - x[i])
            y_cross = y[i] + (y[i + 1] - y[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
            
            # Ensure this point is the closest to X=0
            if abs(x[i] - 0) < abs(x[i - 1] - 0):               # Compare to the previous point
                zero_crossings.append(y_cross)

    # If crossings are found, store corresponding Y0 and Y-values
    if pi_half_crossings:
        Y0_pi_half.extend([y0] * len(pi_half_crossings))
        Y_pi_half_list.extend(pi_half_crossings)

    if zero_crossings:
        Y0_zero.extend([y0] * len(zero_crossings))         
        Y_0_crossings_list.extend(zero_crossings)  
        
# Plot
plt.figure()
plt.plot(Y0_pi_half, Y_pi_half_list, 'o', label=r'$Y_{X=\pi/2}$')
plt.plot(Y0_zero, Y_0_crossings_list, 'x', label=r'$Y_{X=0}$')
plt.xlabel(r'$Y_0$ (Initial condition)')
plt.ylabel(r'$Y$ crossings')
plt.title(r'Relationship between $Y_0$ and crossings at $X=\pi/2$ and $X=0$')
plt.grid()
plt.legend()
plt.show()

    
