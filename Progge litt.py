import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameter values
k = 2 * np.pi / 10
omega = np.sqrt(9.81 * k)
epsilon = 1
M = omega * epsilon

# Initial data
x0 = 0
y0 = -1
z0 = [x0, y0]

# Transformed system
X0 = k * x0 - omega * 0              # t0 = 0
Y0 = k * y0              
Z0 = [X0, Y0]

# Time span
t_span = (0, 10)                     # start and sluttid
t = np.linspace(0, 10, 5000)         

def eq_31(t, z, M, k, omega):
    '''
    Funksjon som løser ligning 3.1
    Input
      t: en-dimensjonal vektor
      z: to dimensjonal vektor z=[x, y]
    Output:
      to-dimensjonal vektor med løsningene [dx, dy]
    '''
    x, y = z 
    dx = M * np.exp(k * y) * np.cos(k * x - omega * t)
    dy = M * np.exp(k * y) * np.sin(k * x - omega * t)
    
    return np.array([dx, dy])

def transformed_system(t, z, M, k, omega, y0=Y0):
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

# Solution for the system in (x,y)
solution_xy = solve_ivp(eq_31, t_span, z0, args=(M, k, omega), t_eval=t, rtol=1e-10, atol=1e-12)
print(solution_xy)

# Solution for the transformed system
solution_XY = solve_ivp(transformed_system, t_span, Z0, t_eval=t, args=(M, k, omega))

# Plot (x,y)
plt.plot(solution_xy.y[0], solution_xy.y[1], label='Particle path (x,y)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Particle Path in 2D System (x,y)')
plt.grid()
plt.legend()
plt.show()

# Plot (X,Y)
plt.plot(solution_XY.y[0], solution_XY.y[1], label='Particle path (X,Y)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Particle Path in 2D System (X,Y)')
plt.grid()
plt.legend()
plt.show()

# Find excapt values
x = solution_XY.y[0]
y = solution_XY.y[1]
 
crossings = []                      # tom liste for å lagre krysning

for i in range(len(x) - 1):
    # Check if they cross, that happens if opposite signs
    if (x[i] - np.pi) * (x[i + 1] - np.pi) < 0:                                         
        t_cross = t[i] + (t[i + 1] - t[i]) * (np.pi - x[i]) / (x[i + 1] - x[i])         
        y_cross = y[i] + (y[i + 1] - y[i]) * (t_cross - t[i]) / (t[i + 1] - t[i])
        
        crossings.append((t_cross, y_cross))

# Display crossings
print("Poincaré crossings (t, Y):")
for crossing in crossings:
    print(crossing)

# Plot trajectories
plt.figure()
plt.plot(x, y, label='Particle Path (x, y)')
plt.scatter([np.pi] * len(crossings), [c[1] for c in crossings], color='red', label='Poincaré Crossings')
plt.axvline(np.pi, color='gray', linestyle='--', label='Poincaré Section (X = π)')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Particle Path with Poincaré Crossings')
plt.legend()
plt.grid()
plt.show()



'''
Plotte Y0 mot Y_pi
Bruke ulike y0 verdier
Lage to vektorerer
Bruke mye av det prinsippet som jeg har brukt nå
Bruke samme funksjon til å få ut (t,y) verdier for hvor de krysser et punkt (det røde)
Ønsker å lagre de nye y verdiene og plotte de mot et sett av y0 vektor


Kan være de innebygde funksjonene justerer steglengden ulikt for ulike verdier
Må denne måten må man sjekke at det punktet man lander på er nærmest det røde
Sjekke x[i]-pi og x[i+1]-pi tror jeg
'''



