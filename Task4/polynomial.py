import numpy as np
from scipy.optimize import fsolve





def cubic_polynomial_coefficients(x0, y0, xf, yf):
    # Define the system of equations
    def equations(coefficients):
        a, b, c, d = coefficients
        eq1 = a * x0**3 + b * x0**2 + c * x0 + d - y0
        eq2 = a * xf**3 + b * xf**2 + c * xf + d - yf
        eq3 = 3 * a * x0**2 + 2 * b * x0 + c
        eq4 = 3 * a * xf**2 + 2 * b * xf + c
        return [eq1, eq2, eq3, eq4]

    # Initial guess for coefficients (can be adjusted based on specific requirements)
    initial_guess = [1, 1, 1, 1]

    # Solve the system of equations to find coefficients
    coefficients = fsolve(equations, initial_guess)

    return coefficients

def cubic_polynomial(x, coefficients):
    a, b, c, d = coefficients
    result = np.zeros(x.shape[0])
    for tt in range(result.shape[0]):
        result[tt] = a * x[tt]**3 + b * x[tt]**2 + c * x[tt] + d
    return result

def quintic_polynomial_coefficients(x0, y0, xf, yf):
    # Define the system of equations
    def equations(coefficients):
        a, b, c, d, e, f = coefficients
        eq1 = a*x0**5 + b*x0**4 + c*x0**3 + d*x0**2 + e*x0 + f - y0
        eq2 = a*xf**5 + b*xf**4 + c*xf**3 + d*xf**2 + e*xf + f - yf
        eq3 = 5*a*x0**4 + 4*b*x0**3 + 3*c*x0**2 + 2*d*x0 + e
        eq4 = 5*a*xf**4 + 4*b*xf**3 + 3*c*xf**2 + 2*d*xf + e
        eq5 = 20*a*x0**3 + 12*b*x0**2 + 6*c*x0 + 2*d
        eq6 = 20*a*xf**3 + 12*b*xf**2 + 6*c*xf + 2*d

        return [eq1, eq2, eq3, eq4, eq5, eq6]

    # Initial guess for coefficients (can be adjusted based on specific requirements)
    initial_guess = [1, 1, 1, 1, 1, 1]

    # Solve the system of equations to find coefficients
    coefficients = fsolve(equations, initial_guess)

    return coefficients

def quintic_polynomial(x, coefficients):
    a, b, c, d, e, f = coefficients
    result = np.zeros(x.shape[0])
    for tt in range(result.shape[0]):
        result[tt] = a*x[tt]**5 + b*x[tt]**4 + c*x[tt]**3 + d*x[tt]**2 + e*x[tt] + f
    return result
