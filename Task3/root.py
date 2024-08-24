import numpy as np
from scipy.optimize import root

m = 26.82
J = 595.9
g = 32.17
rho = 0.0011
CL = 0.5
CD = 1.59
CM = 0.5
CT = 3.75
b_11 = -3.575
b_12 = 0.065
b_21 = -1.3
b_22 = 6.5








def funcs(uu, x_des):

  x1 = x_des[0]
  x2 = x_des[1]
  x3 = x_des[2]
  x4 = x_des[3]

  u1=float(uu[0])
  u2=float(uu[1])
  u3=float(uu[2])
 
  ### Note that the state is "fixed", we want to vary the inputs.
 
  f1 = 0.5*rho*x1**2*CT*u1*np.cos(x2) - 0.5*CD*rho*x1**2 - m*g*np.sin(x3-x2)   # From dynamics xx_plus[0] take the equation multiplied by dt
  f2 = -1/m/x1*(0.5*rho*x1**2*(CT*u1*np.sin(x2)+CL) + 0.5*rho*x1**2*(b_11*u2+b_12*u3) -m*g*np.cos(x3-x2)) # From dynamics xx_plus[1] take the equation multiplied by dt
  f3 = 0.5*rho*x1**2*(CM+b_21*u2+b_22*u3) # From dynamics xx_plus[3] take the equation multiplied by dt
 
  ### Note that the function of  xx_plus[2] is not required since it does not depend on the inputs!
 
  return [f1, f2, f3]

def findRoots(x_des):
  initial_guess = np.array([0.0, 0.0, 0.0])  
  optimal_input = root(funcs, initial_guess, args=x_des)


#   ############### PRINT THE SOLUTION #######################
#   if optimal_input.success:
#     print("Root found:", optimal_input.x)
#   else:
#       print("Root-finding was unsuccessful. Message:", optimal_input.message)
 
  return np.array(optimal_input.x)
