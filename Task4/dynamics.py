import numpy as np






############## Function of Dynamics ################

def nominal_dynamics(xx, uu, dt):


################### Nominal Parameters ##################

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

   
    """
        Dynamics of a discrete-time supersonic aircraft

        Args
        - xx : vector in R^4 representing the state at time t
        - uu : vector in R^3 representing the input at time t

        Return
        - xx_plus : The state at time t+1
        - At      : Gradient of the discrete f(x,u) wrt x, evaluated at (xx,uu)
        - Bt      : Gradient of the discrete f(x,u) wrt u, evaluated at (xx,uu)
       
       
    """

    x1=float(xx[0])
    x2=float(xx[1])
    x3=float(xx[2])
    x4=float(xx[3])
    u1=float(uu[0])
    u2=float(uu[1])
    u3=float(uu[2])
   
   
    ################## NEW STATE EVALUATION ###############
    xx_plus = np.zeros((4,1))
    xx_plus[0] = x1 + dt*((rho*x1*x1)*(CT*u1*np.cos(x2) - CD)/(2*m) - g*np.sin(x3-x2))
    xx_plus[1] = x2 + dt*(x4 - (rho*x1)*(CT*u1*np.sin(x2) + CL + b_11*u2 + b_12*u3)/(2*m) + (g*np.cos(x3-x2))/x1)
    xx_plus[2] = x3 + dt*(x4)
    xx_plus[3] = x4 + dt*((rho*x1*x1)*(CM + b_21*u2 + b_22*u3)/(2*J))
  
   
   
    ################## LINEARIZATION AND DISCRETIZATION OF THE MATRIX A AND B   ###############
   
    A = np.array([[ (rho*x1)*(CT*u1*np.cos(x2) - CD)/m, -(rho*x1*x1)*(CT*u1*np.sin(x2) + CD)/(2*m) + g*np.cos(x3-x2), -g*np.cos(x3-x2) , 0.0],
                  [ -rho*(CT*u1*np.sin(x2) + CL + b_11*u2 + b_12*u3)/(2*m) - (g*np.cos(x3 - x2))/(x1*x1) ,-(rho*x1*CT*u1*np.cos(x2))/(2*m) + (g*np.sin(x3-x2))/x1 , -(g*np.sin(x3-x2))/x1 , 1 ],
                  [0.0, 0.0, 0.0, 1.0],
                  [rho*x1*(CM + b_21*u2 + b_22*u3)/J, 0.0, 0.0 , 0.0]])
   
    I = np.eye(4) # Identity matrix 4x4
   
    At = I + dt*A # Discretized version of the linearized matrix (gradient of f(x,u) wrt x)
   
    B = np.array([[rho*x1*x1*np.cos(x2)*CT/(2*m), 0, 0],
                  [-rho*x1*CT*np.sin(x2)/(2*m), -rho*x1*b_11/(2*m), -rho*x1*b_12/(2*m)],
                  [0, 0 ,0],
                  [0, rho*x1*x1*b_21/(2*J), rho*x1*x1*b_22/(2*J)]])
   
    Bt = dt*B     # Discretized version of the linearized matrix (gradient of f(x,u) wrt u)
   
    return xx_plus.squeeze(), At , Bt


def real_dynamics(xx, uu, dt):



    ################### Real Parameters ##################

    error_percentage = 0.00

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

    m += m * error_percentage
    J -= J * error_percentage
    g -= g * error_percentage
    rho -= rho * error_percentage
    CL += CL * error_percentage
    CD += CD * error_percentage
    CM -= CM * error_percentage
    CT += CT * error_percentage
    b_11 -= b_11 * error_percentage
    b_12 -= b_12 * error_percentage
    b_21 -= b_21 * error_percentage
    b_22 += b_22 * error_percentage

    
    """
        Dynamics of a discrete-time supersonic aircraft

        Args
        - xx : vector in R^4 representing the state at time t
        - uu : vector in R^3 representing the input at time t

        Return
        - xx_plus : The state at time t+1
        - At      : Gradient of the discrete f(x,u) wrt x, evaluated at (xx,uu)
        - Bt      : Gradient of the discrete f(x,u) wrt u, evaluated at (xx,uu)
       
       
    """

    x1=float(xx[0])
    x2=float(xx[1])
    x3=float(xx[2])
    x4=float(xx[3])
    u1=float(uu[0])
    u2=float(uu[1])
    u3=float(uu[2])
   
   
    ################## NEW STATE EVALUATION ###############
    xx_plus = np.zeros((4,1))
    xx_plus[0] = x1 + dt*((rho*x1*x1)*(CT*u1*np.cos(x2) - CD)/(2*m) - g*np.sin(x3-x2))
    xx_plus[1] = x2 + dt*(x4 - (rho*x1)*(CT*u1*np.sin(x2) + CL + b_11*u2 + b_12*u3)/(2*m) + (g*np.cos(x3-x2))/x1)
    xx_plus[2] = x3 + dt*(x4)
    xx_plus[3] = x4 + dt*((rho*x1*x1)*(CM + b_21*u2 + b_22*u3)/(2*J))
  
   
   
    ################## LINEARIZATION AND DISCRETIZATION OF THE MATRIX A AND B   ###############
   
    A = np.array([[ (rho*x1)*(CT*u1*np.cos(x2) - CD)/m, -(rho*x1*x1)*(CT*u1*np.sin(x2) + CD)/(2*m) + g*np.cos(x3-x2), -g*np.cos(x3-x2) , 0.0],
                  [ -rho*(CT*u1*np.sin(x2) + CL + b_11*u2 + b_12*u3)/(2*m) - (g*np.cos(x3 - x2))/(x1*x1) ,-(rho*x1*CT*u1*np.cos(x2))/(2*m) + (g*np.sin(x3-x2))/x1 , -(g*np.sin(x3-x2))/x1 , 1 ],
                  [0.0, 0.0, 0.0, 1.0],
                  [rho*x1*(CM + b_21*u2 + b_22*u3)/J, 0.0, 0.0 , 0.0]])
   
    I = np.eye(4) # Identity matrix 4x4
   
    At = I + dt*A # Discretized version of the linearized matrix (gradient of f(x,u) wrt x)
   
    B = np.array([[rho*x1*x1*np.cos(x2)*CT/(2*m), 0, 0],
                  [-rho*x1*CT*np.sin(x2)/(2*m), -rho*x1*b_11/(2*m), -rho*x1*b_12/(2*m)],
                  [0, 0 ,0],
                  [0, rho*x1*x1*b_21/(2*J), rho*x1*x1*b_22/(2*J)]])
   
    Bt = dt*B     # Discretized version of the linearized matrix (gradient of f(x,u) wrt u)
   
    return xx_plus.squeeze(), At , Bt