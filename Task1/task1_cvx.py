#
# Finite-time LQR for tracking
# Lorenzo Sforni
# Bologna, 20/11/2023
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from LQR_affine_ltv_solver import ltv_LQR
import dynamics
import math
from scipy.optimize import root
import cvxpy as cvx

np.set_printoptions(suppress=True)

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (20,16)
plt.rcParams.update({'font.size': 22})

DEG_TO_RAD = math.pi/180
RAD_TO_DEG = 180/math.pi

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

def stage_cost(xx, xx_ref, uu, uu_ref, QQ, RR):

  xx = xx.reshape((-1, 1))
  xx_ref = xx_ref.reshape((-1, 1))
  uu = uu.reshape((-1, 1))
  uu_ref = uu_ref.reshape((-1, 1))

 
  JJ = 0.5*(xx - xx_ref).T@QQ@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RR@(uu - uu_ref)
 

  dJ1 = QQ@(xx - xx_ref)
  dJ2 = RR@(uu - uu_ref)

  ddJ11 = QQ
  ddJ12 = np.zeros((3, 4))
  ddJ22 = RR

 
  return JJ.squeeze(), dJ1.squeeze(), dJ2.squeeze(), ddJ11.squeeze(), ddJ12.squeeze(), ddJ22.squeeze()


def term_cost(xx, xx_ref, QQT):

  xx = xx.reshape((-1, 1))
  xx_ref = xx_ref.reshape((-1, 1))


  JJT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)
  dJT1 = QQT@(xx - xx_ref)
  ddJT11 = QQT


  return JJT.squeeze(), dJT1.squeeze(), ddJT11.squeeze()


max_iters = int(10)

#########################
##Trajectory Parameters##
#########################

initial_xx = np.array([1500, 0*DEG_TO_RAD, 0*DEG_TO_RAD, 0])
final_xx = np.array([1800, 0*DEG_TO_RAD, 8*DEG_TO_RAD, 0])
sim_length = 10 #final time in seconds

ns = 4
ni = 3
dt = 0.05
TT = int(sim_length/dt)
armijo_maxiters = 10
term_cond = 1e-6
stepsize0 = 1
cc = 0.5
beta = 0.7
closed_loop = False

QQ = np.diag((1e-2,5e2,5e2,1)) # Define QQ matrix of wheight for state error wrt desired traj
QQT = QQ 
RR = np.diag((1,1,1)) # Define RR matrix of weight for input error wrt desired traj




######################################
# Reference curve
######################################

xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))
T_mid = int((TT/2))

for tt in range(TT):
  if tt < T_mid:
    xx_ref[:, tt] = initial_xx
    uu_ref[:, tt] = findRoots(initial_xx)
  else:
    xx_ref[:, tt] = final_xx
    uu_ref[:, tt] = findRoots(final_xx)

####################
#Initial guess
####################


xx_init = np.zeros((ns,TT))
uu_init = np.zeros((ni,TT))
xx_init[:,0] = initial_xx   # xx_init = [xx[0], 0, 0, ... , 0] just the first value is given (initialization)
uu_init= uu_ref             # uu_init = [uu[1], uu[2], ... , uu[TT-1] ] the whole quasi static trajectory is given  

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))
uu = np.zeros((ni, TT, max_iters))
JJ = np.zeros(max_iters) # cost
dJ1 = np.zeros((ns, TT, max_iters)) # gradient of cost wrt x
dJ2 = np.zeros((ni, TT, max_iters)) # gradient of cost wrt u
ddJ11 = np.zeros((ns, ns, TT, max_iters)) # hessian of cost wrt x,x
ddJ12 = np.zeros((ni, ns, TT, max_iters)) # hessian of cost wrt x,u = u,x
ddJ22 = np.zeros((ni, ni, TT, max_iters)) # hessian of cost wrt u,u
JJT = np.zeros(max_iters) # terminal cost
dJT1 = np.zeros((ns, max_iters)) # gradient of term cost wrt x
ddJT11 = np.zeros((ns, ns, max_iters)) # hessian of term cost wrt x,x
descent = np.zeros(max_iters)
descent_arm = np.zeros(max_iters)
descent_norm = np.zeros(max_iters)
deltau_norm = []
lmbd = np.zeros((ns, TT, max_iters))
dJ = np.zeros((ni, TT, max_iters))





position_centre = np.zeros((2,TT))
position_back = np.zeros((2,TT))
position_front = np.zeros((2,TT))
position_centre_ref = np.zeros((2,TT))
position_back_ref = np.zeros((2,TT))
position_front_ref = np.zeros((2,TT))
position_centre[:,0] = [0,1000]
position_back[:,0] = [0,1000]
position_front[:,0] = [0,1000] 
position_centre_ref[:,0] = [0,1000]
position_back_ref[:,0] = [0,1000]
position_front_ref[:,0] = [0,1000]       
length = 400



######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

xx[:,:,0] = xx_init # set xx to the inizialization value
uu[:,:,0] = uu_init # set uu to the inizialization value






for kk in range(max_iters-1): #Start of algorithm for cost minimization
    #########################
    # Dynamics
    #########################

    At = np.zeros((ns, ns, TT))
    Bt = np.zeros((ns, ni, TT))

    for tt in range(TT-1):
        xx[:,tt+1, kk], At[:, :, tt], Bt[:, :, tt] = dynamics.dynamics(xx[:, tt, kk], uu[:, tt, kk], dt) #Evaluation of state-input trajectory and matrix linear approximation at iteration kk
        if tt > 0:
            position_centre[:,tt] = position_centre[:,tt-1] + [dt * xx[0,tt,kk] * np.cos(xx[2,tt,kk] - xx[1,tt,kk]), dt * xx[0,tt,kk] * np.sin(xx[2,tt,kk] - xx[1,tt,kk])]
            position_back[:,tt] = position_centre[:,tt] - [length*np.cos(xx[2,tt,kk]), length*np.sin(xx[2,tt,kk])] 
            position_front[:,tt] = position_centre[:,tt] + [length*np.cos(xx[2,tt,kk]), length*np.sin(xx[2,tt,kk])] 
            
            position_centre_ref[:,tt] = position_centre_ref[:,tt-1] + [dt * xx_ref[0,tt] * np.cos(xx_ref[2,tt] - xx_ref[1,tt]), dt * xx_ref[0,tt] * np.sin(xx_ref[2,tt] - xx_ref[1,tt])]
            position_back_ref[:,tt] = position_centre_ref[:,tt] - [length*np.cos(xx_ref[2,tt]), length*np.sin(xx_ref[2,tt])] 
            position_front_ref[:,tt] = position_centre_ref[:,tt] + [length*np.cos(xx_ref[2,tt]), length*np.sin(xx_ref[2,tt])] 

    ##############################
    # Cost
    ##############################


    JJ[kk] = 0


    #Evaluate the  cost for the actual state-input trajectory
    for tt in range(TT-1):
        temp_cost, dJ1[:,tt,kk], dJ2[:,tt,kk], ddJ11[:,:,tt,kk], ddJ12[:,:,tt,kk], ddJ22[:,:,tt,kk] = stage_cost(xx[:, tt, kk], xx_ref[:, tt], uu[:, tt, kk], uu_ref[:, tt], QQ, RR)
        JJ[kk] += temp_cost


        
    temp_cost, dJT1[:, kk], ddJT11[:, :, kk] = term_cost(xx[:, -1, kk], xx_ref[:, -1], QQT)
    JJ[kk] += temp_cost


    ##################################
    # Descent direction calculation
    ##################################





    
    # Affine terms (for tracking)
    Qin = np.zeros((ns, ns, TT))
    Rin = np.zeros((ni, ni, TT))
    Sin = np.zeros((ni, ns, TT))
    QinT = np.zeros((ns, ns))
    qq = np.zeros((ns,TT))
    rr = np.zeros((ni,TT))
    qqT = np.zeros((ns))

    Qin = ddJ11[:, :, :, kk]
    Rin = ddJ22[:, :, :, kk]
    Sin = ddJ12[:, :, :, kk]
    QinT = ddJT11[:, :, kk]

    qq = dJ1[:, :, kk]
    rr = dJ2[:, :, kk]
    qqT = dJT1[:, kk]

    ##############################
    # Solver
    ##############################

    # initial condition
    delta_x0 = np.array([0.0, 0.0, 0.0, 0.0]) #delta x0
    delta_u0 = np.array([0.0, 0.0, 0.0]) #delta u0

    if closed_loop:
      KK,sigma,_, _ = ltv_LQR(At, Bt, Qin, Rin, Sin, QinT, TT, delta_x0, qq, rr, qqT)[:4]


  # Compute descent direction using cvxpy
    
    delta_x = cvx.Variable((ns, TT))
    delta_u = cvx.Variable((ni, TT))
    cost = 0
    constr = []

    # FIll cost and constraint list
    constr += [delta_x[:,0] == 0]
    constr += [delta_u[:,0] == 0]

    for tt in range(TT-1):
        # Readibility purposes
        qqt = qq[:,tt]
        rrt = rr[:,tt]
        delta_xt = delta_x[:,tt]
        delta_ut = delta_u[:,tt]
        QQt = Qin[:,:,tt]
        RRt = Rin[:,:,tt]
        AAt = At[:,:,tt]
        BBt = Bt[:,:,tt]

        cost += qqt.T@delta_xt + rrt.T@delta_ut + 0.5*cvx.quad_form(delta_xt, QQt) + 0.5*cvx.quad_form(delta_ut, RRt)
        constr += [delta_x[:,tt+1] == AAt@delta_xt + BBt@delta_ut]

    cost += qqT.T@delta_x[:,-1] + 0.5*cvx.quad_form(delta_x[:,-1], QQT)

    problem = cvx.Problem(cvx.Minimize(cost), constr)
    problem.solve()

    deltax = delta_x.value
    deltau = delta_u.value
    deltau_norm.append(np.linalg.norm(deltau))

    ##################################
    # Stepsize selection - ARMIJO
    ##################################

    stepsizes = []  # list of stepsizes
    costs_armijo = []


    stepsize = stepsize0
    lmbd_temp = term_cost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

    for tt in reversed(range(TT-1)):  # integration backward in time

      lmbd_temp = At[:,:,tt].T@lmbd[:,tt+1,kk] + qq[:,tt]     # costate equation
      dJ_temp = Bt[:,:,tt].T@lmbd[:,tt+1,kk] + rr[:,tt]        # gradient of J wrt u
    
      lmbd[:,tt,kk] = lmbd_temp.squeeze()
      dJ[:,tt,kk] = dJ_temp.squeeze()

      descent[kk] += deltau[:,tt].T@deltau[:,tt]
      descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt]
    descent_norm[kk] = np.abs(descent_arm[kk])
    
        
    for ii in range(armijo_maxiters):

      # temp solution update

      xx_temp = np.zeros((ns,TT))
      uu_temp = uu_init.copy()

      xx_temp[:,0] = initial_xx.copy()
      

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt]
        xx_temp[:,tt+1] = dynamics.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        JJ_temp += stage_cost(xx_temp[:,tt], xx_ref[:,tt], uu_temp[:,tt], uu_ref[:,tt], QQ, RR)[0]
      JJ_temp  += term_cost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]

      stepsizes.append(stepsize)      # save the stepsize
      costs_armijo.append(JJ_temp)  # save the cost associated to the stepsize
      if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
          # update the stepsize
          stepsize = beta*stepsize
      
      else:
          print('Armijo stepsize = {:.3e}'.format(stepsize))
          break
    print("stepsizes = ", stepsizes)
    

    #################################################################
    # Update The input trajectory and evaluate new state trajectory
    #################################################################

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))  
    xx_temp[:,0] = initial_xx #Initial condition

    for tt in range(TT-1):
        if closed_loop:
          uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt] + KK[:,:,tt]@(xx_temp[:,tt] - xx[:,tt,kk] - stepsize*delta_x[:,tt])
        else:   
          uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt]
        xx_temp[:,tt+1] = dynamics.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]
    
    uu[:,:,kk+1] = uu_temp
    xx[:,:,kk+1] = xx_temp


   ############################
    # Termination condition
    ############################

    print('Iter = {}\t Descent_arm = {:.3e} Cost = {:.3e}'.format(kk, -descent_arm[kk], JJ[kk]))

    if -descent_arm[kk] <= term_cond:
      for i in range(kk, max_iters):
        xx[:,:,i] = xx[:,:,kk]
        uu[:,:,i] = uu[:,:,kk]
      break 

    ############################
    # # Armijo plot
    # ############################
    if False:
        
      steps = np.linspace(0,stepsize0,int(2e1))
      costs = np.zeros(len(steps))

      for ii in range(len(steps)):

        step = steps[ii]

        # temp solution update

        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = initial_xx

        for tt in range(TT-1):
          uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt]
          xx_temp[:,tt+1] = dynamics.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT-1):
          temp_cost = stage_cost(xx_temp[:,tt], xx_ref[:,tt], uu_temp[:,tt],  uu_ref[:,tt], QQ, RR)[0]
          JJ_temp += temp_cost

        temp_cost = term_cost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]
        JJ_temp += temp_cost

        costs[ii] = np.min(JJ_temp)
        


      plt.figure(1)
      plt.clf()

      plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
      plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
      # plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
      plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

      plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

      plt.grid()
      plt.xlabel('stepsize')
      plt.legend()
      plt.draw()

      plt.show()






np.save("Project/Data/position_centre.npy", position_centre)
np.save("Project/Data/position_back.npy", position_back)
np.save("Project/Data/position_front.npy", position_front)
np.save("Project/Data/position_centre_ref.npy", position_centre_ref)
np.save("Project/Data/position_back_ref.npy", position_back_ref)
np.save("Project/Data/position_front_ref.npy", position_front_ref)
np.save("Project/Data/TT.npy",TT)
np.save("Project/Data/dt.npy",dt)
np.save("Project/Data/x_traj", xx[:,:,-1])
 
#######################################
# Plots
#######################################

tt_hor = range(TT)

fig1, axs1 = plt.subplots(4, 1, sharex='all')
axs1[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2, label="$x_1$ reference")
axs1[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2, label="$x_2$ reference")
axs1[2].plot(tt_hor, xx_ref[2,:], 'g--', linewidth=2, label="$x_3$ reference")
axs1[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2, label="$x_4$ reference")

for kk in range(max_iters-1, max_iters) :


  axs1[0].plot(tt_hor, xx[0,:,kk], linewidth=2, label=f"$x_1^{kk}$ optimal")
  axs1[0].grid()
  axs1[0].set_ylabel('$x_1$')
  axs1[0].legend(fontsize= 12, loc="upper right")


  axs1[1].plot(tt_hor, xx[1,:,kk], linewidth=2, label=f"$x_2^{kk}$ optimal")
  axs1[1].grid()
  axs1[1].set_ylabel('$x_2$')
  axs1[1].legend(fontsize= 12, loc="upper right")


  axs1[2].plot(tt_hor, xx[2,:,kk], linewidth=2, label=f"$x_3^{kk}$ optimal")
  axs1[2].grid()
  axs1[2].set_ylabel('$x_3$')
  axs1[2].legend(fontsize= 12, loc="upper right")


  axs1[3].plot(tt_hor, xx[3,:,kk], linewidth=2, label=f"$x_4^{kk}$ optimal")
  axs1[3].grid()
  axs1[3].set_ylabel('$x_4$')
  axs1[3].legend(fontsize= 12, loc="upper right")

  fig1.align_ylabels(axs1)
  fig1.suptitle('State Trajectory', fontsize=25)
plt.show()

fig2, axs2 = plt.subplots(3, 1, sharex='all')
axs2[0].plot(tt_hor, uu_ref[0,:],'g--', linewidth=2, label="$u_1$ reference")
axs2[1].plot(tt_hor, uu_ref[1,:],'g--', linewidth=2, label="$u_2$ reference")
axs2[2].plot(tt_hor, uu_ref[2,:],'g--', linewidth=2, label="$u_3$ reference")

for kk in range(max_iters-1, max_iters) :
  axs2[0].plot(tt_hor, uu[0,:,kk], linewidth=2, label=f"$u_1^{kk}$ optimal")
  axs2[0].grid()
  axs2[0].set_ylabel('$u1$')
  axs2[0].set_xlabel('time')
  axs2[0].legend(fontsize= 12, loc="upper right")


  axs2[1].plot(tt_hor, uu[1,:,kk], linewidth=2, label=f"$u_2^{kk}$ optimal")
  axs2[1].grid()
  axs2[1].set_ylabel('$u2$')
  axs2[1].set_xlabel('time')
  axs2[1].legend(fontsize= 12, loc="upper right")


  axs2[2].plot(tt_hor, uu[2,:,kk], linewidth=2, label=f"$u_3^{kk}$ optimal")
  axs2[2].grid()
  axs2[2].set_ylabel('$u3$')
  axs2[2].set_xlabel('time')
  axs2[2].legend(fontsize= 12, loc="upper right")


  fig2.align_ylabels(axs2)
  fig2.suptitle('Input Trajectory', fontsize=25)
plt.show()


plt.figure()
plt.title("Cost", fontsize=25)
plt.plot(np.arange(max_iters), JJ)
plt.yscale("log")
plt.grid()
plt.xlabel("$k$")
plt.ylabel("$J(\\mathbf{u}^k)$")
plt.show()

plt.figure()
plt.title("Descent Norm", fontsize=25)
# plt.plot(np.arange(np.array(deltau_norm).shape[0]), np.array(deltau_norm))
plt.plot(np.arange(np.array(descent).shape[0]), descent)
plt.yscale("log")
plt.xlabel("$k$")
plt.ylabel("$||\\nabla J(\\mathbf{u}^k)||$")
plt.grid()
plt.show()




