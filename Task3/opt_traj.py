

import numpy as np
import matplotlib.pyplot as plt
import dynamics
import math
from scipy.optimize import root
from scipy.optimize import fsolve
import cvxpy as cvx
import cost as cst


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

def optimal_trajectory(xx_ref, uu_ref, initial_xx, dt, max_iters):
    
    max_iters = int(10)

    #########################
    ##Trajectory Parameters##
    #########################

    
    ns = xx_ref.shape[0]
    ni = uu_ref.shape[0]
    TT = xx_ref.shape[1]

    armijo_maxiters = 10
    term_cond = 1e-6
    stepsize0 = 1
    cc = 0.5
    beta = 0.7

    QQ = np.diag((1e-2,5e2,5e2,1)) # Define QQ matrix of wheight for state error wrt desired traj
    QQT = QQ 
    RR = np.diag((1,1,1)) # Define RR matrix of weight for input error wrt desired traj


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
            temp_cost, dJ1[:,tt,kk], dJ2[:,tt,kk], ddJ11[:,:,tt,kk], ddJ12[:,:,tt,kk], ddJ22[:,:,tt,kk] = cst.stage_cost(xx[:, tt, kk], xx_ref[:, tt], uu[:, tt, kk], uu_ref[:, tt], QQ, RR)
            JJ[kk] += temp_cost


            
        temp_cost, dJT1[:, kk], ddJT11[:, :, kk] = cst.term_cost(xx[:, -1, kk], xx_ref[:, -1], QQT)
        JJ[kk] += temp_cost


        ##################################
        # Descent direction calculation
        ##################################

        lmbd = np.zeros((ns, TT, max_iters))
        dJ = np.zeros((ni, TT, max_iters))
        du = np.zeros((ni, TT, max_iters))
        descent = np.zeros(max_iters)
        descent_arm = np.zeros(max_iters)



        
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
        #############################


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

        ##################################
        # Stepsize selection - ARMIJO
        ##################################

        stepsizes = []  # list of stepsizes
        costs_armijo = []
        descent_arm = np.zeros(max_iters)

        stepsize = stepsize0
        lmbd_temp = cst.term_cost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

        for tt in reversed(range(TT-1)):  # integration backward in time

            lmbd_temp = At[:,:,tt].T@lmbd[:,tt+1,kk] + qq[:,tt]     # costate equation
            dJ_temp = Bt[:,:,tt].T@lmbd[:,tt+1,kk] + rr[:,tt]        # gradient of J wrt u
            
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()

            descent[kk] += deltau[:,tt].T@deltau[:,tt]
            descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt]
                
            
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
                JJ_temp += cst.stage_cost(xx_temp[:,tt], xx_ref[:,tt], uu_temp[:,tt], uu_ref[:,tt], QQ, RR)[0]
            JJ_temp  += cst.term_cost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]

            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(JJ_temp)  # save the cost associated to the stepsize
            if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
                # update the stepsize
                stepsize = beta*stepsize
            
            else:
                print("iteration = ", ii)
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
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt]# + KK[:,:,tt]@(xx_temp[:,tt] - xx[:,tt,kk] - stepsize*delta_x[:,tt])
            xx_temp[:,tt+1] = dynamics.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]
        
        uu[:,:,kk+1] = uu_temp
        xx[:,:,kk+1] = xx_temp


        ############################
        # Termination condition
        ############################

        print('Iter = {}\t Descent = {:.3e}\t Descent_arm = {:.3e} Cost = {:.3e}'.format(kk, descent[kk], -descent_arm[kk], JJ[kk]))

        

        if -descent_arm[kk] <= term_cond:
            exit = True
            for i in range(kk, max_iters):
                xx[:,:,i] = xx[:,:,kk]
                uu[:,:,i] = uu[:,:,kk]
                break 




    np.save("Project/Data/uu.npy", uu)
    np.save("Project/Data/xx.npy", xx)

    

    return xx, uu, JJ