

# Finite-time LQR for tracking
# Lorenzo Sforni
# Bologna, 20/11/2023
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import solver as slv
import dynamics as dyn
import cost as cst
import root as rt
import polynomial as poly
from opt_traj import optimal_trajectory
import math



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




max_iters = int(10)

#########################
##Trajectory Parameters##
#########################

initial_xx = np.array([1500, 0*DEG_TO_RAD, 0*DEG_TO_RAD, 0])
final_xx = np.array([1500, 0*DEG_TO_RAD, 0*DEG_TO_RAD, 0])
sim_length = 12 #final time in seconds
transition_length = 3
double_transition = True
middle_xx = np.array([1600, 0*DEG_TO_RAD, 7*DEG_TO_RAD, 0])

ns = 4
ni = 3
dt = 0.01
TT = int(sim_length/dt)
armijo_maxiters = 10
term_cond = 1e-6
stepsize0 = 1
cc = 0.5
beta = 0.7





step_reference = False
cubic = True
quintic = False
to_compute = True


if not (step_reference or cubic or quintic):
    print("Please choose a reference trajectory type (in trajectory parameters)")
    exit()



if to_compute:
    ######################################
    # Reference curve
    ######################################

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))



    if not step_reference:
        if double_transition:
            coefficients1 = []
            coefficients2 = []
            transition_values1 = []
            transition_values2 = []


            t01 = int(sim_length/dt/3) - int(transition_length/dt/2)
            tf1 = int(sim_length/dt/3) + int(transition_length/dt/2)
            t02 = int(2*sim_length/dt/3) - int(transition_length/dt/2)
            tf2 = int(2*sim_length/dt/3) + int(transition_length/dt/2)
            time_values1 = np.linspace(t01, tf1, tf1 - t01 + 1)
            time_values2 = np.linspace(t02, tf2, tf2 - t02 + 1)


            if cubic:
                coefficients1.append([poly.cubic_polynomial_coefficients(t01, initial_xx[i], tf1, middle_xx[i]) for i in range(ns)])
                coefficients1 = np.array(coefficients1).reshape((ns,4))

                transition_values1.append([poly.cubic_polynomial(time_values1, coefficients1[i]) for i in range(ns)])
                transition_values1 = np.array(transition_values1).reshape((4,tf1-t01+1))

                coefficients2.append([poly.cubic_polynomial_coefficients(t02, middle_xx[i], tf2, final_xx[i]) for i in range(ns)])
                coefficients2 = np.array(coefficients2).reshape((ns,4))

                transition_values2.append([poly.cubic_polynomial(time_values2, coefficients2[i]) for i in range(ns)])
                transition_values2 = np.array(transition_values2).reshape((4,tf2-t02+1))

            if quintic:
                coefficients1.append([poly.quintic_polynomial_coefficients(t01, initial_xx[i], tf1, middle_xx[i]) for i in range(ns)])
                coefficients1 = np.array(coefficients1).reshape((ns,6))

                transition_values1.append([poly.quintic_polynomial(time_values1, coefficients1[i]) for i in range(ns)])
                transition_values1 = np.array(transition_values1).reshape((4,tf1-t01+1))

                coefficients2.append([poly.quintic_polynomial_coefficients(t02, middle_xx[i], tf2, final_xx[i]) for i in range(ns)])
                coefficients2 = np.array(coefficients2).reshape((ns,6))

                transition_values2.append([poly.quintic_polynomial(time_values2, coefficients2[i]) for i in range(ns)])
                transition_values2 = np.array(transition_values2).reshape((4,tf2-t02+1))

            for tt in range(TT):
                if tt < t01:
                    xx_ref[:,tt] = initial_xx
                if tt >= t01 and tt <= tf1:
                    xx_ref[:,tt] = transition_values1[:,tt - t01]
                if tt > tf1 and tt < t02:
                    xx_ref[:,tt] = middle_xx
                if tt >= t02 and tt <= tf2:
                    xx_ref[:,tt] = transition_values2[:,tt - t02]
                if tt > tf2:
                    xx_ref[:,tt] = final_xx
            for tt in range(TT):
                uu_ref[:,tt] = rt.findRoots(xx_ref[:,tt])

        else:
            coefficients = []
            transition_values = []

            t0 = int(sim_length/dt/2) - int(transition_length/dt/2)
            tf = int(sim_length/dt/2) + int(transition_length/dt/2)
            time_values = np.linspace(t0, tf, tf - t0 + 1)

            if cubic:
                coefficients.append([poly.cubic_polynomial_coefficients(t0, initial_xx[i], tf, final_xx[i]) for i in range(ns)])
                coefficients = np.array(coefficients).reshape((ns,4))

                transition_values.append([poly.cubic_polynomial(time_values, coefficients[i]) for i in range(ns)])
                transition_values = np.array(transition_values).reshape((4,tf-t0+1))
            if quintic:
                coefficients.append([poly.quintic_polynomial_coefficients(t0, initial_xx[i], tf, final_xx[i]) for i in range(ns)])
                coefficients = np.array(coefficients).reshape((ns,6))

                transition_values.append([poly.quintic_polynomial(time_values, coefficients[i]) for i in range(ns)])
                transition_values = np.array(transition_values).reshape((4,tf-t0+1))
            

            for tt in range(TT):
                if tt < t0:
                    xx_ref[:,tt] = initial_xx
                if tt >= t0 and tt <= tf:
                    xx_ref[:,tt] = transition_values[:,tt - t0]
                if tt > tf:
                    xx_ref[:,tt] = final_xx
            for tt in range(TT):
                uu_ref[:,tt] = rt.findRoots(xx_ref[:,tt])
            


    else:
        T_mid = int((TT/2))

        for tt in range(TT):
            if tt < T_mid:
                xx_ref[:, tt] = initial_xx
                uu_ref[:, tt] = rt.findRoots(initial_xx)
            else:
                xx_ref[:, tt] = final_xx
                uu_ref[:, tt] = rt.findRoots(final_xx)


    xx,uu,JJ = optimal_trajectory(xx_ref, uu_ref, initial_xx, dt, max_iters)


else:
    uu = np.load("Project/Data/uu.npy")
    xx = np.load("Project/Data/xx.npy")





#############################
# Model Predictive Control
#############################
perturbation = np.array([
    np.random.uniform(-100, 100),   # First value from 0 to 500
    np.random.uniform(-5 * DEG_TO_RAD, 5 * DEG_TO_RAD),     # Second value from 0 to 1
    np.random.uniform(-5 * DEG_TO_RAD, 5 * DEG_TO_RAD),    # Third value from 0 to 10
    np.random.uniform(-5 * DEG_TO_RAD, 5 * DEG_TO_RAD)     # Fourth value from -5 to 5
])

# perturbation1 = np.array([10, 1 * DEG_TO_RAD, -1 * DEG_TO_RAD, 1 * DEG_TO_RAD])
# perturbation2 = np.array([50, 3 * DEG_TO_RAD, -2 * DEG_TO_RAD, 2 * DEG_TO_RAD])
# perturbation3 = np.array([100, -5 * DEG_TO_RAD, 5 * DEG_TO_RAD, -3 * DEG_TO_RAD])
# perturbation4 = np.array([200, 10 * DEG_TO_RAD, -8 * DEG_TO_RAD, 7 * DEG_TO_RAD])



xx_opt = np.zeros((ns, TT))
uu_opt = np.zeros((ni, TT))

xx_real_opt = np.zeros((ns, TT))
uu_real_opt = np.zeros((ni, TT))

xx_opt = xx[:,:,-1]
uu_opt = uu[:,:,-1] #optimal input for nominal dynamics in order to obtain optimal state (xx_opt)

xx0 = initial_xx + perturbation #set the initial condition
xx_real_opt[:,0] = xx0 #set the first measured value to the initial condition




for tt in range(TT-1):

    if tt%5 == 0: # print every 5 time instants
      print('LQR:\t t = {}'.format(tt))

    # System evolution - real with optimal control input (open-loop)
    uu_real_opt[:,tt] = uu_opt[:,tt]
    xx_real_opt[:,tt+1] = dyn.real_dynamics(xx_real_opt[:,tt], uu_real_opt[:,tt], dt)[0]
   



# Constraints on input and output

x_limits = np.zeros((ns, 2))
x_limits[0] = np.array([0, 2000])
x_limits[1] = np.array([-10 * DEG_TO_RAD, 10 * DEG_TO_RAD])
x_limits[2] = np.array([-10 * DEG_TO_RAD, 10 * DEG_TO_RAD])
x_limits[3] = np.array([-10 * DEG_TO_RAD, 10 * DEG_TO_RAD])

u_limits = np.zeros((ni, 2))
u_limits[0] = np.array([-20, 20])
u_limits[1] = np.array([-20, 20])
u_limits[2] = np.array([-20, 20])

# Prediction Parameters

Tpred = 5
QQ = np.diag([1,1,1,1])
QQT = QQ
RR = np.diag([1,1,1])

xx_real_mpc = np.zeros((ns,TT))
uu_real_mpc = np.zeros((ni,TT))
du_real_mpc = np.zeros((ni,TT))
xx_mpc = np.zeros((ns, Tpred, TT))
dx_mpc = np.zeros((ns, Tpred, TT))
uu_mpc = np.zeros((ni, Tpred, TT))
du_mpc = np.zeros((ni, Tpred, TT))
AAt = np.zeros((ns,ns, Tpred))
BBt = np.zeros((ns,ni, Tpred))


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


uu_mpc[:, :Tpred, 0]= uu_opt[:, :Tpred]   #initialize the first mpc sequence of input as the optimal one
xx_real_mpc[:,0] = xx0.squeeze()          #set the first value of real mpc as the initial conditio






for tt in range(TT-Tpred):
    # System evolution - real with MPC

    xx_t_mpc = xx_real_mpc[:,tt] # get initial condition at time t
   
    deltax = xx_t_mpc - xx_opt[:,tt]
    xx_dummy = xx_t_mpc
    for kk in range(Tpred):
        xx_dummy, AAt[:,:,kk], BBt[:,:, kk] = dyn.nominal_dynamics(xx_dummy, uu_mpc[:, kk, tt], dt)
       
    # Solve MPC problem - apply first input

    if tt%5 == 0: # print every 5 time instants
      print('MPC:\t t = {}'.format(tt))
     
    du_real_mpc[:,tt], dx_mpc[:,:,tt], du_mpc[:,:,tt+1] = slv.linear_mpc(AAt, BBt, QQ, RR, QQT, deltax, x_limits, u_limits, Tpred)
    uu_real_mpc[:,tt] = uu_opt[:, tt] + du_real_mpc[:,tt]
    uu_mpc[:,:,tt+1] = uu_opt[:,tt:tt+Tpred] + du_mpc[:,:,tt+1]
    xx_real_mpc[:,tt+1] = dyn.real_dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt], dt)[0]


for tt in range(TT):
    if tt > 0:
        position_centre[:,tt] = position_centre[:,tt-1] + [dt * xx_real_mpc[0,tt] * np.cos(xx_real_mpc[2,tt] - xx_real_mpc[1,tt]), dt * xx_real_mpc[0,tt] * np.sin(xx_real_mpc[2,tt] - xx_real_mpc[1,tt])]
        position_back[:,tt] = position_centre[:,tt] - [length*np.cos(xx_real_mpc[2,tt]), length*np.sin(xx_real_mpc[2,tt])] 
        position_front[:,tt] = position_centre[:,tt] + [length*np.cos(xx_real_mpc[2,tt]), length*np.sin(xx_real_mpc[2,tt])] 

        position_centre_ref[:,tt] = position_centre_ref[:,tt-1] + [dt * xx[0,tt,-1] * np.cos(xx[2,tt,-1] - xx[1,tt,-1]), dt * xx[0,tt,-1] * np.sin(xx[2,tt,-1] - xx[1,tt,-1])]
        position_back_ref[:,tt] = position_centre_ref[:,tt] - [length*np.cos(xx[2,tt,-1]), length*np.sin(xx[2,tt,-1])] 
        position_front_ref[:,tt] = position_centre_ref[:,tt] + [length*np.cos(xx[2,tt,-1]), length*np.sin(xx[2,tt,-1])] 
   
   
np.save("Project/Data/position_centre.npy", position_centre)
np.save("Project/Data/position_back.npy", position_back)
np.save("Project/Data/position_front.npy", position_front)
np.save("Project/Data/position_centre_ref.npy", position_centre_ref)
np.save("Project/Data/position_back_ref.npy", position_back_ref)
np.save("Project/Data/position_front_ref.npy", position_front_ref)
np.save("Project/Data/TT.npy",TT)
np.save("Project/Data/dt.npy",dt)


#######################################
# Plots
#######################################

error = xx_opt - xx_real_mpc


tt_hor = range(TT-5)

fig1, axs1 = plt.subplots(4, 1, sharex='all')

axs1[0].plot(tt_hor, xx_opt[0,:-5], 'b--', linewidth=2, label="Optimal")
axs1[0].plot(tt_hor, xx_real_mpc[0,:-5], "r", linewidth=2, label=f"MPC")
axs1[0].grid()
axs1[0].set_ylabel("$x_1$")
axs1[0].legend(fontsize= 12, loc="upper right")

axs1[1].plot(tt_hor, xx_opt[1,:-5], 'b--', linewidth=2, label="Optimal")
axs1[1].plot(tt_hor, xx_real_mpc[1,:-5], "r", linewidth=2, label=f"MPC")
axs1[1].grid()
axs1[1].set_ylabel("$x_2$")
axs1[1].legend(fontsize= 12, loc="upper right")

axs1[2].plot(tt_hor, xx_opt[2,:-5], 'b--', linewidth=2, label="Optimal")
axs1[2].plot(tt_hor, xx_real_mpc[2,:-5], "r", linewidth=2, label=f"MPC")
axs1[2].grid()
axs1[2].set_ylabel("$x_3$")
axs1[2].legend(fontsize= 12, loc="upper right")

axs1[3].plot(tt_hor, xx_opt[3,:-5], 'b--', linewidth=2, label="Optimal")
axs1[3].plot(tt_hor, xx_real_mpc[3,:-5], "r", linewidth=2, label=f"MPC")
axs1[3].grid()
axs1[3].set_ylabel("$x_4$")
axs1[3].legend(fontsize= 12, loc="upper right")





fig1.suptitle("State Trajectory")
fig1.align_ylabels(axs1)
plt.show()




fig2, axs2 = plt.subplots(3, 1, sharex='all')

axs2[0].plot(tt_hor, uu_real_opt[0,:-5], 'b--', linewidth=2, label="Optimal")
axs2[0].plot(tt_hor, uu_real_mpc[0,:-5], "r", linewidth=2, label=f"MPC")
axs2[0].grid()
axs2[0].set_ylabel("$u_1$")
axs2[0].legend(fontsize= 12, loc="upper right")

axs2[1].plot(tt_hor, uu_real_opt[1,:-5], 'b--', linewidth=2, label="Optimal")
axs2[1].plot(tt_hor, uu_real_mpc[1,:-5], "r", linewidth=2, label=f"MPC")
axs2[1].grid()
axs2[1].set_ylabel("$u_2$")
axs2[1].legend(fontsize= 12, loc="upper right")

axs2[2].plot(tt_hor, uu_real_opt[2,:-5], 'b--', linewidth=2, label="Optimal")
axs2[2].plot(tt_hor, uu_real_mpc[2,:-5], "r", linewidth=2, label=f"MPC")
axs2[2].grid()
axs2[2].set_ylabel("$u_3$")
axs2[2].legend(fontsize= 12, loc="upper right")





fig2.suptitle("Input Trajectory")
fig2.align_ylabels(axs2)
plt.show()


fig3, axs3 = plt.subplots(4, 1, sharex='all')

axs3[0].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
axs3[1].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
axs3[2].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
axs3[3].axhline(y=0, color='red', linestyle='--', label='Reference Axis')

axs3[0].plot(tt_hor, error[0,:-5], linewidth=2, label=f"MPC")
axs3[0].grid()
axs3[0].set_ylabel("$e_1$")
axs3[0].legend(fontsize= 12, loc="upper right")

axs3[1].plot(tt_hor, error[1,:-5], linewidth=2, label=f"MPC")
axs3[1].grid()
axs3[1].set_ylabel("$e_2$")
axs3[1].legend(fontsize= 12, loc="upper right")

axs3[2].plot(tt_hor, error[2,:-5], linewidth=2, label=f"MPC")
axs3[2].grid()
axs3[2].set_ylabel("$e_3$")
axs3[2].legend(fontsize= 12, loc="upper right")


axs3[2].plot(tt_hor, error[2,:-5], "r", linewidth=2, label=f"MPC")
axs3[2].grid()
axs3[2].set_ylabel("$e_3$")
axs3[2].legend(fontsize= 12, loc="upper right")



fig3.suptitle("error")
fig3.align_ylabels(axs3)
plt.show()














# np.save("Project/Data/uu_mpc_dyn4.npy", uu_real_mpc)
# np.save("Project/Data/xx_mpc_dyn4.npy", xx_real_mpc)
# np.save("Project/Data/error_mpc_dyn4.npy", error)



# xx_plot1 = np.load("Project/Data/xx_mpc_dyn1.npy")
# xx_plot2 = np.load("Project/Data/xx_mpc_dyn2.npy")
# xx_plot3 = np.load("Project/Data/xx_mpc_dyn3.npy")
# xx_plot4 = np.load("Project/Data/xx_mpc_dyn4.npy")
# xx_plot = np.array([xx_plot1, xx_plot2, xx_plot3, xx_plot4])

# uu_plot1 = np.load("Project/Data/uu_mpc_dyn1.npy")
# uu_plot2 = np.load("Project/Data/uu_mpc_dyn2.npy")
# uu_plot3 = np.load("Project/Data/uu_mpc_dyn3.npy")
# uu_plot4 = np.load("Project/Data/uu_mpc_dyn4.npy")
# uu_plot = np.array([uu_plot1, uu_plot2, uu_plot3, uu_plot4])


# error1 = np.load("Project/Data/error_mpc_dyn1.npy")
# error2 = np.load("Project/Data/error_mpc_dyn2.npy")
# error3 = np.load("Project/Data/error_mpc_dyn3.npy")
# error4 = np.load("Project/Data/error_mpc_dyn4.npy")
# error_plot = np.array([error1, error2, error3, error4])

# init_cond = np.array([[10,1,-1,1], [50,3,-2,2], [100,-5,5,-3], [200,10,-8,7]])

# tt_hor = range(TT-5)

# fig1, axs1 = plt.subplots(4, 1, sharex='all')

# axs1[0].plot(tt_hor, xx_opt[0,:-5], 'b--', linewidth=2, label="Optimal")
# axs1[1].plot(tt_hor, xx_opt[1,:-5], 'b--', linewidth=2, label="Optimal")
# axs1[2].plot(tt_hor, xx_opt[2,:-5], 'b--', linewidth=2, label="Optimal")
# axs1[3].plot(tt_hor, xx_opt[3,:-5], 'b--', linewidth=2, label="Optimal")


# for kk in range(0,4):
#     # axs1[0].plot(tt_hor, xx_real_opt[0,:-5], 'g--', linewidth=2)
#     axs1[0].plot(tt_hor, xx_plot[kk,0,:-5], linewidth=2, label=f"MPC, {kk*10 +10}% error")
#     axs1[0].grid()
#     axs1[0].set_ylabel("$x_1$")
#     axs1[0].legend(fontsize= 12, loc="upper right")

#     # axs1[1].plot(tt_hor, xx_real_opt[1,:-5], 'g--', linewidth=2)
#     axs1[1].plot(tt_hor, xx_plot[kk,1,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs1[1].grid()
#     axs1[1].set_ylabel("$x_2$")
#     axs1[1].legend(fontsize= 12, loc="upper right")

#     # axs1[2].plot(tt_hor, xx_real_opt[2,:-5], 'g--', linewidth=2)
#     axs1[2].plot(tt_hor, xx_plot[kk,2,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs1[2].grid()
#     axs1[2].set_ylabel("$x_3$")
#     axs1[2].legend(fontsize= 12, loc="upper right")

#     # axs1[3].plot(tt_hor, xx_real_opt[3,:-5], 'g--', linewidth=2)
#     axs1[3].plot(tt_hor, xx_plot[kk,3,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs1[3].grid()
#     axs1[3].set_ylabel("$x_4$")
#     axs1[3].legend(fontsize= 12, loc="upper right")

# fig1.suptitle("State Trajectory")
# fig1.align_ylabels(axs1)
# plt.show()






# fig2, axs2 = plt.subplots(3, 1, sharex='all')
# axs2[0].plot(tt_hor, uu_real_opt[0,:-5],'g--', linewidth=2, label="Optimal")
# axs2[1].plot(tt_hor, uu_real_opt[1,:-5],'g--', linewidth=2, label="Optimal")
# axs2[2].plot(tt_hor, uu_real_opt[2,:-5],'g--', linewidth=2, label="Optimal")


# for kk in range(0,4):
#     axs2[0].plot(tt_hor, uu_plot[kk,0,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs2[0].grid()
#     axs2[0].set_ylabel('$u_1$')
#     axs2[0].set_xlabel('time')
#     axs2[0].legend(fontsize= 12, loc="upper right")

#     axs2[1].plot(tt_hor, uu_plot[kk,1,:-5], linewidth=2,  label=f"MPC, {kk*10+10}% error")
#     axs2[1].grid()
#     axs2[1].set_ylabel('$u_2$')
#     axs2[1].set_xlabel('time')
#     axs2[1].legend(fontsize= 12, loc="upper right")

#     axs2[2].plot(tt_hor, uu_plot[kk,2,:-5], linewidth=2,  label=f"MPC, {kk*10+10}% error")
#     axs2[2].grid()
#     axs2[2].set_ylabel('$u_3$')
#     axs2[2].set_xlabel('time')
#     axs2[2].legend(fontsize= 12, loc="upper right")

# fig2.suptitle("Input trajectory")
# fig2.align_ylabels(axs2)
# plt.show()



# fig3, axs3 = plt.subplots(4, 1, sharex='all')

# axs3[0].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
# axs3[1].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
# axs3[2].axhline(y=0, color='red', linestyle='--', label='Reference Axis')
# axs3[3].axhline(y=0, color='red', linestyle='--', label='Reference Axis')



# for kk in range(0,4):
#     axs3[0].plot(tt_hor, error_plot[kk,0,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs3[0].grid()
#     axs3[0].set_ylabel("$e_1$")
#     axs3[0].legend(fontsize= 12, loc="upper right")

#     axs3[1].plot(tt_hor, error_plot[kk,1,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs3[1].grid()
#     axs3[1].set_ylabel("$e_2$")
#     axs3[1].legend(fontsize= 12, loc="upper right")

#     axs3[2].plot(tt_hor, error_plot[kk,2,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs3[2].grid()
#     axs3[2].set_ylabel("$e_3$")
#     axs3[2].legend(fontsize= 12, loc="upper right")

#     axs3[3].plot(tt_hor, error_plot[kk,3,:-5], linewidth=2, label=f"MPC, {kk*10+10}% error")
#     axs3[3].grid()
#     axs3[3].set_ylabel("$e_4$")
#     axs3[3].legend(fontsize= 12, loc="upper right")

# fig1.suptitle("Error")
# fig1.align_ylabels(axs1)
# plt.show()














