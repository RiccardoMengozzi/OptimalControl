import numpy as np
import cvxpy as cp


def ltv_affine_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
  LQR for LTV system with (time-varying) affine cost
 
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
 
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  if qqin is not None or rrin is not None or qqfin is not None:
    augmented = True

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
 
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0
 
  PP[:,:,-1] = QQf
  pp[:,-1] = qqf
 
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    # print("t = ", tt)
    # print("MMt_inv = ",MMt_inv)
    # print("PPtp = ",PPtp)
    # print("BBt = ", BBt)
    # print("SSt = ", SSt)
    # print("QQt = ", QQt)
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Evaluate KK
 
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    # for other purposes we could add a regularization step here...

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()

   
 

  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
    xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return KK, sigma, xxout, uuout, PP



def ltv_LQR(AA, BB, QQ, RR, QQf, TT):

  """
  LQR for LTV system with (time-varying) cost
 
  Args
    - AA (nn x nn (x TT)) matrix
    - BB (nn x mm (x TT)) matrix
    - QQ (nn x nn (x TT)), RR (mm x mm (x TT)) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
 
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array
    ns, lA = AA.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AA = AA[:,:,None]
    ns, lA = AA.shape[1:]

  try:  
    nu, lB = BB.shape[1:]
  except:
    BB = BB[:,:,None]
    ni, lB = BB.shape[1:]

  try:
      nQ, lQ = QQ.shape[1:]
  except:
      QQ = QQ[:,:,None]
      nQ, lQ = QQ.shape[1:]

  try:
      nR, lR = RR.shape[1:]
  except:
      RR = RR[:,:,None]
      nR, lR = RR.shape[1:]

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()


  if lA < TT:
      AA = AA.repeat(TT, axis=2)
  if lB < TT:
      BB = BB.repeat(TT, axis=2)
  if lQ < TT:
      QQ = QQ.repeat(TT, axis=2)
  if lR < TT:
      RR = RR.repeat(TT, axis=2)
 
  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
 
  PP[:,:,-1] = QQf
 
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
   
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
 
  # Evaluate KK
 
 
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
   
    KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

  return KK, PP
   
def linear_mpc(AA, BB, QQ, RR, QQf, xxt, x_limits, u_limits,  T_pred ):
    """
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - T: time (prediction) horizon

        Returns
          - u_t: input to be applied at t
          - xx, uu predicted trajectory

    """




    xxt = xxt.squeeze()

    ns, ni = BB.shape[:2]

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []

    for tt in range(T_pred-1):
        cost += cp.quad_form(xx_mpc[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt], RR)
        constr += [xx_mpc[:,tt+1] == AA[:,:,tt]@xx_mpc[:,tt] + BB[:,:,tt]@uu_mpc[:,tt]] # dynamics constraint
                # uu_mpc[0,tt] >= u_limits[0,0],
                # uu_mpc[0,tt] <= u_limits[0,1],

                # uu_mpc[1,tt] >= u_limits[1,0],
                # uu_mpc[1,tt] <= u_limits[1,1],

                # uu_mpc[2,tt] >= u_limits[2,0],
                # uu_mpc[2,tt] <= u_limits[2,1]]

                # xx_mpc[0,tt] >= x_limits[0,0],
                # xx_mpc[0,tt] <= x_limits[0,1]]

                # xx_mpc[1,tt] >= x_limits[1,0],
                # xx_mpc[1,tt] <= x_limits[1,1],

                # xx_mpc[2,tt] >= x_limits[2,0],
                # xx_mpc[2,tt] <= x_limits[2,1]]
               
                # xx_mpc[3,tt] >= x_limits[3,0],
                # xx_mpc[3,tt] <= x_limits[3,1]]


    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)
    constr += [xx_mpc[:,0] == xxt]
    constr += [xx_mpc[:,T_pred-1] == np.array([0.0,0.0,0.0,0.0])]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value